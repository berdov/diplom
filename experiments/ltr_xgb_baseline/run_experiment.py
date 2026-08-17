#!/usr/bin/env python3
"""Запуск XGBoost LambdaMART baseline для KuaiRand Protocol B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


KS = (5, 10, 20, 50)
QUERY_FEATURE_COLUMNS = [
    "user_history_length",
    "user_unique_items",
    "user_history_unique_ratio",
    "user_history_popularity_mean",
    "user_history_popularity_median",
    "user_history_popularity_max",
    "user_history_popularity_last",
    "user_history_span_days",
    "user_history_mean_gap_hours",
]
ITEM_FEATURE_COLUMNS = [
    "item_train_popularity",
    "log1p_item_train_popularity",
    "item_popularity_rank",
    "item_popularity_percentile",
]
RESULT_COLUMNS = [
    "run_id",
    "timestamp",
    "model",
    "branch",
    "git_commit",
    "protocol",
    "candidate_protocol",
    "evaluation_protocol",
    "train_candidate_protocol",
    "eval_candidate_protocol",
    "item_universe_size",
    "mask_seen_items",
    "protocol_version",
    "n_negatives",
    "seed",
    "features",
    "HR@5",
    "HR@10",
    "HR@20",
    "HR@50",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "NDCG@50",
    "Recall@5",
    "Recall@10",
    "Recall@20",
    "Recall@50",
    "train_time_sec",
    "inference_time_sec",
    "remote_artifact_path",
    "notes",
    "validation_HR@5",
    "validation_HR@10",
    "validation_HR@20",
    "validation_HR@50",
    "validation_NDCG@5",
    "validation_NDCG@10",
    "validation_NDCG@20",
    "validation_NDCG@50",
    "validation_Recall@5",
    "validation_Recall@10",
    "validation_Recall@20",
    "validation_Recall@50",
]


@dataclass
class QuerySpec:
    query_index: int
    split: str
    user_id: int
    target_item_id: int
    target_timestamp: int
    context_items: list[int]
    context_timestamps: list[int]
    target_history_occurrence_count: int
    target_removed_count: int
    remove_target_from_context: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def git_value(args: list[str], default: str) -> str:
    env_commit = os.environ.get("LTR_GIT_COMMIT")
    if args == ["rev-parse", "HEAD"] and env_commit:
        return env_commit
    env_branch = os.environ.get("LTR_GIT_BRANCH")
    if args == ["rev-parse", "--abbrev-ref", "HEAD"] and env_branch:
        return env_branch
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


def artifact_dir(config: dict[str, Any], sanity: bool) -> Path:
    base = Path(config["remote_artifact_dir"]).expanduser()
    if sanity:
        return base.with_name(base.name + "_sanity")
    return base


def stage_dirs(root: Path) -> dict[str, Path]:
    return {
        "candidates": root / "candidates",
        "features": root / "features",
        "model": root / "model",
        "rankings": root / "rankings",
        "metrics": root / "metrics",
        "logs": root / "logs",
    }


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / f"{split}.parquet")
    expected = {"user_id", "item_id", "timestamp", "source_row_id"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{split}.parquet: нет колонок {sorted(missing)}")
    df = df.loc[:, ["user_id", "item_id", "timestamp", "source_row_id"]].copy()
    for col in ["user_id", "item_id", "timestamp", "source_row_id"]:
        df[col] = df[col].astype("int64")
    return df.sort_values(["user_id", "timestamp", "source_row_id"]).reset_index(drop=True)


def user_sequences(df: pd.DataFrame, allowed_users: set[int] | None = None) -> dict[int, tuple[list[int], list[int]]]:
    if allowed_users is not None:
        df = df[df["user_id"].isin(allowed_users)]
    result: dict[int, tuple[list[int], list[int]]] = {}
    for user_id, group in df.groupby("user_id", sort=True):
        result[int(user_id)] = (
            [int(x) for x in group["item_id"].to_numpy()],
            [int(x) for x in group["timestamp"].to_numpy()],
        )
    return result


def single_targets(df: pd.DataFrame, allowed_users: set[int]) -> dict[int, tuple[int, int]]:
    filtered = df[df["user_id"].isin(allowed_users)]
    counts = filtered.groupby("user_id").size()
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(f"Split должен иметь ровно 1 target на user; bad users={len(bad)}")
    return {
        int(row.user_id): (int(row.item_id), int(row.timestamp))
        for row in filtered.itertuples(index=False)
    }


def without_target(items: list[int], timestamps: list[int], target: int) -> tuple[list[int], list[int], int]:
    kept_items: list[int] = []
    kept_timestamps: list[int] = []
    removed = 0
    for item, ts in zip(items, timestamps):
        if item == target:
            removed += 1
        else:
            kept_items.append(item)
            kept_timestamps.append(ts)
    return kept_items, kept_timestamps, removed


def context_for_target(
    items: list[int],
    timestamps: list[int],
    target: int,
    remove_target: bool,
) -> tuple[list[int], list[int], int, int]:
    occurrence_count = sum(1 for item in items if item == target)
    if remove_target:
        context_items, context_timestamps, removed_count = without_target(items, timestamps, target)
    else:
        context_items = list(items)
        context_timestamps = list(timestamps)
        removed_count = 0
    return context_items, context_timestamps, occurrence_count, removed_count


def remove_target_from_context_config(config: dict[str, Any]) -> dict[str, bool]:
    configured = config.get("context", {}).get("remove_target_from_context", {})
    return {
        "train": bool(configured.get("train", True)),
        "validation": bool(configured.get("validation", True)),
        "test": bool(configured.get("test", True)),
    }


def query_specs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sanity_max_users: int | None,
    remove_target_from_context: dict[str, bool],
) -> dict[str, list[QuerySpec]]:
    train_users = set(int(x) for x in train_df["user_id"].unique())
    validation_users = set(int(x) for x in validation_df["user_id"].unique())
    test_users = set(int(x) for x in test_df["user_id"].unique())
    users = sorted(train_users & validation_users & test_users)
    if sanity_max_users is not None:
        users = users[:sanity_max_users]
    allowed = set(users)

    train_seq = user_sequences(train_df, allowed)
    validation_targets = single_targets(validation_df, allowed)
    test_targets = single_targets(test_df, allowed)

    specs: dict[str, list[QuerySpec]] = {"train": [], "validation": [], "test": []}
    for query_index, user_id in enumerate(users):
        items, timestamps = train_seq[user_id]
        if len(items) < 1:
            raise ValueError(f"user_id={user_id}: пустой train history")

        train_target = items[-1]
        train_target_ts = timestamps[-1]
        context_items, context_ts, occurrences, removed = context_for_target(
            items[:-1],
            timestamps[:-1],
            train_target,
            remove_target_from_context["train"],
        )
        specs["train"].append(
            QuerySpec(
                query_index=query_index,
                split="train",
                user_id=user_id,
                target_item_id=train_target,
                target_timestamp=train_target_ts,
                context_items=context_items,
                context_timestamps=context_ts,
                target_history_occurrence_count=occurrences,
                target_removed_count=removed,
                remove_target_from_context=remove_target_from_context["train"],
            )
        )

        validation_target, validation_ts = validation_targets[user_id]
        context_items, context_ts, occurrences, removed = context_for_target(
            items,
            timestamps,
            validation_target,
            remove_target_from_context["validation"],
        )
        specs["validation"].append(
            QuerySpec(
                query_index=query_index,
                split="validation",
                user_id=user_id,
                target_item_id=validation_target,
                target_timestamp=validation_ts,
                context_items=context_items,
                context_timestamps=context_ts,
                target_history_occurrence_count=occurrences,
                target_removed_count=removed,
                remove_target_from_context=remove_target_from_context["validation"],
            )
        )

        test_target, test_ts = test_targets[user_id]
        validation_item, validation_time = validation_targets[user_id]
        test_history_items = [*items, validation_item]
        test_history_ts = [*timestamps, validation_time]
        context_items, context_ts, occurrences, removed = context_for_target(
            test_history_items,
            test_history_ts,
            test_target,
            remove_target_from_context["test"],
        )
        specs["test"].append(
            QuerySpec(
                query_index=query_index,
                split="test",
                user_id=user_id,
                target_item_id=test_target,
                target_timestamp=test_ts,
                context_items=context_items,
                context_timestamps=context_ts,
                target_history_occurrence_count=occurrences,
                target_removed_count=removed,
                remove_target_from_context=remove_target_from_context["test"],
            )
        )
    return specs


def item_popularity_from_train_contexts(specs: list[QuerySpec], item_universe: list[int]) -> pd.DataFrame:
    counts: Counter[int] = Counter()
    for spec in specs:
        counts.update(spec.context_items)
    ranked = sorted(item_universe, key=lambda item: (-counts.get(item, 0), item))
    rank_by_item = {item: rank + 1 for rank, item in enumerate(ranked)}
    max_rank = len(ranked)
    rows = []
    for item in item_universe:
        pop = int(counts.get(item, 0))
        rank = int(rank_by_item[item])
        rows.append(
            {
                "candidate_item_id": item,
                "item_train_popularity": pop,
                "log1p_item_train_popularity": float(np.log1p(pop)),
                "item_popularity_rank": rank,
                "item_popularity_percentile": 1.0 - ((rank - 1) / max(max_rank - 1, 1)),
            }
        )
    return pd.DataFrame(rows)


def query_metadata(specs: list[QuerySpec], item_popularity: dict[int, int]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        length = len(spec.context_items)
        unique_items = len(set(spec.context_items))
        pops = [item_popularity.get(item, 0) for item in spec.context_items]
        timestamps = spec.context_timestamps
        if pops:
            pop_mean = float(np.mean(pops))
            pop_median = float(np.median(pops))
            pop_max = float(np.max(pops))
            last_pop = float(item_popularity.get(spec.context_items[-1], 0))
        else:
            pop_mean = 0.0
            pop_median = 0.0
            pop_max = 0.0
            last_pop = 0.0
        if len(timestamps) >= 2:
            span_days = float((timestamps[-1] - timestamps[0]) / 86_400_000)
            gaps = np.diff(np.array(timestamps, dtype=np.int64))
            mean_gap_hours = float(np.mean(gaps) / 3_600_000)
        else:
            span_days = 0.0
            mean_gap_hours = 0.0
        rows.append(
            {
                "query_index": spec.query_index,
                "split": spec.split,
                "user_id": spec.user_id,
                "target_item_id": spec.target_item_id,
                "target_timestamp": spec.target_timestamp,
                "user_history_length": length,
                "user_unique_items": unique_items,
                "user_history_unique_ratio": float(unique_items / length) if length else 0.0,
                "user_history_popularity_mean": pop_mean,
                "user_history_popularity_median": pop_median,
                "user_history_popularity_max": pop_max,
                "user_history_popularity_last": last_pop,
                "user_history_span_days": span_days,
                "user_history_mean_gap_hours": mean_gap_hours,
                "target_occurrences_in_history_before_masking": spec.target_history_occurrence_count,
                "target_removed_from_history_count": spec.target_removed_count,
                "remove_target_from_context": spec.remove_target_from_context,
                "context_contains_target": spec.target_item_id in set(spec.context_items),
                "context_items": [int(x) for x in spec.context_items],
                "context_timestamps": [int(x) for x in spec.context_timestamps],
            }
        )
    return pd.DataFrame(rows)


def sample_negatives(
    item_universe: list[int],
    excluded: set[int],
    n_negatives: int,
    seed: int,
    split: str,
    user_id: int,
) -> list[int]:
    if len(item_universe) - len(excluded) < n_negatives:
        raise ValueError(f"user_id={user_id}: недостаточно unseen negatives")
    rng = random.Random(stable_seed(seed, split, user_id))
    negatives: list[int] = []
    used = set(excluded)
    while len(negatives) < n_negatives:
        item = item_universe[rng.randrange(len(item_universe))]
        if item in used:
            continue
        used.add(item)
        negatives.append(item)
    return negatives


def candidates_for_split(specs: list[QuerySpec], item_universe: list[int], n_negatives: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split = specs[0].split if specs else "unknown"
    for spec in specs:
        context_set = set(spec.context_items)
        negatives = sample_negatives(
            item_universe=item_universe,
            excluded=context_set | {spec.target_item_id},
            n_negatives=n_negatives,
            seed=seed,
            split=split,
            user_id=spec.user_id,
        )
        query_candidates = [spec.target_item_id, *negatives]
        for order, item in enumerate(query_candidates):
            rows.append(
                {
                    "query_index": spec.query_index,
                    "split": split,
                    "user_id": spec.user_id,
                    "candidate_item_id": int(item),
                    "label": 1 if order == 0 else 0,
                    "target_item_id": spec.target_item_id,
                    "candidate_order": order,
                    "candidate_seen_before": int(item in context_set),
                }
            )
    return pd.DataFrame(rows)


def content_checksum(df: pd.DataFrame, columns: list[str]) -> str:
    h = hashlib.sha256()
    h.update("|".join(columns).encode("utf-8"))
    for col in columns:
        h.update(b"\0")
        if pd.api.types.is_integer_dtype(df[col]):
            h.update(np.ascontiguousarray(df[col].to_numpy(dtype="<i8")).tobytes())
        elif pd.api.types.is_float_dtype(df[col]):
            h.update(np.ascontiguousarray(df[col].to_numpy(dtype="<f8")).tobytes())
        else:
            for value in df[col].astype(str):
                h.update(value.encode("utf-8"))
                h.update(b"\n")
    return h.hexdigest()


def validate_candidates(candidates: pd.DataFrame, n_negatives: int) -> dict[str, Any]:
    expected_group_size = n_negatives + 1
    group_sizes = candidates.groupby("query_index").size()
    positives = candidates.groupby("query_index")["label"].sum()
    unique_items = candidates.groupby("query_index")["candidate_item_id"].nunique()
    if not (group_sizes == expected_group_size).all():
        raise AssertionError("Не все queries имеют 1 positive + n negatives")
    if not (positives == 1).all():
        raise AssertionError("Не все queries имеют ровно один positive")
    if not (unique_items == expected_group_size).all():
        raise AssertionError("Внутри query есть duplicate candidate item")
    negative_target = candidates[(candidates["label"] == 0) & (candidates["candidate_item_id"] == candidates["target_item_id"])]
    if not negative_target.empty:
        raise AssertionError("negative candidate совпал с target")
    positive_bad = candidates[(candidates["label"] == 1) & (candidates["candidate_item_id"] != candidates["target_item_id"])]
    if not positive_bad.empty:
        raise AssertionError("positive candidate не совпал с target")
    return {
        "queries": int(group_sizes.shape[0]),
        "rows": int(candidates.shape[0]),
        "candidates_per_query": expected_group_size,
        "positive_per_query": 1,
        "duplicate_candidate_queries": int((unique_items != expected_group_size).sum()),
        "negative_equals_positive_rows": int(negative_target.shape[0]),
        "content_sha256": content_checksum(
            candidates,
            ["query_index", "user_id", "candidate_item_id", "label", "target_item_id", "candidate_order"],
        ),
    }


def prepare_candidates(config: dict[str, Any], root: Path, sanity: bool, force: bool) -> dict[str, Any]:
    dirs = stage_dirs(root)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["protocol"]["data_dir"])
    train_df = read_split(data_dir, "train")
    validation_df = read_split(data_dir, "validation")
    test_df = read_split(data_dir, "test")
    full_df = pd.read_parquet(data_dir / "full_filtered.parquet", columns=["user_id", "item_id"])
    item_universe = sorted(int(x) for x in full_df["item_id"].unique())
    dataset_fingerprint = {
        "users": int(full_df["user_id"].nunique()),
        "items": int(full_df["item_id"].nunique()),
        "interactions": int(full_df.shape[0]),
        "train": int(train_df.shape[0]),
        "validation": int(validation_df.shape[0]),
        "test": int(test_df.shape[0]),
    }
    sanity_max_users = int(config["sanity"]["max_users"]) if sanity else None
    remove_target_config = remove_target_from_context_config(config)
    specs = query_specs(train_df, validation_df, test_df, sanity_max_users, remove_target_config)

    item_features = item_popularity_from_train_contexts(specs["train"], item_universe)
    item_features_path = dirs["features"] / "item_train_popularity.parquet"
    item_features.to_parquet(item_features_path, index=False)
    item_popularity = {
        int(row.candidate_item_id): int(row.item_train_popularity)
        for row in item_features.itertuples(index=False)
    }

    split_summaries: dict[str, Any] = {}
    for split, split_specs in specs.items():
        candidates_path = dirs["candidates"] / f"{split}_candidates.parquet"
        queries_path = dirs["candidates"] / f"{split}_queries.parquet"
        if candidates_path.exists() and queries_path.exists() and not force:
            candidates = pd.read_parquet(candidates_path)
            queries = pd.read_parquet(queries_path)
        else:
            queries = query_metadata(split_specs, item_popularity)
            candidates = candidates_for_split(
                split_specs,
                item_universe,
                int(config["candidate_generation"]["n_negatives"]),
                int(config["seed"]),
            )
            queries.to_parquet(queries_path, index=False)
            candidates.to_parquet(candidates_path, index=False)

        validation = validate_candidates(candidates, int(config["candidate_generation"]["n_negatives"]))
        if bool(queries["context_contains_target"].any()) and remove_target_config[split]:
            raise AssertionError(f"{split}: target присутствует в context")
        split_summaries[split] = {
            **validation,
            "queries_path": str(queries_path),
            "candidates_path": str(candidates_path),
            "target_occurrences_in_history_total": int(queries["target_occurrences_in_history_before_masking"].sum()),
            "target_occurrences_in_history_queries": int((queries["target_occurrences_in_history_before_masking"] > 0).sum()),
            "target_removed_from_history_total": int(queries["target_removed_from_history_count"].sum()),
            "target_removed_from_history_queries": int((queries["target_removed_from_history_count"] > 0).sum()),
            "context_contains_target_queries": int(queries["context_contains_target"].sum()),
            "remove_target_from_context": bool(remove_target_config[split]),
        }

    summary = {
        "stage": "prepare_candidates",
        "sanity": sanity,
        "seed": int(config["seed"]),
        "n_negatives": int(config["candidate_generation"]["n_negatives"]),
        "item_universe": len(item_universe),
        "dataset_fingerprint": dataset_fingerprint,
        "remove_target_from_context": remove_target_config,
        "item_features_path": str(item_features_path),
        "splits": split_summaries,
    }
    save_json(dirs["metrics"] / "candidate_summary.json", summary)
    return summary


def build_features(config: dict[str, Any], root: Path) -> dict[str, Any]:
    dirs = stage_dirs(root)
    item_features = pd.read_parquet(dirs["features"] / "item_train_popularity.parquet")
    feature_names = list(config["features"]["names"])
    summaries: dict[str, Any] = {}

    for split in ["train", "validation", "test"]:
        candidates = pd.read_parquet(dirs["candidates"] / f"{split}_candidates.parquet")
        queries = pd.read_parquet(dirs["candidates"] / f"{split}_queries.parquet")
        df = candidates.merge(
            queries,
            on=["query_index", "split", "user_id", "target_item_id"],
            how="left",
            validate="many_to_one",
        )
        df = df.merge(item_features, on="candidate_item_id", how="left", validate="many_to_one")
        df["candidate_seen_before"] = df["candidate_seen_before"].astype("float32")
        mean_pop = df["user_history_popularity_mean"].replace(0, np.nan)
        df["candidate_popularity_to_user_mean"] = (df["item_train_popularity"] / mean_pop).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["candidate_popularity_minus_user_mean"] = df["item_train_popularity"] - df["user_history_popularity_mean"]
        df["candidate_is_more_popular_than_user_mean"] = (
            df["item_train_popularity"] > df["user_history_popularity_mean"]
        ).astype("int8")

        for name in feature_names:
            if name not in df.columns:
                raise ValueError(f"{split}: feature {name} не построен")
            df[name] = pd.to_numeric(df[name], errors="raise").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if not np.isfinite(df[feature_names].to_numpy(dtype=np.float64)).all():
            raise AssertionError(f"{split}: feature matrix содержит NaN/inf")

        columns = [
            "query_index",
            "split",
            "user_id",
            "candidate_item_id",
            "label",
            "target_item_id",
            "candidate_order",
            *feature_names,
        ]
        df = df.loc[:, columns].sort_values(["query_index", "candidate_order"]).reset_index(drop=True)
        out_path = dirs["features"] / f"{split}_features.parquet"
        df.to_parquet(out_path, index=False)
        summaries[split] = {
            "rows": int(df.shape[0]),
            "queries": int(df["query_index"].nunique()),
            "features_path": str(out_path),
            "feature_sha256": content_checksum(df, ["query_index", "candidate_item_id", "label", *feature_names]),
        }

    summary = {"stage": "build_features", "feature_names": feature_names, "splits": summaries}
    save_json(dirs["metrics"] / "feature_summary.json", summary)
    return summary


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("query_index", sort=False).size().to_numpy(dtype=np.uint32)


def validate_xgb_group_structure(
    df: pd.DataFrame,
    split: str,
    expected_group_size: int,
    expected_groups: int | None = None,
) -> dict[str, Any]:
    query_values = df["query_index"].to_numpy(dtype=np.int64)
    if query_values.size and bool(np.any(query_values[1:] < query_values[:-1])):
        raise AssertionError(f"{split}: rows не отсортированы/сгруппированы по query_index")
    groups = group_sizes(df)
    if int(groups.sum()) != int(df.shape[0]):
        raise AssertionError(f"{split}: sum(group_sizes) != number_of_rows")
    if expected_groups is not None and int(groups.shape[0]) != int(expected_groups):
        raise AssertionError(f"{split}: groups={groups.shape[0]}, expected={expected_groups}")
    if not bool(np.all(groups == expected_group_size)):
        raise AssertionError(f"{split}: не все groups имеют размер {expected_group_size}")
    positives = df.groupby("query_index", sort=False)["label"].sum().to_numpy(dtype=np.float64)
    if not bool(np.all(positives == 1.0)):
        raise AssertionError(f"{split}: не в каждой group ровно один positive")
    return {
        "split": split,
        "rows": int(df.shape[0]),
        "groups": int(groups.shape[0]),
        "expected_group_size": int(expected_group_size),
        "sum_group_sizes": int(groups.sum()),
        "one_positive_per_group": True,
        "query_index_monotonic_non_decreasing": True,
    }


def feature_matrix(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return np.ascontiguousarray(df.loc[:, feature_names].to_numpy(dtype=np.float32))


def train_xgboost(config: dict[str, Any], root: Path, sanity: bool) -> dict[str, Any]:
    import xgboost as xgb

    dirs = stage_dirs(root)
    feature_names = list(config["features"]["names"])
    train_df = pd.read_parquet(dirs["features"] / "train_features.parquet")
    validation_df = pd.read_parquet(dirs["features"] / "validation_features.parquet")

    params = dict(config["xgboost"])
    num_boost_round = int(params.pop("num_boost_round"))
    early_stopping_rounds_config = params.pop("early_stopping_rounds", None)
    early_stopping_rounds = None if early_stopping_rounds_config is None else int(early_stopping_rounds_config)
    if early_stopping_rounds is not None and early_stopping_rounds <= 0:
        early_stopping_rounds = None
    if sanity:
        num_boost_round = int(config["sanity"]["num_boost_round"])
        sanity_early_stopping = config["sanity"].get("early_stopping_rounds")
        early_stopping_rounds = None if sanity_early_stopping is None else int(sanity_early_stopping)
        if early_stopping_rounds is not None and early_stopping_rounds <= 0:
            early_stopping_rounds = None
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        params["nthread"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    params.setdefault("verbosity", 1)
    expected_group_size = int(config["candidate_generation"]["n_negatives"]) + 1
    expected_train_groups = int(train_df["query_index"].nunique())
    group_validation = {
        "train": validate_xgb_group_structure(train_df, "train", expected_group_size, expected_train_groups),
        "validation_sampled_diagnostic": validate_xgb_group_structure(
            validation_df,
            "validation",
            expected_group_size,
            expected_train_groups,
        ),
    }

    dtrain = xgb.DMatrix(
        feature_matrix(train_df, feature_names),
        label=train_df["label"].to_numpy(dtype=np.float32),
        feature_names=feature_names,
        group=group_sizes(train_df),
    )
    dvalid = xgb.DMatrix(
        feature_matrix(validation_df, feature_names),
        label=validation_df["label"].to_numpy(dtype=np.float32),
        feature_names=feature_names,
        group=group_sizes(validation_df),
    )

    evals_result: dict[str, Any] = {}
    start = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=25,
    )
    train_time = time.perf_counter() - start

    model_path = dirs["model"] / "xgb_lambdamart.json"
    booster.save_model(model_path)
    best_iteration = getattr(booster, "best_iteration", None)
    best_score = getattr(booster, "best_score", None)
    summary = {
        "stage": "train",
        "xgboost_version": xgb.__version__,
        "params": params,
        "num_boost_round_requested": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "model_selection": config.get("model_selection", {}),
        "group_validation": group_validation,
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "best_score": None if best_score is None else float(best_score),
        "num_boosted_rounds": int(booster.num_boosted_rounds()),
        "train_time_sec": float(train_time),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "feature_importance_gain": booster.get_score(importance_type="gain"),
        "evals_result": evals_result,
    }
    save_json(dirs["metrics"] / "train_summary.json", summary)
    return summary


def ranked_frame(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    if scores.shape[0] != df.shape[0]:
        raise ValueError("Число scores не совпадает с числом candidate rows")
    if not np.isfinite(scores).all():
        raise AssertionError("scores содержат NaN/inf")
    ranked = df.loc[:, ["query_index", "user_id", "candidate_item_id", "label"]].copy()
    ranked["score"] = scores.astype("float32")
    ranked = ranked.sort_values(
        ["query_index", "score", "candidate_item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked["rank"] = ranked.groupby("query_index", sort=False).cumcount() + 1
    return ranked.reset_index(drop=True)


def metrics_from_ranked(ranked: pd.DataFrame, expected_group_size: int) -> dict[str, float]:
    group_sizes = ranked.groupby("query_index").size()
    if not (group_sizes == expected_group_size).all():
        raise AssertionError("ranking не имеет ожидаемого числа строк на query")
    positives = ranked[ranked["label"] == 1].copy()
    if positives.shape[0] != group_sizes.shape[0]:
        raise AssertionError("ranking не имеет ровно одного positive на query")
    result: dict[str, float] = {}
    ranks = positives["rank"].to_numpy(dtype=np.int64)
    for k in KS:
        hits = ranks <= k
        hr = float(np.mean(hits))
        recall = hr
        ndcg = float(np.mean(np.where(hits, 1.0 / np.log2(ranks + 1), 0.0)))
        result[f"HR@{k}"] = hr
        result[f"Recall@{k}"] = recall
        result[f"NDCG@{k}"] = ndcg
        if abs(hr - recall) >= 1e-12:
            raise AssertionError(f"HR@{k} != Recall@{k}")
        for name, value in [(f"HR@{k}", hr), (f"Recall@{k}", recall), (f"NDCG@{k}", ndcg)]:
            if not 0.0 <= value <= 1.0:
                raise AssertionError(f"{name} вне [0,1]: {value}")
    return result


def random_scores(df: pd.DataFrame, seed: int, split: str) -> np.ndarray:
    scores = np.empty(df.shape[0], dtype=np.float32)
    for query_index, positions in df.groupby("query_index", sort=False).indices.items():
        rng = np.random.default_rng(stable_seed(seed, "random", split, int(query_index)))
        scores[np.array(positions, dtype=np.int64)] = rng.random(len(positions), dtype=np.float32)
    return scores


def predict_xgb(config: dict[str, Any], root: Path, split_df: pd.DataFrame, train_summary: dict[str, Any]) -> np.ndarray:
    import xgboost as xgb

    feature_names = list(config["features"]["names"])
    booster = xgb.Booster()
    booster.load_model(stage_dirs(root)["model"] / "xgb_lambdamart.json")
    dmat = xgb.DMatrix(feature_matrix(split_df, feature_names), feature_names=feature_names)
    best_iteration = train_summary.get("best_iteration")
    if best_iteration is None:
        return booster.predict(dmat)
    return booster.predict(dmat, iteration_range=(0, int(best_iteration) + 1))


def evaluate_sampled(config: dict[str, Any], root: Path, train_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    dirs = stage_dirs(root)
    if train_summary is None:
        train_summary = json.loads((dirs["metrics"] / "train_summary.json").read_text())
    feature_names = list(config["features"]["names"])
    expected_group_size = int(config["candidate_generation"]["n_negatives"]) + 1
    metrics: dict[str, Any] = {}
    inference_seconds: dict[str, float] = {}

    for split in ["validation", "test"]:
        df = pd.read_parquet(dirs["features"] / f"{split}_features.parquet")
        split_metrics: dict[str, Any] = {}
        model_scores = {
            "random": lambda: random_scores(df, int(config["seed"]), split),
            "mostpop": lambda: df["item_train_popularity"].to_numpy(dtype=np.float32),
            "ltr_xgb": lambda: predict_xgb(config, root, df, train_summary),
        }
        for model_name, score_fn in model_scores.items():
            start = time.perf_counter()
            scores = score_fn()
            ranked = ranked_frame(df, scores)
            elapsed = time.perf_counter() - start
            inference_seconds[f"{model_name}_{split}"] = float(elapsed)
            model_metrics = metrics_from_ranked(ranked, expected_group_size)
            split_metrics[model_name] = model_metrics
            ranking_path = dirs["rankings"] / f"{model_name}_{split}_ranking.parquet"
            ranked.to_parquet(ranking_path, index=False)
            split_metrics[model_name]["ranking_path"] = str(ranking_path)
            split_metrics[model_name]["ranking_sha256"] = sha256_file(ranking_path)
            split_metrics[model_name]["queries"] = int(df["query_index"].nunique())
            split_metrics[model_name]["rows"] = int(df.shape[0])
        metrics[split] = split_metrics

    summary = {
        "stage": "evaluate",
        "evaluation_protocol": config.get("evaluation_protocol", "B_split_sampled_100_candidates"),
        "feature_names": feature_names,
        "metrics": metrics,
        "inference_time_sec": inference_seconds,
    }
    save_json(dirs["metrics"] / "evaluation_summary.json", summary)
    return summary


def normalise_context_items(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist()]
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    return [int(x) for x in value]


def full_feature_batch(
    queries: pd.DataFrame,
    item_features: pd.DataFrame,
    item_ids: np.ndarray,
    item_to_pos: dict[int, int],
    feature_names: list[str],
) -> np.ndarray:
    batch_size = int(queries.shape[0])
    n_items = int(item_ids.shape[0])
    n_rows = batch_size * n_items
    matrix = np.empty((n_rows, len(feature_names)), dtype=np.float32)

    item_arrays = {
        name: item_features[name].to_numpy(dtype=np.float32)
        for name in ITEM_FEATURE_COLUMNS
    }
    repeated_mean = np.repeat(
        queries["user_history_popularity_mean"].to_numpy(dtype=np.float32),
        n_items,
    )
    tiled_pop = np.tile(item_arrays["item_train_popularity"], batch_size)
    seen = np.zeros((batch_size, n_items), dtype=np.float32)
    for row_idx, value in enumerate(queries["context_items"].to_numpy(dtype=object)):
        positions = [item_to_pos[item] for item in set(normalise_context_items(value)) if item in item_to_pos]
        if positions:
            seen[row_idx, np.array(positions, dtype=np.int64)] = 1.0

    for col_idx, name in enumerate(feature_names):
        if name in QUERY_FEATURE_COLUMNS:
            matrix[:, col_idx] = np.repeat(queries[name].to_numpy(dtype=np.float32), n_items)
        elif name in ITEM_FEATURE_COLUMNS:
            matrix[:, col_idx] = np.tile(item_arrays[name], batch_size)
        elif name == "candidate_seen_before":
            matrix[:, col_idx] = seen.reshape(-1)
        elif name == "candidate_popularity_to_user_mean":
            ratio = tiled_pop / np.where(repeated_mean == 0.0, np.nan, repeated_mean)
            matrix[:, col_idx] = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        elif name == "candidate_popularity_minus_user_mean":
            matrix[:, col_idx] = tiled_pop - repeated_mean
        elif name == "candidate_is_more_popular_than_user_mean":
            matrix[:, col_idx] = (tiled_pop > repeated_mean).astype(np.float32)
        else:
            raise ValueError(f"full evaluation: feature {name} не поддержан")

    if not np.isfinite(matrix).all():
        raise AssertionError("full evaluation: feature matrix содержит NaN/inf")
    return matrix


def scores_to_topk_frame(
    queries: pd.DataFrame,
    scores: np.ndarray,
    item_ids: np.ndarray,
    topk: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if scores.ndim != 2:
        raise ValueError("scores должен быть matrix [batch_users, n_items]")
    if not np.isfinite(scores).all():
        raise AssertionError("scores содержат NaN/inf")
    kth = min(topk, scores.shape[1]) - 1
    top_unsorted = np.argpartition(-scores, kth=kth, axis=1)[:, :topk]
    top_scores_unsorted = np.take_along_axis(scores, top_unsorted, axis=1)
    top_items_unsorted = item_ids[top_unsorted]
    top_sorted = np.empty_like(top_unsorted)
    for row_idx in range(top_unsorted.shape[0]):
        order = np.lexsort((top_items_unsorted[row_idx], -top_scores_unsorted[row_idx]))
        top_sorted[row_idx] = top_unsorted[row_idx, order]

    top_scores = np.take_along_axis(scores, top_sorted, axis=1)
    top_items = item_ids[top_sorted]
    target_items = queries["target_item_id"].to_numpy(dtype=np.int64)
    ranks = np.full(scores.shape[0], np.inf, dtype=np.float64)
    rank_positions = np.arange(1, topk + 1, dtype=np.float64)
    hit_matrix = top_items == target_items[:, None]
    hit_rows = np.where(hit_matrix.any(axis=1))[0]
    if hit_rows.size:
        ranks[hit_rows] = rank_positions[np.argmax(hit_matrix[hit_rows], axis=1)]

    frame = pd.DataFrame(
        {
            "query_index": np.repeat(queries["query_index"].to_numpy(dtype=np.int64), topk),
            "user_id": np.repeat(queries["user_id"].to_numpy(dtype=np.int64), topk),
            "candidate_item_id": top_items.reshape(-1).astype(np.int64),
            "score": top_scores.reshape(-1).astype(np.float32),
            "rank": np.tile(np.arange(1, topk + 1, dtype=np.int16), scores.shape[0]),
            "target_item_id": np.repeat(target_items, topk),
        }
    )
    frame["label"] = (frame["candidate_item_id"] == frame["target_item_id"]).astype("int8")
    return frame, ranks


def metrics_from_target_ranks(target_ranks: np.ndarray) -> dict[str, float]:
    if target_ranks.ndim != 1:
        raise ValueError("target_ranks должен быть one-dimensional")
    result: dict[str, float] = {}
    for k in KS:
        hits = target_ranks <= k
        hr = float(np.mean(hits))
        recall = hr
        ndcg_values = np.where(hits, 1.0 / np.log2(target_ranks + 1.0), 0.0)
        ndcg = float(np.mean(ndcg_values))
        result[f"HR@{k}"] = hr
        result[f"Recall@{k}"] = recall
        result[f"NDCG@{k}"] = ndcg
        if abs(hr - recall) >= 1e-12:
            raise AssertionError(f"HR@{k} != Recall@{k}")
        for name, value in [(f"HR@{k}", hr), (f"Recall@{k}", recall), (f"NDCG@{k}", ndcg)]:
            if not 0.0 <= value <= 1.0:
                raise AssertionError(f"{name} вне [0,1]: {value}")
    return result


def recbole_sequential_full_semantics(mask_seen_items: bool, item_universe_size: int) -> dict[str, Any]:
    return {
        "recbole_version_checked": "1.2.0",
        "source": {
            "sequential_yaml": "recbole/properties/quick_start_config/sequential.yaml",
            "dataset_split": "recbole/data/dataset/dataset.py",
            "full_sort_loader": "recbole/data/dataloader/general_dataloader.py",
            "trainer_full_sort": "recbole/trainer/trainer.py",
            "metrics": "recbole/evaluator/collector.py, recbole/evaluator/metrics.py",
        },
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            "group_by": "user",
            "mode": {"valid": "full", "test": "full"},
        },
        "leave_one_out": "после сортировки по timestamp последние две user interactions становятся validation и test",
        "validation_history": "train history до validation target",
        "test_history": "train history + validation interaction до test target",
        "candidate_universe": f"все real Protocol B items: {item_universe_size}",
        "sequential_history_mask": bool(mask_seen_items),
        "mask_seen_items": bool(mask_seen_items),
        "padding_mask": "RecBole trainer ставит score[:, 0] = -inf для internal padding item; raw Protocol B item_id не являются internal padding id",
        "repeated_targets": "если target item встречался раньше, для sequential full-sort он остаётся evaluable при mask_seen_items=false",
        "excluded_items": "нет sampled negatives; из raw universe ничего не исключается при mask_seen_items=false",
        "metrics": "top-k Hit/Recall/NDCG; при одном positive HR@K == Recall@K",
    }


def apply_full_masks(
    scores: np.ndarray,
    queries: pd.DataFrame,
    item_to_pos: dict[int, int],
    mask_seen_items: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not mask_seen_items:
        candidate_counts = np.full(scores.shape[0], scores.shape[1], dtype=np.int64)
        return scores, candidate_counts

    masked = scores.copy()
    candidate_counts = np.full(scores.shape[0], scores.shape[1], dtype=np.int64)
    for row_idx, value in enumerate(queries["context_items"].to_numpy(dtype=object)):
        positions = [item_to_pos[item] for item in set(normalise_context_items(value)) if item in item_to_pos]
        if positions:
            masked[row_idx, np.array(positions, dtype=np.int64)] = -np.inf
            candidate_counts[row_idx] -= len(positions)
    return masked, candidate_counts


def validate_full_eval_split(
    queries: pd.DataFrame,
    item_to_pos: dict[int, int],
    candidate_counts: list[np.ndarray],
    target_evaluable: list[np.ndarray],
    mask_seen_items: bool,
) -> dict[str, Any]:
    counts = np.concatenate(candidate_counts) if candidate_counts else np.array([], dtype=np.int64)
    evaluable = np.concatenate(target_evaluable) if target_evaluable else np.array([], dtype=bool)
    if int(queries.shape[0]) != int(evaluable.shape[0]):
        raise AssertionError("full evaluation: target_evaluable length mismatch")
    repeated = queries["target_occurrences_in_history_before_masking"].to_numpy(dtype=np.int64) > 0
    summary = {
        "queries": int(queries.shape[0]),
        "candidate_count_min": int(counts.min()) if counts.size else 0,
        "candidate_count_max": int(counts.max()) if counts.size else 0,
        "candidate_count_mean": float(counts.mean()) if counts.size else 0.0,
        "mask_seen_items": bool(mask_seen_items),
        "target_evaluable_queries": int(evaluable.sum()),
        "repeated_target_queries": int(repeated.sum()),
        "repeated_target_evaluable_queries": int((repeated & evaluable).sum()),
    }
    if summary["target_evaluable_queries"] != summary["queries"]:
        raise AssertionError("full evaluation: не все target items evaluable")
    return summary


def evaluate_full(config: dict[str, Any], root: Path, train_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    import xgboost as xgb

    dirs = stage_dirs(root)
    if train_summary is None:
        train_summary = json.loads((dirs["metrics"] / "train_summary.json").read_text())
    feature_names = list(config["features"]["names"])
    eval_config = config.get("evaluation", {})
    batch_users = int(eval_config.get("batch_users", 128))
    topk = int(eval_config.get("topk", max(KS)))
    if topk < max(KS):
        raise ValueError("evaluation.topk должен быть не меньше max(KS)")
    mask_seen_items = bool(eval_config.get("mask_seen_items", False))

    item_features = pd.read_parquet(dirs["features"] / "item_train_popularity.parquet")
    item_features = item_features.sort_values("candidate_item_id").reset_index(drop=True)
    item_ids = item_features["candidate_item_id"].to_numpy(dtype=np.int64)
    item_to_pos = {int(item): pos for pos, item in enumerate(item_ids)}

    booster = xgb.Booster()
    booster.load_model(dirs["model"] / "xgb_lambdamart.json")
    best_iteration = train_summary.get("best_iteration")
    inference_seconds: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    full_eval_stats: dict[str, Any] = {}

    for split in ["validation", "test"]:
        queries = pd.read_parquet(dirs["candidates"] / f"{split}_queries.parquet")
        queries = queries.sort_values("query_index").reset_index(drop=True)
        if int(queries.shape[0]) == 0:
            raise AssertionError(f"{split}: нет queries")
        if not set(queries["target_item_id"].astype("int64")).issubset(item_to_pos):
            raise AssertionError(f"{split}: target item отсутствует в item universe")

        split_metrics: dict[str, Any] = {}
        split_candidate_counts: dict[str, list[np.ndarray]] = {"random": [], "mostpop": [], "ltr_xgb": []}
        split_target_evaluable: dict[str, list[np.ndarray]] = {"random": [], "mostpop": [], "ltr_xgb": []}

        for model_name in ["random", "mostpop", "ltr_xgb"]:
            start = time.perf_counter()
            ranked_batches: list[pd.DataFrame] = []
            rank_batches: list[np.ndarray] = []
            for start_row in range(0, queries.shape[0], batch_users):
                batch = queries.iloc[start_row : start_row + batch_users].copy()
                batch_size = int(batch.shape[0])
                if model_name == "random":
                    scores = np.empty((batch_size, item_ids.shape[0]), dtype=np.float32)
                    for row_idx, query_index in enumerate(batch["query_index"].to_numpy(dtype=np.int64)):
                        rng = np.random.default_rng(stable_seed(int(config["seed"]), "random_full", split, int(query_index)))
                        scores[row_idx, :] = rng.random(item_ids.shape[0], dtype=np.float32)
                elif model_name == "mostpop":
                    scores = np.tile(
                        item_features["item_train_popularity"].to_numpy(dtype=np.float32),
                        (batch_size, 1),
                    )
                else:
                    matrix = full_feature_batch(batch, item_features, item_ids, item_to_pos, feature_names)
                    dmat = xgb.DMatrix(matrix, feature_names=feature_names)
                    if best_iteration is None:
                        predicted = booster.predict(dmat)
                    else:
                        predicted = booster.predict(dmat, iteration_range=(0, int(best_iteration) + 1))
                    scores = predicted.reshape(batch_size, item_ids.shape[0]).astype(np.float32, copy=False)

                scores, candidate_counts = apply_full_masks(scores, batch, item_to_pos, mask_seen_items)
                target_positions = np.array([item_to_pos[int(item)] for item in batch["target_item_id"]], dtype=np.int64)
                target_scores = scores[np.arange(batch_size), target_positions]
                target_evaluable = np.isfinite(target_scores)
                split_candidate_counts[model_name].append(candidate_counts)
                split_target_evaluable[model_name].append(target_evaluable)
                finite_scores = np.where(np.isfinite(scores), scores, np.finfo(np.float32).min)
                ranked_batch, target_ranks = scores_to_topk_frame(batch, finite_scores, item_ids, topk)
                ranked_batches.append(ranked_batch)
                rank_batches.append(target_ranks)

            elapsed = time.perf_counter() - start
            inference_seconds[f"{model_name}_{split}"] = float(elapsed)
            ranked = pd.concat(ranked_batches, ignore_index=True)
            target_ranks = np.concatenate(rank_batches)
            model_metrics = metrics_from_target_ranks(target_ranks)
            ranking_path = dirs["rankings"] / f"{model_name}_{split}_top{topk}.parquet"
            ranked.to_parquet(ranking_path, index=False)
            model_metrics["ranking_path"] = str(ranking_path)
            model_metrics["ranking_sha256"] = sha256_file(ranking_path)
            model_metrics["queries"] = int(queries.shape[0])
            model_metrics["rows_saved"] = int(ranked.shape[0])
            model_metrics["topk_saved"] = int(topk)
            split_metrics[model_name] = model_metrics

            if model_name == "random" and model_metrics["HR@10"] > 0.01:
                raise AssertionError(f"{split}: Random HR@10={model_metrics['HR@10']:.6f}, full-ranking sanity не проходит")

        full_eval_stats[split] = {
            model_name: validate_full_eval_split(
                queries,
                item_to_pos,
                split_candidate_counts[model_name],
                split_target_evaluable[model_name],
                mask_seen_items,
            )
            for model_name in ["random", "mostpop", "ltr_xgb"]
        }
        metrics[split] = split_metrics

    summary = {
        "stage": "evaluate",
        "evaluation_protocol": config.get("evaluation_protocol", "protocol_b_full"),
        "feature_names": feature_names,
        "metrics": metrics,
        "inference_time_sec": inference_seconds,
        "full_evaluation": {
            "item_universe_size": int(item_ids.shape[0]),
            "topk_saved": int(topk),
            "batch_users": int(batch_users),
            "mask_seen_items": bool(mask_seen_items),
            "mask_padding_item": bool(eval_config.get("mask_padding_item", True)),
            "recbole_semantics": recbole_sequential_full_semantics(mask_seen_items, int(item_ids.shape[0])),
            "split_stats": full_eval_stats,
        },
    }
    save_json(dirs["metrics"] / "evaluation_summary.json", summary)
    return summary


def evaluate(config: dict[str, Any], root: Path, train_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    mode = str(config.get("evaluation", {}).get("mode", "sampled")).lower()
    if mode == "full":
        return evaluate_full(config, root, train_summary)
    return evaluate_sampled(config, root, train_summary)


def cluster_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }


def result_row(
    run_id: str,
    model_label: str,
    config: dict[str, Any],
    root: Path,
    git_commit: str,
    git_branch: str,
    metrics: dict[str, Any],
    train_time_sec: float,
    inference_time_sec: float,
    notes: str,
) -> dict[str, Any]:
    test_metrics = metrics["test"]
    validation_metrics = metrics["validation"]
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_label,
        "branch": git_branch,
        "git_commit": git_commit,
        "protocol": config["protocol"]["name"],
        "candidate_protocol": config["candidate_protocol"],
        "evaluation_protocol": config.get("evaluation_protocol", "B_split_sampled_100_candidates"),
        "train_candidate_protocol": config.get("train_candidate_protocol", config["candidate_protocol"]),
        "eval_candidate_protocol": config.get("eval_candidate_protocol", config["candidate_protocol"]),
        "item_universe_size": config.get("protocol", {}).get("item_universe_size", ""),
        "mask_seen_items": config.get("evaluation", {}).get("mask_seen_items", ""),
        "protocol_version": config.get("protocol_version", ""),
        "n_negatives": config["candidate_generation"]["n_negatives"],
        "seed": config["seed"],
        "features": "|".join(config["features"]["names"]),
        "train_time_sec": f"{train_time_sec:.6f}",
        "inference_time_sec": f"{inference_time_sec:.6f}",
        "remote_artifact_path": str(root),
        "notes": notes,
    }
    for k in KS:
        for prefix in ["HR", "NDCG", "Recall"]:
            row[f"{prefix}@{k}"] = f"{test_metrics[f'{prefix}@{k}']:.12f}"
            row[f"validation_{prefix}@{k}"] = f"{validation_metrics[f'{prefix}@{k}']:.12f}"
    return row


def enrich_existing_result_row(row: dict[str, Any]) -> dict[str, Any]:
    run_id = row.get("run_id", "")
    if run_id in {"random_001", "mostpop_001", "ltr_xgb_001"}:
        row.setdefault("evaluation_protocol", "")
        row.setdefault("train_candidate_protocol", "")
        row.setdefault("eval_candidate_protocol", "")
        row.setdefault("item_universe_size", "")
        row.setdefault("mask_seen_items", "")
        row.setdefault("protocol_version", "")
        row["evaluation_protocol"] = row["evaluation_protocol"] or "B_split_sampled_100_candidates"
        row["train_candidate_protocol"] = row["train_candidate_protocol"] or "sampled_100"
        row["eval_candidate_protocol"] = row["eval_candidate_protocol"] or "sampled_100"
        row["item_universe_size"] = row["item_universe_size"] or "7111"
        row["mask_seen_items"] = row["mask_seen_items"] or "sampled_unseen_negatives"
        row["protocol_version"] = row["protocol_version"] or "ltr_xgb_001_sampled_eval_v1"
    return row


def upsert_results(results_path: Path, rows: list[dict[str, Any]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if results_path.exists() and results_path.stat().st_size > 0:
        with results_path.open("r", encoding="utf-8", newline="") as fh:
            existing = [enrich_existing_result_row(row) for row in csv.DictReader(fh)]
    by_run = {row["run_id"]: row for row in existing if row.get("run_id")}
    for row in rows:
        by_run[row["run_id"]] = row
    ordered_run_ids = [row["run_id"] for row in existing if row.get("run_id")]
    for row in rows:
        if row["run_id"] not in ordered_run_ids:
            ordered_run_ids.append(row["run_id"])
    with results_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for run_id in ordered_run_ids:
            writer.writerow({col: by_run[run_id].get(col, "") for col in RESULT_COLUMNS})


def compact_artifacts(
    config: dict[str, Any],
    root: Path,
    candidate_summary: dict[str, Any],
    feature_summary: dict[str, Any],
    train_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
    sanity: bool,
    write_results: bool,
) -> dict[str, Any]:
    git_commit = git_value(["rev-parse", "HEAD"], "unknown")
    git_branch = git_value(["rev-parse", "--abbrev-ref", "HEAD"], "exp/ltr-xgb-baseline")
    run_id = config["run_id"] + ("_sanity" if sanity else "")
    compact_dir = Path(config["compact_runs_dir"])
    compact_dir.mkdir(parents=True, exist_ok=True)

    dataset_sizes = {
        split: {
            "candidate_rows": candidate_summary["splits"][split]["rows"],
            "feature_rows": feature_summary["splits"][split]["rows"],
            "queries": candidate_summary["splits"][split]["queries"],
        }
        for split in ["train", "validation", "test"]
    }
    base = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sanity": sanity,
        "branch": git_branch,
        "git_commit": git_commit,
        "protocol": config["protocol"],
        "candidate_protocol": config["candidate_protocol"],
        "evaluation_protocol": config.get("evaluation_protocol", "B_split_sampled_100_candidates"),
        "train_candidate_protocol": config.get("train_candidate_protocol", config["candidate_protocol"]),
        "eval_candidate_protocol": config.get("eval_candidate_protocol", config["candidate_protocol"]),
        "protocol_version": config.get("protocol_version", ""),
        "candidate_generation": config["candidate_generation"],
        "evaluation_config": config.get("evaluation", {}),
        "model_selection": config.get("model_selection", {}),
        "feature_list": config["features"]["names"],
        "dataset_sizes": dataset_sizes,
        "candidate_summary": candidate_summary,
        "feature_summary": feature_summary,
        "train_summary": train_summary,
        "evaluation_summary": evaluation_summary,
        "cluster": cluster_info(),
        "remote_artifact_location": str(root),
    }

    suffix = config["run_id"].rsplit("_", 1)[-1]
    model_metrics = {
        f"random_{suffix}": ("Random", "random", 0.0),
        f"mostpop_{suffix}": ("MostPopular", "mostpop", 0.0),
        config["run_id"]: ("XGBoost LambdaMART", "ltr_xgb", float(train_summary["train_time_sec"])),
    }
    result_rows: list[dict[str, Any]] = []
    for artifact_run_id, (model_label, key, train_time) in model_metrics.items():
        model_artifact = {
            **base,
            "run_id": artifact_run_id if not sanity else artifact_run_id + "_sanity",
            "model": model_label,
            "metrics": {
                "validation": evaluation_summary["metrics"]["validation"][key],
                "test": evaluation_summary["metrics"]["test"][key],
            },
        }
        save_json(compact_dir / f"{model_artifact['run_id']}.json", model_artifact)
        if not sanity:
            result_rows.append(
                result_row(
                    artifact_run_id,
                    model_label,
                    config,
                    root,
                    git_commit,
                    git_branch,
                    model_artifact["metrics"],
                    train_time,
                    sum(
                        evaluation_summary["inference_time_sec"][f"{key}_{split}"]
                        for split in ["validation", "test"]
                    ),
                    "full-ranking reference baseline" if key != "ltr_xgb" else "fixed-round LambdaMART full-ranking baseline",
                )
            )

    if write_results and not sanity:
        upsert_results(Path(config["results_csv"]), result_rows)

    notes_path = compact_dir / f"{config['run_id']}_notes.md"
    if not sanity:
        notes_path.write_text(build_notes(config, root, evaluation_summary, train_summary), encoding="utf-8")
    save_json(root / "metrics" / "run_summary.json", base)
    return base


def metric_table(metrics_by_model: dict[str, dict[str, float]]) -> str:
    columns = ["Model", *[f"HR@{k}" for k in KS], *[f"NDCG@{k}" for k in KS], *[f"Recall@{k}" for k in KS]]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for label, metrics in metrics_by_model.items():
        values = [label]
        for metric in [*[f"HR@{k}" for k in KS], *[f"NDCG@{k}" for k in KS], *[f"Recall@{k}" for k in KS]]:
            values.append(f"{metrics[metric]:.6f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def literature_table() -> str:
    lines = [
        "| Source | Model | HR@10 | HR@20 | NDCG@10 | NDCG@20 |",
        "| --- | --- | --- | --- | --- | --- |",
        "| SSD4Rec arXiv 2409.01192v1 Table 4 | SASRec | 0.1040 | 0.1705 | 0.0567 | 0.0733 |",
        "| SSD4Rec arXiv 2409.01192v1 Table 4 | SSD4Rec | 0.1076 | 0.1704 | 0.0602 | 0.0759 |",
    ]
    return "\n".join(lines)


def build_notes_002(
    config: dict[str, Any],
    root: Path,
    evaluation_summary: dict[str, Any],
    train_summary: dict[str, Any],
) -> str:
    validation = {
        "Random": evaluation_summary["metrics"]["validation"]["random"],
        "MostPopular": evaluation_summary["metrics"]["validation"]["mostpop"],
        "XGBoost LambdaMART": evaluation_summary["metrics"]["validation"]["ltr_xgb"],
    }
    test = {
        "Random": evaluation_summary["metrics"]["test"]["random"],
        "MostPopular": evaluation_summary["metrics"]["test"]["mostpop"],
        "XGBoost LambdaMART": evaluation_summary["metrics"]["test"]["ltr_xgb"],
    }
    full_eval = evaluation_summary.get("full_evaluation", {})
    val_stats = full_eval.get("split_stats", {}).get("validation", {}).get("ltr_xgb", {})
    test_stats = full_eval.get("split_stats", {}).get("test", {}).get("ltr_xgb", {})
    random_hr10_full = test["Random"]["HR@10"]
    random_hr10_sampled = 0.100246336270
    return "\n".join(
        [
            "# ltr_xgb_002 notes",
            "",
            "## Цель",
            "",
            "Исправить sampled evaluation из `ltr_xgb_001` и получить простой XGBoost LambdaMART baseline с full-ranking evaluation для KuaiRand Protocol B.",
            "",
            "## Что было неправильно в 001",
            "",
            "`ltr_xgb_001` оценивался на 101 candidate на query: 1 positive + 100 sampled negatives. Поэтому абсолютные HR/NDCG/Recall не сопоставимы с опубликованными sequential full-ranking результатами SSD4Rec/TiM4Rec.",
            "",
            f"Sanity check: при 101 candidate Random HR@10 теоретически около `10 / 101 = 0.099010`; фактически в 001 на test было `{random_hr10_sampled:.6f}`. В 002 full-ranking Random HR@10 на test стал `{random_hr10_full:.6f}`.",
            "",
            "## Evaluation protocol",
            "",
            f"- Training candidate protocol: `{config.get('train_candidate_protocol', config['candidate_protocol'])}`.",
            f"- Evaluation candidate protocol: `{config.get('eval_candidate_protocol', config['candidate_protocol'])}`.",
            f"- Evaluation protocol: `{config.get('evaluation_protocol')}`.",
            f"- Item universe: `{full_eval.get('item_universe_size')}` real Protocol B items.",
            f"- Mask seen items: `{full_eval.get('mask_seen_items')}`.",
            "- RecBole `sequential.yaml` задаёт `split: {'LS': 'valid_and_test'}`, `order: TO`, `mode: full`; default `group_by=user`.",
            "- `leave_one_out` после timestamp ordering оставляет последние две interactions пользователя под validation и test.",
            "- Для sequential `FullSortEvalDataLoader` возвращает `history_index=None`; `Trainer._full_sort_batch_eval` маскирует только internal item id 0. Поэтому в raw Protocol B universe history items не исключаются.",
            "- Validation context: train history до validation target. Test context: train history + validation interaction до test target.",
            "- Repeated target items остаются evaluable при `mask_seen_items=false`.",
            "",
            "## Repeated targets",
            "",
            f"- Validation repeated target queries: `{val_stats.get('repeated_target_queries')}`; evaluable: `{val_stats.get('repeated_target_evaluable_queries')}`.",
            f"- Test repeated target queries: `{test_stats.get('repeated_target_queries')}`; evaluable: `{test_stats.get('repeated_target_evaluable_queries')}`.",
            "",
            "## Результаты",
            "",
            "Validation:",
            "",
            metric_table(validation),
            "",
            "Test:",
            "",
            metric_table(test),
            "",
            "## Сравнение с 001",
            "",
            "| Run | Eval candidates | Random HR@10 test | MostPopular HR@10 test | XGBoost HR@10 test |",
            "| --- | --- | --- | --- | --- |",
            f"| ltr_xgb_001 | 1 positive + 100 sampled negatives | {random_hr10_sampled:.6f} | 0.495637 | 0.494802 |",
            f"| ltr_xgb_002 | full item universe | {test['Random']['HR@10']:.6f} | {test['MostPopular']['HR@10']:.6f} | {test['XGBoost LambdaMART']['HR@10']:.6f} |",
            "",
            "## Сравнение с literature",
            "",
            literature_table(),
            "",
            "Источник: SSD4Rec, arXiv:2409.01192v1, Table 4. Сопоставимо только если split и full-ranking semantics полностью совпадают.",
            "",
            "## Training",
            "",
            f"- Boosting rounds requested: `{train_summary['num_boost_round_requested']}`.",
            f"- Trees trained: `{train_summary['num_boosted_rounds']}`.",
            f"- Early stopping: `{train_summary['early_stopping_rounds']}`.",
            f"- Best iteration: `{train_summary['best_iteration']}`.",
            "- Для 002 sampled validation metric не используется для выбора test model; число деревьев фиксировано заранее, чтобы изменение 001 -> 002 отражало именно evaluation protocol.",
            "",
            "## Артефакты",
            "",
            f"- Remote artifact path: `{root}`.",
            "- Full scores не сохраняются; для каждого split/model сохранён только Top-50 ranking.",
            "",
            "## Вывод",
            "",
            "002 измеряет ту же простую popularity/history модель при корректной full-ranking оценке. Абсолютные метрики больше не завышены sampled candidate protocol и могут служить честным слабым baseline для дальнейших sequential моделей.",
            "",
        ]
    )


def build_notes(config: dict[str, Any], root: Path, evaluation_summary: dict[str, Any], train_summary: dict[str, Any]) -> str:
    if config["run_id"] == "ltr_xgb_002":
        return build_notes_002(config, root, evaluation_summary, train_summary)

    validation = {
        "Random": evaluation_summary["metrics"]["validation"]["random"],
        "MostPopular": evaluation_summary["metrics"]["validation"]["mostpop"],
        "XGBoost LambdaMART": evaluation_summary["metrics"]["validation"]["ltr_xgb"],
    }
    test = {
        "Random": evaluation_summary["metrics"]["test"]["random"],
        "MostPopular": evaluation_summary["metrics"]["test"]["mostpop"],
        "XGBoost LambdaMART": evaluation_summary["metrics"]["test"]["ltr_xgb"],
    }
    xgb_test = test["XGBoost LambdaMART"]
    mostpop_test = test["MostPopular"]
    random_test = test["Random"]
    observations = [
        f"XGBoost HR@10 на test: `{xgb_test['HR@10']:.6f}`.",
        f"MostPopular HR@10 на test: `{mostpop_test['HR@10']:.6f}`.",
        f"Random HR@10 на test: `{random_test['HR@10']:.6f}`.",
    ]
    if xgb_test["HR@10"] > random_test["HR@10"]:
        observations.append("XGBoost превосходит Random по HR@10.")
    else:
        observations.append("XGBoost не превосходит Random по HR@10.")
    if xgb_test["HR@10"] > mostpop_test["HR@10"]:
        observations.append("XGBoost превосходит MostPopular по HR@10.")
    else:
        observations.append("XGBoost не превосходит MostPopular по HR@10.")

    return "\n".join(
        [
            "# ltr_xgb_001 notes",
            "",
            "## Hypothesis",
            "",
            "Простой LambdaMART на popularity/history features должен превосходить Random и желательно MostPopular.",
            "",
            "## Setup",
            "",
            f"- Protocol: `{config['protocol']['name']}`.",
            f"- Candidate protocol: `{config['candidate_protocol']}`.",
            f"- Query: `user_id`; positives: один target item; negatives/query: `{config['candidate_generation']['n_negatives']}`.",
            "- Training design: один query на пользователя, positive = последний item из train, context = train history без этого target.",
            "- Raw `user_id` и `item_id` не использовались как числовые признаки модели.",
            f"- Remote artifact path: `{root}`.",
            f"- XGBoost trees trained: `{train_summary['num_boosted_rounds']}`; best_iteration: `{train_summary['best_iteration']}`.",
            "",
            "## Result",
            "",
            "Validation:",
            "",
            metric_table(validation),
            "",
            "Test:",
            "",
            metric_table(test),
            "",
            "## Comparison",
            "",
            "- Random и MostPopular посчитаны тем же evaluation pipeline на тех же fixed candidates.",
            "- HR и Recall равны по всем K, потому что в каждом query ровно один relevant item.",
            "",
            "## Observations",
            "",
            *[f"- {line}" for line in observations],
            "",
            "## Problems",
            "",
            "- XGBoost отсутствовал в `.conda` на cHARISMa; установлен `xgboost` из внутреннего PyPI proxy кластера.",
            "- Для queries, где target item уже встречался в history, target исключался из context перед построением признаков.",
            "",
            "## Next step",
            "",
            "Добавить более содержательные leakage-safe item/user metadata features или сравнить с RecBole sequential baseline на том же fixed candidate protocol.",
            "",
        ]
    )


def run_all(config: dict[str, Any], root: Path, sanity: bool, force: bool, write_results: bool) -> dict[str, Any]:
    if force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    save_yaml(root / "config_snapshot.yaml", config)
    start = time.perf_counter()
    candidate_summary = prepare_candidates(config, root, sanity=sanity, force=force)
    feature_summary = build_features(config, root)
    train_summary = train_xgboost(config, root, sanity=sanity)
    evaluation_summary = evaluate(config, root, train_summary)
    compact = compact_artifacts(
        config,
        root,
        candidate_summary,
        feature_summary,
        train_summary,
        evaluation_summary,
        sanity=sanity,
        write_results=write_results,
    )
    compact["total_runtime_sec"] = float(time.perf_counter() - start)
    save_json(root / "metrics" / "run_summary.json", compact)
    return compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--stage",
        choices=["all", "prepare-candidates", "build-features", "train", "evaluate"],
        default="all",
    )
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = artifact_dir(config, args.sanity)

    if args.stage == "all":
        summary = run_all(config, root, sanity=args.sanity, force=args.force, write_results=args.write_results)
    elif args.stage == "prepare-candidates":
        summary = prepare_candidates(config, root, sanity=args.sanity, force=args.force)
    elif args.stage == "build-features":
        summary = build_features(config, root)
    elif args.stage == "train":
        summary = train_xgboost(config, root, sanity=args.sanity)
    else:
        summary = evaluate(config, root)

    print(json.dumps({"stage": args.stage, "artifact_dir": str(root), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
