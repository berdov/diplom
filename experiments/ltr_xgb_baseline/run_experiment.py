#!/usr/bin/env python3
"""Запуск первого XGBoost LambdaMART baseline для KuaiRand Protocol B."""

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
RESULT_COLUMNS = [
    "run_id",
    "timestamp",
    "model",
    "branch",
    "git_commit",
    "protocol",
    "candidate_protocol",
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
    target_removed_count: int


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


def query_specs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sanity_max_users: int | None,
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
        context_items, context_ts, removed = without_target(items[:-1], timestamps[:-1], train_target)
        specs["train"].append(
            QuerySpec(
                query_index=query_index,
                split="train",
                user_id=user_id,
                target_item_id=train_target,
                target_timestamp=train_target_ts,
                context_items=context_items,
                context_timestamps=context_ts,
                target_removed_count=removed,
            )
        )

        validation_target, validation_ts = validation_targets[user_id]
        context_items, context_ts, removed = without_target(items, timestamps, validation_target)
        specs["validation"].append(
            QuerySpec(
                query_index=query_index,
                split="validation",
                user_id=user_id,
                target_item_id=validation_target,
                target_timestamp=validation_ts,
                context_items=context_items,
                context_timestamps=context_ts,
                target_removed_count=removed,
            )
        )

        test_target, test_ts = test_targets[user_id]
        validation_item, validation_time = validation_targets[user_id]
        test_history_items = [*items, validation_item]
        test_history_ts = [*timestamps, validation_time]
        context_items, context_ts, removed = without_target(test_history_items, test_history_ts, test_target)
        specs["test"].append(
            QuerySpec(
                query_index=query_index,
                split="test",
                user_id=user_id,
                target_item_id=test_target,
                target_timestamp=test_ts,
                context_items=context_items,
                context_timestamps=context_ts,
                target_removed_count=removed,
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
                "target_removed_from_history_count": spec.target_removed_count,
                "context_contains_target": False,
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
        if spec.target_item_id in context_set:
            raise AssertionError(f"{split} user_id={spec.user_id}: target найден в context")
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
    full_df = pd.read_parquet(data_dir / "full_filtered.parquet", columns=["item_id"])
    item_universe = sorted(int(x) for x in full_df["item_id"].unique())
    sanity_max_users = int(config["sanity"]["max_users"]) if sanity else None
    specs = query_specs(train_df, validation_df, test_df, sanity_max_users)

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
        if bool(queries["context_contains_target"].any()):
            raise AssertionError(f"{split}: target присутствует в context")
        split_summaries[split] = {
            **validation,
            "queries_path": str(queries_path),
            "candidates_path": str(candidates_path),
            "target_removed_from_history_total": int(queries["target_removed_from_history_count"].sum()),
            "target_removed_from_history_queries": int((queries["target_removed_from_history_count"] > 0).sum()),
        }

    summary = {
        "stage": "prepare_candidates",
        "sanity": sanity,
        "seed": int(config["seed"]),
        "n_negatives": int(config["candidate_generation"]["n_negatives"]),
        "item_universe": len(item_universe),
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
    early_stopping_rounds = int(params.pop("early_stopping_rounds"))
    if sanity:
        num_boost_round = int(config["sanity"]["num_boost_round"])
        early_stopping_rounds = int(config["sanity"]["early_stopping_rounds"])
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        params["nthread"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    params.setdefault("verbosity", 1)

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


def evaluate(config: dict[str, Any], root: Path, train_summary: dict[str, Any] | None = None) -> dict[str, Any]:
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
        "feature_names": feature_names,
        "metrics": metrics,
        "inference_time_sec": inference_seconds,
    }
    save_json(dirs["metrics"] / "evaluation_summary.json", summary)
    return summary


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


def upsert_results(results_path: Path, rows: list[dict[str, Any]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if results_path.exists() and results_path.stat().st_size > 0:
        with results_path.open("r", encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))
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
        "candidate_generation": config["candidate_generation"],
        "feature_list": config["features"]["names"],
        "dataset_sizes": dataset_sizes,
        "candidate_summary": candidate_summary,
        "feature_summary": feature_summary,
        "train_summary": train_summary,
        "evaluation_summary": evaluation_summary,
        "cluster": cluster_info(),
        "remote_artifact_location": str(root),
    }

    model_metrics = {
        "random_001": ("Random", "random", 0.0),
        "mostpop_001": ("MostPopular", "mostpop", 0.0),
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
                    "reference baseline" if key != "ltr_xgb" else "first LambdaMART baseline",
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


def build_notes(config: dict[str, Any], root: Path, evaluation_summary: dict[str, Any], train_summary: dict[str, Any]) -> str:
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
