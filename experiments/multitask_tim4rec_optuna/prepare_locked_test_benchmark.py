#!/usr/bin/env python
"""Подготовить benchmark train/valid/test для locked final test."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FINGERPRINT = {
    "users": 23951,
    "items": 7111,
    "interactions": 1134420,
    "train": 1086518,
    "validation": 23951,
    "test": 23951,
}
EXPECTED_IDENTITY_HASH = "954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7"
TARGETS = ("is_click", "long_view", "is_like", "is_profile_enter")
MAX_ITEM_LIST_LENGTH = 50
BENCHMARK_HEADER = [
    "user_id:token",
    "item_id:token",
    "timestamp:float",
    "source_row_id:float",
    "is_click:float",
    "long_view:float",
    "is_like:float",
    "is_profile_enter:float",
    "item_id_list:token_seq",
    "timestamp_list:float_seq",
    "item_length:float",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multitask-dir", default="/home/daryumin/iberdov/diplom/data/processed/protocol_b_multitask")
    parser.add_argument("--manifest", default=str(ROOT / "outputs/data/protocol_b_multitask_manifest.json"))
    parser.add_argument(
        "--item-id-mapping",
        default="/home/daryumin/iberdov/diplom/data/processed/protocol_b/item_id_mapping.parquet",
    )
    parser.add_argument(
        "--output-root",
        default="/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/locked_test_recbole",
    )
    parser.add_argument("--dataset", default="kuairand_multitask_locked")
    parser.add_argument(
        "--summary-json",
        default="/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/locked_test_recbole/locked_test_dataset.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    fingerprint = manifest["dataset_fingerprint"]
    observed = {
        "users": int(fingerprint["users"]),
        "items": int(fingerprint["items"]),
        "interactions": int(fingerprint["interactions"]),
        "train": int(fingerprint["split_counts"]["train"]),
        "validation": int(fingerprint["split_counts"]["validation"]),
        "test": int(fingerprint["split_counts"]["test"]),
    }
    if observed != EXPECTED_FINGERPRINT:
        raise RuntimeError(f"Protocol B multitask fingerprint mismatch: {observed}")
    identity_hash = fingerprint["identity_hash_user_item_timestamp_split"]
    if identity_hash != EXPECTED_IDENTITY_HASH:
        raise RuntimeError(f"Identity hash mismatch: {identity_hash}")
    if not bool(manifest["join_diagnostics"]["join_is_exact"]):
        raise RuntimeError(f"Join is not exact: {manifest['join_diagnostics']}")
    return manifest


def format_seq(values: list[int | float]) -> str:
    return " ".join(str(value) for value in values)


def write_benchmark_split(path: Path, rows: list[dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(BENCHMARK_HEADER) + "\n")
        for row in rows:
            values = [
                str(row["user_id"]),
                str(row["item_id"]),
                str(row["timestamp"]),
                str(row["source_row_id"]),
                str(row["is_click"]),
                str(row["long_view"]),
                str(row["is_like"]),
                str(row["is_profile_enter"]),
                format_seq(row["item_id_list"]),
                format_seq(row["timestamp_list"]),
                str(row["item_length"]),
            ]
            handle.write("\t".join(values) + "\n")
    return len(rows)


def make_train_examples(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for idx in range(1, len(train_rows)):
        history = train_rows[max(0, idx - MAX_ITEM_LIST_LENGTH) : idx]
        current = dict(train_rows[idx])
        current["item_id_list"] = [int(row["item_id"]) for row in history]
        current["timestamp_list"] = [float(row["timestamp"]) for row in history]
        current["item_length"] = len(history)
        examples.append(current)
    return examples


def make_eval_example(target_row: dict[str, Any], history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    history = history_rows[-MAX_ITEM_LIST_LENGTH:]
    example = dict(target_row)
    example["item_id_list"] = [int(row["item_id"]) for row in history]
    example["timestamp_list"] = [float(row["timestamp"]) for row in history]
    example["item_length"] = len(history)
    return example


def source_ids(frame: Any) -> list[str]:
    return [str(int(value)) for value in frame.select("source_row_id").sort("source_row_id")["source_row_id"].to_list()]


def main() -> None:
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise RuntimeError("prepare_locked_test_benchmark.py requires polars; run it with the project .conda env.") from exc

    args = parse_args()
    multitask_dir = Path(args.multitask_dir)
    output_root = Path(args.output_root)
    dataset_dir = output_root / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = assert_manifest(Path(args.manifest))
    paths = {
        "train": multitask_dir / "train.parquet",
        "validation": multitask_dir / "validation.parquet",
        "test": multitask_dir / "test.parquet",
    }
    missing = {split: str(path) for split, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing multitask parquet files: {missing}")

    cols = ["user_id", "item_id", "timestamp", "source_row_id", *TARGETS]
    train = pl.read_parquet(paths["train"], columns=cols).select(cols).sort(["user_id", "timestamp", "source_row_id", "item_id"])
    validation = (
        pl.read_parquet(paths["validation"], columns=cols)
        .select(cols)
        .sort(["user_id", "timestamp", "source_row_id", "item_id"])
    )
    test = pl.read_parquet(paths["test"], columns=cols).select(cols).sort(["user_id", "timestamp", "source_row_id", "item_id"])
    observed_rows = {"train": train.height, "validation": validation.height, "test": test.height}
    for split, expected in (("train", EXPECTED_FINGERPRINT["train"]), ("validation", EXPECTED_FINGERPRINT["validation"]), ("test", EXPECTED_FINGERPRINT["test"])):
        if int(observed_rows[split]) != expected:
            raise RuntimeError(f"Unexpected {split} rows: {observed_rows}")

    item_mapping_path = Path(args.item_id_mapping)
    item_mapping = pl.read_parquet(item_mapping_path).sort("item_index")
    if item_mapping.height != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Unexpected item mapping rows: {item_mapping.height}")
    known_items = set(int(value) for value in item_mapping["item_id"].to_list())
    combined = pl.concat([train, validation, test], how="vertical")
    observed_items = set(int(value) for value in combined.select("item_id").unique()["item_id"].to_list())
    missing_from_mapping = observed_items.difference(known_items)
    if missing_from_mapping:
        raise RuntimeError(f"Observed item ids missing from mapping: {sorted(missing_from_mapping)[:10]}")

    validation_by_user = {int(row["user_id"]): row for row in validation.to_dicts()}
    test_by_user = {int(row["user_id"]): row for row in test.to_dicts()}
    train_examples: list[dict[str, Any]] = []
    validation_examples: list[dict[str, Any]] = []
    test_examples: list[dict[str, Any]] = []
    for user_id, user_frame in train.group_by("user_id", maintain_order=True):
        user = int(user_id[0] if isinstance(user_id, tuple) else user_id)
        if user not in validation_by_user or user not in test_by_user:
            raise RuntimeError(f"Missing validation/test target for user {user}")
        train_rows = user_frame.sort(["timestamp", "source_row_id", "item_id"]).to_dicts()
        validation_row = validation_by_user[user]
        test_row = test_by_user[user]
        if len(train_rows) < 1:
            raise RuntimeError(f"User {user} has no train history")
        train_examples.extend(make_train_examples(train_rows))
        validation_examples.append(make_eval_example(validation_row, train_rows))
        test_examples.append(make_eval_example(test_row, [*train_rows, validation_row]))

    expected_train_examples = EXPECTED_FINGERPRINT["train"] - EXPECTED_FINGERPRINT["users"]
    expected_eval_examples = EXPECTED_FINGERPRINT["users"]
    if len(train_examples) != expected_train_examples:
        raise RuntimeError(f"Unexpected benchmark train examples: {len(train_examples)} != {expected_train_examples}")
    if len(validation_examples) != expected_eval_examples or len(test_examples) != expected_eval_examples:
        raise RuntimeError(
            f"Unexpected benchmark eval examples: valid={len(validation_examples)}, test={len(test_examples)}"
        )

    key = lambda row: (int(row["user_id"]), float(row["timestamp"]), int(row["source_row_id"]), int(row["item_id"]))
    train_examples.sort(key=key)
    validation_examples.sort(key=key)
    test_examples.sort(key=key)

    train_inter_path = dataset_dir / f"{args.dataset}.train.inter"
    valid_inter_path = dataset_dir / f"{args.dataset}.valid.inter"
    test_inter_path = dataset_dir / f"{args.dataset}.test.inter"
    item_path = dataset_dir / f"{args.dataset}.item"
    validation_ids_path = dataset_dir / "validation_source_row_ids.txt"
    test_ids_path = dataset_dir / "test_source_row_ids.txt"

    train_rows_written = write_benchmark_split(train_inter_path, train_examples)
    validation_rows_written = write_benchmark_split(valid_inter_path, validation_examples)
    test_rows_written = write_benchmark_split(test_inter_path, test_examples)
    item_mapping.select(pl.col("item_id").alias("item_id:token")).write_csv(item_path, separator="\t")
    validation_ids_path.write_text("\n".join(source_ids(validation)) + "\n", encoding="utf-8")
    test_ids_path.write_text("\n".join(source_ids(test)) + "\n", encoding="utf-8")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "output_root": str(output_root),
        "dataset_dir": str(dataset_dir),
        "benchmark_filename": ["train", "valid", "test"],
        "train_inter_path": str(train_inter_path),
        "valid_inter_path": str(valid_inter_path),
        "test_inter_path": str(test_inter_path),
        "item_path": str(item_path),
        "validation_source_row_ids_path": str(validation_ids_path),
        "test_source_row_ids_path": str(test_ids_path),
        "loaded_source_paths": {split: str(path) for split, path in paths.items()} | {"item_id_mapping": str(item_mapping_path)},
        "test_open_policy": "prepared only after locked validation reproduction passed",
        "rows": {
            "train": int(train.height),
            "validation": int(validation.height),
            "test": int(test.height),
            "total": int(combined.height),
        },
        "sequential_examples": {
            "train": train_rows_written,
            "validation": validation_rows_written,
            "test": test_rows_written,
            "max_item_list_length": MAX_ITEM_LIST_LENGTH,
        },
        "users": int(combined.select(pl.col("user_id").n_unique()).item()),
        "items_sidecar_rows": EXPECTED_FINGERPRINT["items"],
        "items_observed_train_validation_test": int(combined.select(pl.col("item_id").n_unique()).item()),
        "item_id_range_train_validation_test": {
            "min": int(combined.select(pl.col("item_id").min()).item()),
            "max": int(combined.select(pl.col("item_id").max()).item()),
        },
        "protocol_fingerprint": EXPECTED_FINGERPRINT,
        "identity_hash": EXPECTED_IDENTITY_HASH,
        "manifest_path": str(Path(args.manifest)),
        "manifest_sha256": sha256_file(Path(args.manifest)),
        "manifest_join_is_exact": bool(manifest["join_diagnostics"]["join_is_exact"]),
        "files": {
            "train_inter_sha256": sha256_file(train_inter_path),
            "valid_inter_sha256": sha256_file(valid_inter_path),
            "test_inter_sha256": sha256_file(test_inter_path),
            "item_sha256": sha256_file(item_path),
            "validation_source_row_ids_sha256": sha256_file(validation_ids_path),
            "test_source_row_ids_sha256": sha256_file(test_ids_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
