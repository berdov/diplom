#!/usr/bin/env python
"""Create a physical train+validation-only RecBole dataset for Optuna search."""

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
    parser.add_argument(
        "--multitask-dir",
        default="/home/daryumin/iberdov/diplom/data/processed/protocol_b_multitask",
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "outputs" / "data" / "protocol_b_multitask_manifest.json"),
    )
    parser.add_argument(
        "--item-id-mapping",
        default="/home/daryumin/iberdov/diplom/data/processed/protocol_b/item_id_mapping.parquet",
    )
    parser.add_argument(
        "--output-root",
        default="/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/validation_only_recbole",
    )
    parser.add_argument("--dataset", default="kuairand_multitask_validonly")
    parser.add_argument(
        "--summary-json",
        default="/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/validation_only_recbole/validation_only_dataset.json",
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


def write_benchmark_split(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    empty: bool = False,
) -> int:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(BENCHMARK_HEADER) + "\n")
        if empty:
            return 0
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


def sequential_rows(train_rows: list[dict[str, Any]], validation_row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_examples: list[dict[str, Any]] = []
    for idx in range(1, len(train_rows)):
        history = train_rows[max(0, idx - MAX_ITEM_LIST_LENGTH) : idx]
        current = dict(train_rows[idx])
        current["item_id_list"] = [int(row["item_id"]) for row in history]
        current["timestamp_list"] = [float(row["timestamp"]) for row in history]
        current["item_length"] = len(history)
        train_examples.append(current)

    history = train_rows[-MAX_ITEM_LIST_LENGTH:]
    validation_example = dict(validation_row)
    validation_example["item_id_list"] = [int(row["item_id"]) for row in history]
    validation_example["timestamp_list"] = [float(row["timestamp"]) for row in history]
    validation_example["item_length"] = len(history)
    return train_examples, validation_example


def main() -> None:
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise RuntimeError("prepare_validation_only.py requires polars; run it with the project .conda env.") from exc

    args = parse_args()
    multitask_dir = Path(args.multitask_dir)
    output_root = Path(args.output_root)
    dataset_dir = output_root / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = assert_manifest(Path(args.manifest))
    train_path = multitask_dir / "train.parquet"
    validation_path = multitask_dir / "validation.parquet"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(f"Missing train/validation parquet under {multitask_dir}")

    cols = ["user_id", "item_id", "timestamp", "source_row_id", *TARGETS]
    train = pl.read_parquet(train_path, columns=cols).with_columns(pl.lit("train").alias("_split"))
    validation = pl.read_parquet(validation_path, columns=cols).with_columns(pl.lit("validation").alias("_split"))
    if train.height != EXPECTED_FINGERPRINT["train"] or validation.height != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Unexpected row counts: train={train.height}, validation={validation.height}")

    train = train.select(cols).sort(["user_id", "timestamp", "source_row_id", "item_id"])
    validation = validation.select(cols).sort(["user_id", "timestamp", "source_row_id", "item_id"])
    combined = pl.concat([train, validation], how="vertical")
    expected_rows = EXPECTED_FINGERPRINT["train"] + EXPECTED_FINGERPRINT["validation"]
    if combined.height != expected_rows:
        raise RuntimeError(f"Unexpected validation-only row count: {combined.height}")

    user_count = int(combined.select(pl.col("user_id").n_unique()).item())
    item_min = int(combined.select(pl.col("item_id").min()).item())
    item_max = int(combined.select(pl.col("item_id").max()).item())
    if user_count != EXPECTED_FINGERPRINT["users"]:
        raise RuntimeError(f"Unexpected user count in train+validation: {user_count}")

    item_mapping_path = Path(args.item_id_mapping)
    if not item_mapping_path.exists():
        raise FileNotFoundError(f"Missing item mapping parquet: {item_mapping_path}")
    item_mapping = pl.read_parquet(item_mapping_path).sort("item_index")
    if item_mapping.height != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Unexpected item mapping rows: {item_mapping.height}")
    if item_mapping.select(pl.col("item_id").n_unique()).item() != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError("item_id_mapping.parquet has duplicate item_id values")
    known_items = set(int(value) for value in item_mapping["item_id"].to_list())
    observed_items = set(int(value) for value in combined.select("item_id").unique()["item_id"].to_list())
    missing_from_mapping = observed_items.difference(known_items)
    if missing_from_mapping:
        sample = sorted(missing_from_mapping)[:10]
        raise RuntimeError(f"Train+validation item ids missing from mapping: {sample}")

    train_inter_path = dataset_dir / f"{args.dataset}.train.inter"
    valid_inter_path = dataset_dir / f"{args.dataset}.valid.inter"
    item_path = dataset_dir / f"{args.dataset}.item"
    validation_ids_path = dataset_dir / "validation_source_row_ids.txt"

    validation_by_user = {
        int(row["user_id"]): row
        for row in validation.to_dicts()
    }
    train_examples: list[dict[str, Any]] = []
    validation_examples: list[dict[str, Any]] = []
    for user_id, user_frame in train.group_by("user_id", maintain_order=True):
        user = int(user_id[0] if isinstance(user_id, tuple) else user_id)
        if user not in validation_by_user:
            raise RuntimeError(f"Missing validation row for user {user}")
        rows = user_frame.sort(["timestamp", "source_row_id", "item_id"]).to_dicts()
        if len(rows) < 1:
            raise RuntimeError(f"User {user} has no train history")
        user_train_examples, user_validation_example = sequential_rows(rows, validation_by_user[user])
        train_examples.extend(user_train_examples)
        validation_examples.append(user_validation_example)
    if len(validation_examples) != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Unexpected benchmark validation examples: {len(validation_examples)}")
    expected_train_examples = EXPECTED_FINGERPRINT["train"] - EXPECTED_FINGERPRINT["users"]
    if len(train_examples) != expected_train_examples:
        raise RuntimeError(f"Unexpected benchmark train examples: {len(train_examples)} != {expected_train_examples}")

    train_examples.sort(key=lambda row: (int(row["user_id"]), float(row["timestamp"]), int(row["source_row_id"]), int(row["item_id"])))
    validation_examples.sort(key=lambda row: (int(row["user_id"]), float(row["timestamp"]), int(row["source_row_id"]), int(row["item_id"])))
    train_rows_written = write_benchmark_split(train_inter_path, train_examples)
    validation_rows_written = write_benchmark_split(valid_inter_path, validation_examples)

    item_frame = item_mapping.select(pl.col("item_id").alias("item_id:token"))
    item_frame.write_csv(item_path, separator="\t")

    validation_ids = validation.select("source_row_id").sort("source_row_id")
    validation_ids_path.write_text(
        "\n".join(str(int(value)) for value in validation_ids["source_row_id"].to_list()) + "\n",
        encoding="utf-8",
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "output_root": str(output_root),
        "dataset_dir": str(dataset_dir),
        "benchmark_filename": ["train", "valid"],
        "train_inter_path": str(train_inter_path),
        "valid_inter_path": str(valid_inter_path),
        "item_path": str(item_path),
        "validation_source_row_ids_path": str(validation_ids_path),
        "loaded_source_paths": {
            "train": str(train_path),
            "validation": str(validation_path),
            "item_id_mapping": str(item_mapping_path),
        },
        "forbidden_test_paths_loaded": [],
        "test_path_passed_to_search": False,
        "test_rows_in_inter_file": 0,
        "test_rows_in_benchmark_file": 0,
        "rows": {
            "train": int(train.height),
            "validation": int(validation.height),
            "train_plus_validation": int(combined.height),
            "test": 0,
        },
        "sequential_examples": {
            "train": train_rows_written,
            "validation": validation_rows_written,
            "test": 0,
            "max_item_list_length": MAX_ITEM_LIST_LENGTH,
        },
        "users": user_count,
        "items_sidecar_rows": EXPECTED_FINGERPRINT["items"],
        "items_observed_train_validation": int(combined.select(pl.col("item_id").n_unique()).item()),
        "item_id_range_train_validation": {"min": item_min, "max": item_max},
        "item_id_mapping_path": str(item_mapping_path),
        "item_id_mapping_sha256": sha256_file(item_mapping_path),
        "protocol_fingerprint": EXPECTED_FINGERPRINT,
        "identity_hash": EXPECTED_IDENTITY_HASH,
        "manifest_path": str(Path(args.manifest)),
        "manifest_sha256": sha256_file(Path(args.manifest)),
        "manifest_join_is_exact": bool(manifest["join_diagnostics"]["join_is_exact"]),
        "files": {
            "train_inter_sha256": sha256_file(train_inter_path),
            "valid_inter_sha256": sha256_file(valid_inter_path),
            "item_sha256": sha256_file(item_path),
            "validation_source_row_ids_sha256": sha256_file(validation_ids_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
