"""CPU-only target audit for Stage 3.

This module intentionally avoids Torch/RecBole imports so it can run in the
data-preparation environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BINARY_AUDIT_TARGETS = (
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "strong_positive",
    "explicit_positive",
    "deep_engagement",
)
CONTINUOUS_AUDIT_FIELDS = (
    "play_time_ms",
    "duration_ms",
    "play_ratio",
    "profile_stay_time",
    "comment_stay_time",
)
CURRENT_AUX_TARGETS = ("is_click", "long_view", "is_like", "is_profile_enter")


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel_path(path: str | Path) -> str:
    resolved = project_path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = project_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with project_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    env_map = {
        ("rev-parse", "HEAD"): "STAGE3_GIT_COMMIT",
        ("branch", "--show-current"): "STAGE3_GIT_BRANCH",
    }
    env_key = env_map.get(tuple(args))
    if env_key and os.environ.get(env_key):
        return str(os.environ[env_key])
    for git_bin in (os.environ.get("STAGE3_GIT_BIN"), "/usr/bin/git", "git"):
        if not git_bin:
            continue
        try:
            return subprocess.check_output(
                [git_bin, *args],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return ""


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_stats_rows(path: str | Path) -> list[dict[str, str]]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stats_by_split_target(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
    return {(row["scope"], row["field"]): row for row in rows}


def float_from_row(row: Mapping[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def missing_rate(row: Mapping[str, str] | None) -> float | None:
    if row is None:
        return None
    rows = float_from_row(row, "rows")
    missing = float_from_row(row, "missing")
    if rows in (None, 0.0) or missing is None:
        return None
    return missing / rows


def derive_binary_columns(frame: Any) -> Any:
    import polars as pl

    additions = []
    if {"is_like", "is_follow", "is_comment", "is_forward"}.issubset(frame.columns):
        additions.append(
            (
                (pl.col("is_like") == 1)
                | (pl.col("is_follow") == 1)
                | (pl.col("is_comment") == 1)
                | (pl.col("is_forward") == 1)
            )
            .cast(pl.Int8)
            .alias("explicit_positive")
        )
    if {"long_view", "is_like", "is_follow", "is_comment", "is_forward"}.issubset(frame.columns):
        additions.append(
            (
                (pl.col("long_view") == 1)
                | (pl.col("is_like") == 1)
                | (pl.col("is_follow") == 1)
                | (pl.col("is_comment") == 1)
                | (pl.col("is_forward") == 1)
            )
            .cast(pl.Int8)
            .alias("deep_engagement")
        )
    if {"is_like", "is_follow", "is_comment", "is_forward", "is_profile_enter"}.issubset(frame.columns):
        additions.append(
            (
                (pl.col("is_like") == 1)
                | (pl.col("is_follow") == 1)
                | (pl.col("is_comment") == 1)
                | (pl.col("is_forward") == 1)
                | (pl.col("is_profile_enter") == 1)
            )
            .cast(pl.Int8)
            .alias("strong_positive")
        )
    if additions:
        frame = frame.with_columns(additions)
    return frame


def binary_relationships(frame: Any, binary_targets: Sequence[str]) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    available = [target for target in binary_targets if target in frame.columns]
    total = frame.height
    for idx, left in enumerate(available):
        left_values = frame[left].fill_null(0).cast(bool)
        left_pos = int(left_values.sum())
        for right in available[idx + 1 :]:
            right_values = frame[right].fill_null(0).cast(bool)
            right_pos = int(right_values.sum())
            both = int((left_values & right_values).sum())
            left_only = left_pos - both
            right_only = right_pos - both
            neither = total - both - left_only - right_only
            denominator = math.sqrt(
                max((both + left_only) * (right_only + neither) * (both + right_only) * (left_only + neither), 0)
            )
            phi = ((both * neither - left_only * right_only) / denominator) if denominator else float("nan")
            union = both + left_only + right_only
            relationships.append(
                {
                    "left": left,
                    "right": right,
                    "n": total,
                    "both_positive": both,
                    "p_left": left_pos / total if total else float("nan"),
                    "p_right": right_pos / total if total else float("nan"),
                    "p_right_given_left": both / left_pos if left_pos else float("nan"),
                    "p_left_given_right": both / right_pos if right_pos else float("nan"),
                    "jaccard": both / union if union else float("nan"),
                    "phi": phi,
                }
            )
    return relationships


def continuous_binary_relationships(
    frame: Any,
    continuous_fields: Sequence[str],
    binary_targets: Sequence[str],
) -> list[dict[str, Any]]:
    import polars as pl

    rows: list[dict[str, Any]] = []
    binaries = [target for target in binary_targets if target in frame.columns]
    continuous = [field for field in continuous_fields if field in frame.columns]
    for field in continuous:
        for target in binaries:
            data = frame.select([pl.col(field), pl.col(target).fill_null(0).cast(pl.Int8)]).drop_nulls(field)
            if data.height == 0:
                continue
            pos = data.filter(pl.col(target) == 1)
            neg = data.filter(pl.col(target) == 0)
            rows.append(
                {
                    "continuous": field,
                    "binary": target,
                    "n": data.height,
                    "mean_if_positive": float(pos[field].mean()) if pos.height else float("nan"),
                    "mean_if_negative": float(neg[field].mean()) if neg.height else float("nan"),
                    "median_if_positive": float(pos[field].median()) if pos.height else float("nan"),
                    "median_if_negative": float(neg[field].median()) if neg.height else float("nan"),
                    "pearson": float(data.select(pl.corr(field, target)).item())
                    if data[target].n_unique() > 1
                    else float("nan"),
                }
            )
    return rows


def target_audit(stage_cfg: Mapping[str, Any]) -> dict[str, Any]:
    rows = read_stats_rows(stage_cfg["source"]["target_statistics"])
    indexed = stats_by_split_target(rows)
    candidates = stage_cfg["candidate_targets"]
    audit_rows: list[dict[str, Any]] = []

    for target, metadata in candidates.items():
        train_row = indexed.get(("train", target))
        valid_row = indexed.get(("validation", target))
        audit_rows.append(
            {
                "target": target,
                "display_name": metadata["display_name"],
                "type": metadata["type"],
                "currently_used": bool(metadata["currently_used"]),
                "stage3_ablation_enabled": bool(metadata["ablation_enabled"]),
                "train_observations": int(float_from_row(train_row, "rows") or 0),
                "train_positive_count": int(float_from_row(train_row, "positives") or 0)
                if metadata["type"] == "binary"
                else None,
                "train_positive_rate": float_from_row(train_row, "positive_rate")
                if metadata["type"] == "binary"
                else None,
                "validation_positive_rate": float_from_row(valid_row, "positive_rate")
                if metadata["type"] == "binary"
                else None,
                "missing_rate_train": missing_rate(train_row),
                "basic_distribution_train": {
                    key: float_from_row(train_row, key)
                    for key in ("mean", "std", "median", "p90", "p95", "p99", "max")
                    if float_from_row(train_row, key) is not None
                },
                "construction": metadata["construction"],
                "leakage_note": metadata["leakage_note"],
            }
        )

    protocol_dir = project_path(stage_cfg["source"]["protocol_b_multitask_dir"])
    train_parquet = protocol_dir / "train.parquet"
    relationship_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []
    if train_parquet.exists():
        import polars as pl

        columns = sorted(set(BINARY_AUDIT_TARGETS) | set(CONTINUOUS_AUDIT_FIELDS))
        available_columns = pl.scan_parquet(train_parquet).collect_schema().names()
        read_columns = [column for column in columns if column in available_columns]
        frame = pl.read_parquet(train_parquet, columns=read_columns)
        frame = derive_binary_columns(frame)
        relationship_rows = binary_relationships(frame, BINARY_AUDIT_TARGETS)
        continuous_rows = continuous_binary_relationships(
            frame,
            CONTINUOUS_AUDIT_FIELDS,
            BINARY_AUDIT_TARGETS[:8],
        )
        relationship_source = rel_path(train_parquet)
    else:
        relationship_source = f"missing: {rel_path(train_parquet)}"

    payload = {
        "run_id": stage_cfg["outputs"]["target_audit_run_id"],
        "status": "COMPLETE",
        "created_at_utc": now_utc(),
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "remote_head": git_value("rev-parse", "origin/exp/moo-8families-benchmark"),
        },
        "protocol": {
            "dataset": stage_cfg["protocol"]["dataset"],
            "split": stage_cfg["protocol"]["split"],
            "evaluation_split": "train/validation descriptive audit only",
            "test_evaluation_count": 0,
            "test_dataset_loaded": False,
            "test_metrics_present": False,
        },
        "target_statistics_source": rel_path(stage_cfg["source"]["target_statistics"]),
        "target_statistics_sha256": sha256_file(stage_cfg["source"]["target_statistics"]),
        "relationship_source": relationship_source,
        "target_audit": audit_rows,
        "binary_relationships_train": relationship_rows,
        "continuous_binary_relationships_train": continuous_rows,
        "candidate_selection": {
            "single_auxiliary_ablation_targets": list(CURRENT_AUX_TARGETS),
            "not_ablationed_now": {
                "is_follow": "Very low train prevalence and not in current model scope.",
                "is_comment": "Very low train prevalence and not in current model scope.",
                "is_forward": "Very low train prevalence and not in current model scope.",
                "is_hate": "Extreme rarity, negative-feedback semantics, and not in current model scope.",
                "play_time_ms": "Continuous post-exposure signal; requires a separate objective design.",
                "play_ratio": "Continuous post-exposure signal; requires a separate objective design.",
            },
        },
        "test_evaluation_count": 0,
    }
    output_path = project_path(stage_cfg["outputs"]["runs_dir"]) / f"{payload['run_id']}.json"
    save_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/stage3_auxiliary_analysis/config.yaml")
    args = parser.parse_args(argv)
    target_audit(load_yaml(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
