"""Memory-efficient KuaiRand-27K EDA aggregation.

This script is intended for cHARISMa/Slurm execution. It scans the full 27K
interaction logs with Polars lazy queries and writes compact summaries to
``outputs/eda``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda_utils import (  # noqa: E402
    FEEDBACK_COLUMNS,
    WATCH_TIME_COLUMNS,
    available_columns,
    collect_lazy,
    concat_lazy_frames,
    dataset_inventory,
    discover_kuairand_files,
    ensure_output_dir,
    human_size,
    lazy_schema_names,
    numeric_summary,
    require_polars,
    scan_table,
    total_size_bytes,
    version_roots,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/daryumin/iberdov/Corpora"),
        help="Root directory containing KuaiRand-* folders.",
    )
    parser.add_argument(
        "--version-root",
        type=Path,
        default=None,
        help="Explicit KuaiRand-27K/KuaiRand-27K directory. Overrides --data-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eda",
        help="Directory for JSON and compact CSV outputs.",
    )
    parser.add_argument(
        "--infer-schema-length",
        type=int,
        default=10_000,
        help="Rows used by Polars for CSV schema inference.",
    )
    return parser.parse_args()


def resolve_27k_root(data_root: Path, version_root: Path | None) -> Path:
    if version_root is not None:
        return version_root

    canonical = version_roots(data_root)["27K"]
    if canonical.exists():
        return canonical

    fallback = data_root / "KuaiRand-27K"
    if fallback.exists():
        return fallback

    return canonical


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [make_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def scan_log(path: Path, policy: str, source_label: str, infer_schema_length: int) -> Any:
    pl = require_polars()
    return scan_table(path, infer_schema_length=infer_schema_length).with_columns(
        [
            pl.lit(policy).alias("policy"),
            pl.lit(source_label).alias("source_log"),
        ]
    )


def discover_interaction_scans(files: dict[str, Path | None], infer_schema_length: int) -> list[Any]:
    scans: list[Any] = []
    if files.get("standard_early") is not None:
        scans.append(
            scan_log(
                files["standard_early"],
                policy="standard",
                source_label="log_standard_4_08_to_4_21",
                infer_schema_length=infer_schema_length,
            )
        )
    if files.get("standard_late") is not None:
        scans.append(
            scan_log(
                files["standard_late"],
                policy="standard",
                source_label="log_standard_4_22_to_5_08",
                infer_schema_length=infer_schema_length,
            )
        )
    if files.get("random") is not None:
        scans.append(
            scan_log(
                files["random"],
                policy="random",
                source_label="log_random_4_22_to_5_08",
                infer_schema_length=infer_schema_length,
            )
        )
    return scans


def policy_counts(lf: Any) -> list[dict[str, Any]]:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    exprs = [pl.len().alias("interactions")]
    if "user_id" in names:
        exprs.append(pl.col("user_id").n_unique().alias("users"))
    if "video_id" in names:
        exprs.append(pl.col("video_id").n_unique().alias("items"))

    return collect_lazy(lf.group_by("policy").agg(exprs).sort("policy")).to_dicts()


def feedback_summary(lf: Any) -> list[dict[str, Any]]:
    pl = require_polars()
    feedback_cols = available_columns(lf, FEEDBACK_COLUMNS)
    if not feedback_cols:
        return []

    exprs = []
    for column in feedback_cols:
        value = pl.col(column).cast(pl.Float64, strict=False)
        exprs.extend(
            [
                value.sum().alias(f"{column}_positive_count"),
                value.mean().alias(f"{column}_positive_rate"),
                pl.col(column).is_null().sum().alias(f"{column}_missing_count"),
            ]
        )
    return collect_lazy(lf.group_by("policy").agg(exprs).sort("policy")).to_dicts()


def binary_feedback_quality(lf: Any) -> list[dict[str, Any]]:
    pl = require_polars()
    feedback_cols = available_columns(lf, FEEDBACK_COLUMNS)
    if not feedback_cols:
        return []

    exprs = []
    for column in feedback_cols:
        value = pl.col(column).cast(pl.Int64, strict=False)
        invalid = value.is_not_null() & (~value.is_in([0, 1]))
        exprs.append(invalid.sum().alias(f"{column}_invalid_count"))
    return collect_lazy(lf.group_by("policy").agg(exprs).sort("policy")).to_dicts()


def write_temporal_tables(lf: Any, output_dir: Path) -> dict[str, str]:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    outputs: dict[str, str] = {}

    if "date" in names:
        daily = collect_lazy(
            lf.group_by(["policy", "date"])
            .agg(pl.len().alias("interactions"))
            .sort(["date", "policy"])
        )
        path = output_dir / "27k_daily_interactions.csv"
        daily.write_csv(path)
        outputs["daily_interactions"] = str(path)

    if "hourmin" in names:
        hourly_lf = lf.with_columns(
            (pl.col("hourmin").cast(pl.Int64, strict=False) // 100).alias("hour")
        )
        hourly = collect_lazy(
            hourly_lf.group_by(["policy", "hour"])
            .agg(pl.len().alias("interactions"))
            .sort(["policy", "hour"])
        )
        path = output_dir / "27k_hourly_interactions.csv"
        hourly.write_csv(path)
        outputs["hourly_interactions"] = str(path)

    return outputs


def sequence_length_summary(lf: Any, output_dir: Path) -> dict[str, Any]:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if "user_id" not in names:
        return {"summary": {}, "csv": None}

    exprs = [pl.len().alias("interactions")]
    if "video_id" in names:
        exprs.append(pl.col("video_id").n_unique().alias("unique_videos"))
    if "time_ms" in names:
        time_col = pl.col("time_ms").cast(pl.Int64, strict=False)
        exprs.extend(
            [
                time_col.min().alias("first_time_ms"),
                time_col.max().alias("last_time_ms"),
            ]
        )

    sequence_lengths = collect_lazy(
        lf.group_by("user_id").agg(exprs).sort("interactions", descending=True)
    )
    path = output_dir / "27k_sequence_lengths.csv"
    sequence_lengths.write_csv(path)

    summary = numeric_summary(sequence_lengths["interactions"].to_list())
    return {"summary": summary, "csv": str(path)}


def tab_summary(lf: Any, output_dir: Path) -> dict[str, Any]:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if "tab" not in names:
        return {"records": [], "csv": None}

    exprs = [pl.len().alias("interactions")]
    if "user_id" in names:
        exprs.append(pl.col("user_id").n_unique().alias("users"))
    if "video_id" in names:
        exprs.append(pl.col("video_id").n_unique().alias("items"))
    for column in ("is_click", "long_view", "is_like"):
        if column in names:
            exprs.append(pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"{column}_rate"))

    table = collect_lazy(
        lf.group_by(["policy", "tab"]).agg(exprs).sort(["policy", "interactions"], descending=[False, True])
    )
    path = output_dir / "27k_tab_summary.csv"
    table.write_csv(path)
    return {"records": table.to_dicts(), "csv": str(path)}


def watch_time_summary(lf: Any) -> list[dict[str, Any]]:
    pl = require_polars()
    names = set(available_columns(lf, WATCH_TIME_COLUMNS))
    if not {"play_time_ms", "duration_ms"}.issubset(names):
        return []

    ratio = (
        pl.when(pl.col("duration_ms").cast(pl.Float64, strict=False) > 0)
        .then(
            pl.col("play_time_ms").cast(pl.Float64, strict=False)
            / pl.col("duration_ms").cast(pl.Float64, strict=False)
        )
        .otherwise(None)
        .alias("play_ratio")
    )
    wt = lf.with_columns(ratio)
    exprs = [
        pl.col("play_time_ms").cast(pl.Float64, strict=False).mean().alias("play_time_ms_mean"),
        pl.col("duration_ms").cast(pl.Float64, strict=False).mean().alias("duration_ms_mean"),
        pl.col("play_ratio").mean().alias("play_ratio_mean"),
        pl.col("play_ratio").quantile(0.5).alias("play_ratio_median"),
        pl.col("play_ratio").quantile(0.9).alias("play_ratio_p90"),
        pl.col("play_ratio").quantile(0.99).alias("play_ratio_p99"),
    ]
    return collect_lazy(wt.group_by("policy").agg(exprs).sort("policy")).to_dicts()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    version_root = resolve_27k_root(args.data_root, args.version_root)

    if not version_root.exists():
        raise FileNotFoundError(
            f"KuaiRand-27K directory not found: {version_root}. "
            "Run this script on cHARISMa or pass --version-root."
        )

    inventory = dataset_inventory(version_root)
    discovered_files = discover_kuairand_files(version_root)
    scans = discover_interaction_scans(discovered_files, args.infer_schema_length)
    if not scans:
        raise FileNotFoundError(
            f"No 27K interaction logs were discovered under {version_root}."
        )

    interactions = concat_lazy_frames(scans)

    temporal_outputs = write_temporal_tables(interactions, output_dir)
    sequence = sequence_length_summary(interactions, output_dir)
    tabs = tab_summary(interactions, output_dir)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(args.data_root),
        "version_root": str(version_root),
        "disk_size_bytes": total_size_bytes(inventory),
        "disk_size": human_size(total_size_bytes(inventory)),
        "discovered_files": {
            key: str(path) if path is not None else None
            for key, path in discovered_files.items()
        },
        "inventory": inventory,
        "policy_counts": policy_counts(interactions),
        "feedback_summary": feedback_summary(interactions),
        "binary_feedback_quality": binary_feedback_quality(interactions),
        "watch_time_summary": watch_time_summary(interactions),
        "sequence_length_summary": sequence["summary"],
        "tab_summary_csv": tabs["csv"],
        "temporal_outputs": temporal_outputs,
        "sequence_lengths_csv": sequence["csv"],
    }

    json_path = output_dir / "27k_summary.json"
    write_json(json_path, make_jsonable(summary))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
