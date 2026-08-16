"""Агрегация EDA для KuaiRand-27K с экономным использованием памяти.

Скрипт рассчитан на запуск через Slurm на cHARISMa. Он сканирует логи
взаимодействий KuaiRand-27K через lazy-запросы Polars и пишет только компактные
сводки. Для небольшого smoke-теста на login-node используйте ``--sanity-limit``.
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
    discover_kuairand_file_groups,
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

LOG_GROUPS: dict[str, dict[str, str]] = {
    "standard_early": {
        "policy": "standard",
        "period": "2022-04-08_to_2022-04-21",
        "documented_log": "log_standard_4_08_to_4_21",
    },
    "standard_late": {
        "policy": "standard",
        "period": "2022-04-22_to_2022-05-08",
        "documented_log": "log_standard_4_22_to_5_08",
    },
    "random": {
        "policy": "random",
        "period": "2022-04-22_to_2022-05-08",
        "documented_log": "log_random_4_22_to_5_08",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/daryumin/iberdov/Corpora"),
        help="Корневая директория с папками KuaiRand-*.",
    )
    parser.add_argument(
        "--version-root",
        type=Path,
        default=None,
        help="Явная директория KuaiRand-27K/KuaiRand-27K. Имеет приоритет над --data-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eda",
        help="Директория для JSON и компактных CSV-выходов.",
    )
    parser.add_argument(
        "--infer-schema-length",
        type=int,
        default=10_000,
        help="Число строк, по которым Polars определяет CSV-схему.",
    )
    parser.add_argument(
        "--sanity-limit",
        type=int,
        default=None,
        help="Читать не больше этого числа строк из каждого файла взаимодействий для smoke-тестов.",
    )
    parser.add_argument(
        "--skip-item-popularity",
        action="store_true",
        help="Пропустить агрегацию interactions-per-item, если ресурсов кластера недостаточно.",
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


def discover_interaction_sources(version_root: Path) -> list[dict[str, Any]]:
    groups = discover_kuairand_file_groups(version_root)
    sources: list[dict[str, Any]] = []
    for group_key, metadata in LOG_GROUPS.items():
        for path in groups[group_key]:
            sources.append(
                {
                    "group": group_key,
                    "policy": metadata["policy"],
                    "period": metadata["period"],
                    "documented_log": metadata["documented_log"],
                    "source_log": path.stem,
                    "path": path,
                    "size_bytes": path.stat().st_size,
                    "size": human_size(path.stat().st_size),
                }
            )
    return sorted(sources, key=lambda row: (row["policy"], row["period"], str(row["path"])))


def scan_source(source: dict[str, Any], infer_schema_length: int, sanity_limit: int | None) -> Any:
    pl = require_polars()
    lf = scan_table(source["path"], infer_schema_length=infer_schema_length)
    if sanity_limit is not None:
        lf = lf.limit(sanity_limit)
    return lf.with_columns(
        [
            pl.lit(source["policy"]).alias("policy"),
            pl.lit(source["period"]).alias("period"),
            pl.lit(source["documented_log"]).alias("documented_log"),
            pl.lit(source["source_log"]).alias("source_log"),
        ]
    )


def build_interaction_frame(
    sources: list[dict[str, Any]],
    infer_schema_length: int,
    sanity_limit: int | None,
) -> Any:
    scans = [
        scan_source(source, infer_schema_length=infer_schema_length, sanity_limit=sanity_limit)
        for source in sources
    ]
    if not scans:
        raise FileNotFoundError("Не найдены логи взаимодействий KuaiRand-27K.")
    return concat_lazy_frames(scans)


def base_aggregation(lf: Any, keys: list[str]) -> Any:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    exprs = [pl.len().alias("interactions")]
    if "user_id" in names:
        exprs.append(pl.col("user_id").n_unique().alias("users"))
    if "video_id" in names:
        exprs.append(pl.col("video_id").n_unique().alias("items"))
    if "date" in names:
        exprs.extend([pl.col("date").min().alias("date_min"), pl.col("date").max().alias("date_max")])
    if "time_ms" in names:
        time_col = pl.col("time_ms").cast(pl.Int64, strict=False)
        exprs.extend([time_col.min().alias("time_ms_min"), time_col.max().alias("time_ms_max")])
    if "tab" in names:
        exprs.append(pl.col("tab").n_unique().alias("unique_tabs"))
    return collect_lazy(lf.group_by(keys).agg(exprs).sort(keys))


def write_frame(output_dir: Path, prefix: str, name: str, frame: Any) -> str:
    path = output_dir / f"{prefix}_{name}.csv"
    frame.write_csv(path)
    return str(path)


def feedback_summary(lf: Any) -> Any:
    pl = require_polars()
    feedback_cols = available_columns(lf, FEEDBACK_COLUMNS)
    if not feedback_cols:
        return pl.DataFrame()

    long_frames = []
    for column in feedback_cols:
        value = pl.col(column).cast(pl.Float64, strict=False)
        long_frames.append(
            lf.group_by("policy")
            .agg(
                [
                    pl.len().alias("interactions"),
                    value.sum().alias("positive_count"),
                    value.mean().alias("positive_rate"),
                    pl.col(column).is_null().sum().alias("missing_count"),
                ]
            )
            .with_columns(pl.lit(column).alias("signal"))
            .select(
                [
                    "policy",
                    "signal",
                    "interactions",
                    "positive_count",
                    "positive_rate",
                    "missing_count",
                ]
            )
        )
    return collect_lazy(concat_lazy_frames(long_frames).sort(["policy", "signal"]))


def binary_feedback_quality(lf: Any) -> Any:
    pl = require_polars()
    feedback_cols = available_columns(lf, FEEDBACK_COLUMNS)
    if not feedback_cols:
        return pl.DataFrame()

    long_frames = []
    for column in feedback_cols:
        value = pl.col(column).cast(pl.Int64, strict=False)
        invalid = value.is_not_null() & (~value.is_in([0, 1]))
        long_frames.append(
            lf.group_by("policy")
            .agg(
                [
                    pl.len().alias("interactions"),
                    value.n_unique().alias("unique_values"),
                    value.min().alias("min_value"),
                    value.max().alias("max_value"),
                    pl.col(column).is_null().sum().alias("missing_count"),
                    invalid.sum().alias("invalid_count"),
                ]
            )
            .with_columns(pl.lit(column).alias("signal"))
            .select(
                [
                    "policy",
                    "signal",
                    "interactions",
                    "unique_values",
                    "min_value",
                    "max_value",
                    "missing_count",
                    "invalid_count",
                ]
            )
        )
    return collect_lazy(concat_lazy_frames(long_frames).sort(["policy", "signal"]))


def with_play_ratio(lf: Any) -> Any:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if not {"play_time_ms", "duration_ms"}.issubset(names):
        return lf
    return lf.with_columns(
        pl.when(pl.col("duration_ms").cast(pl.Float64, strict=False) > 0)
        .then(
            pl.col("play_time_ms").cast(pl.Float64, strict=False)
            / pl.col("duration_ms").cast(pl.Float64, strict=False)
        )
        .otherwise(None)
        .alias("play_ratio")
    )


def watch_time_summary(lf: Any) -> Any:
    pl = require_polars()
    wt = with_play_ratio(lf)
    names = set(lazy_schema_names(wt))
    columns = [column for column in (*WATCH_TIME_COLUMNS, "play_ratio") if column in names]
    if not columns:
        return pl.DataFrame()

    exprs = []
    if "duration_ms" in names:
        duration = pl.col("duration_ms").cast(pl.Float64, strict=False)
        exprs.extend(
            [
                (duration <= 0).sum().alias("duration_non_positive_count"),
                duration.is_null().sum().alias("duration_missing_count"),
            ]
        )
    for column in columns:
        value = pl.col(column).cast(pl.Float64, strict=False)
        exprs.extend(
            [
                value.count().alias(f"{column}_count"),
                value.mean().alias(f"{column}_mean"),
                value.quantile(0.5).alias(f"{column}_median"),
                value.quantile(0.75).alias(f"{column}_p75"),
                value.quantile(0.9).alias(f"{column}_p90"),
                value.quantile(0.95).alias(f"{column}_p95"),
                value.quantile(0.99).alias(f"{column}_p99"),
                value.max().alias(f"{column}_max"),
            ]
        )
    return collect_lazy(wt.group_by("policy").agg(exprs).sort("policy"))


def write_temporal_tables(lf: Any, output_dir: Path, prefix: str) -> dict[str, str]:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    outputs: dict[str, str] = {}

    if "date" in names:
        daily = collect_lazy(
            lf.group_by(["policy", "date"])
            .agg(pl.len().alias("interactions"))
            .sort(["date", "policy"])
        )
        outputs["daily"] = write_frame(output_dir, prefix, "daily", daily)

    if "hourmin" in names:
        hourly_lf = lf.with_columns(
            (pl.col("hourmin").cast(pl.Int64, strict=False) // 100).alias("hour")
        )
        hourly = collect_lazy(
            hourly_lf.group_by(["policy", "hour"])
            .agg(pl.len().alias("interactions"))
            .sort(["policy", "hour"])
        )
        outputs["hourly"] = write_frame(output_dir, prefix, "hourly", hourly)

    return outputs


def tab_summary(lf: Any) -> Any:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if "tab" not in names:
        return pl.DataFrame()

    exprs = [pl.len().alias("interactions")]
    if "user_id" in names:
        exprs.append(pl.col("user_id").n_unique().alias("users"))
    if "video_id" in names:
        exprs.append(pl.col("video_id").n_unique().alias("items"))
    for column in ("is_click", "long_view", "is_like"):
        if column in names:
            exprs.append(pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"{column}_rate"))

    table = collect_lazy(
        lf.group_by(["policy", "tab"])
        .agg(exprs)
        .sort(["policy", "interactions"], descending=[False, True])
    )
    totals = table.group_by("policy").agg(pl.col("interactions").sum().alias("policy_interactions"))
    return table.join(totals, on="policy").with_columns(
        (pl.col("interactions") / pl.col("policy_interactions")).alias("policy_share")
    )


def user_activity_tables(lf: Any) -> tuple[Any, Any, Any]:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if "user_id" not in names:
        empty = pl.DataFrame()
        return empty, empty, empty

    def user_exprs() -> list[Any]:
        exprs = [pl.len().alias("interactions")]
        if "video_id" in names:
            exprs.append(pl.col("video_id").n_unique().alias("unique_videos"))
        if "time_ms" in names:
            time_col = pl.col("time_ms").cast(pl.Int64, strict=False)
            exprs.extend([time_col.min().alias("first_time_ms"), time_col.max().alias("last_time_ms")])
        return exprs

    by_user = collect_lazy(lf.group_by("user_id").agg(user_exprs()).sort("user_id"))
    by_policy_user = collect_lazy(
        lf.group_by(["policy", "user_id"]).agg(user_exprs()).sort(["policy", "user_id"])
    )

    summary_rows: list[dict[str, Any]] = []
    def partition_key_label(key: Any) -> str:
        if isinstance(key, tuple) and len(key) == 1:
            return str(key[0])
        return str(key)

    partitions = by_policy_user.partition_by("policy", as_dict=True)
    frames = [("all", by_user), *[(partition_key_label(policy), sub) for policy, sub in partitions.items()]]
    for scope, frame in frames:
        interactions = frame["interactions"].to_numpy()
        summary_rows.append(
            {"scope": scope, "quantity": "interactions_per_user", **numeric_summary(interactions)}
        )
        if "unique_videos" in frame.columns:
            unique_videos = frame["unique_videos"].to_numpy()
            summary_rows.append(
                {"scope": scope, "quantity": "unique_videos_per_user", **numeric_summary(unique_videos)}
            )
    return by_user, by_policy_user, pl.DataFrame(summary_rows)


def item_popularity_summary(lf: Any) -> Any:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if not {"policy", "video_id"}.issubset(names):
        return pl.DataFrame()

    exprs = [pl.len().alias("interactions")]
    if "user_id" in names:
        exprs.append(pl.col("user_id").n_unique().alias("unique_users"))

    item_counts = lf.group_by(["policy", "video_id"]).agg(exprs)
    summary_exprs = [
        pl.len().alias("items"),
        pl.col("interactions").min().alias("interactions_min"),
        pl.col("interactions").mean().alias("interactions_mean"),
        pl.col("interactions").quantile(0.5).alias("interactions_median"),
        pl.col("interactions").quantile(0.75).alias("interactions_p75"),
        pl.col("interactions").quantile(0.9).alias("interactions_p90"),
        pl.col("interactions").quantile(0.95).alias("interactions_p95"),
        pl.col("interactions").quantile(0.99).alias("interactions_p99"),
        pl.col("interactions").max().alias("interactions_max"),
    ]
    if "user_id" in names:
        summary_exprs.extend(
            [
                pl.col("unique_users").mean().alias("unique_users_mean"),
                pl.col("unique_users").quantile(0.5).alias("unique_users_median"),
                pl.col("unique_users").quantile(0.99).alias("unique_users_p99"),
                pl.col("unique_users").max().alias("unique_users_max"),
            ]
        )
    return collect_lazy(item_counts.group_by("policy").agg(summary_exprs).sort("policy"))


def dataframe_records(frame: Any) -> list[dict[str, Any]]:
    if frame is None or frame.is_empty():
        return []
    return frame.to_dicts()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    prefix = "27k_sanity" if args.sanity_limit is not None else "27k"
    version_root = resolve_27k_root(args.data_root, args.version_root)

    if not version_root.exists():
        raise FileNotFoundError(
            f"KuaiRand-27K directory not found: {version_root}. "
            "Run this script on cHARISMa or pass --version-root."
        )

    inventory = dataset_inventory(version_root)
    sources = discover_interaction_sources(version_root)
    if not sources:
        raise FileNotFoundError(f"No 27K interaction logs were discovered under {version_root}.")

    interactions = build_interaction_frame(
        sources=sources,
        infer_schema_length=args.infer_schema_length,
        sanity_limit=args.sanity_limit,
    )

    source_counts = base_aggregation(interactions, ["policy", "period", "source_log"])
    policy_counts = base_aggregation(interactions, ["policy"])
    feedback = feedback_summary(interactions)
    binary_quality = binary_feedback_quality(interactions)
    watch_time = watch_time_summary(interactions)
    tabs = tab_summary(interactions)
    temporal_outputs = write_temporal_tables(interactions, output_dir, prefix)
    by_user, by_policy_user, sequence_summary = user_activity_tables(interactions)
    item_summary = None if args.skip_item_popularity else item_popularity_summary(interactions)

    csv_outputs = {
        "source_counts": write_frame(output_dir, prefix, "source_counts", source_counts),
        "policy_counts": write_frame(output_dir, prefix, "policy_counts", policy_counts),
        "feedback_summary": write_frame(output_dir, prefix, "feedback_summary", feedback),
        "binary_feedback_quality": write_frame(output_dir, prefix, "binary_feedback_quality", binary_quality),
        "watch_time_summary": write_frame(output_dir, prefix, "watch_time_summary", watch_time),
        "tab_summary": write_frame(output_dir, prefix, "tab_summary", tabs),
        "user_activity": write_frame(output_dir, prefix, "user_activity", by_user),
        "policy_user_activity": write_frame(output_dir, prefix, "policy_user_activity", by_policy_user),
        "sequence_summary": write_frame(output_dir, prefix, "sequence_summary", sequence_summary),
        **temporal_outputs,
    }
    if item_summary is not None:
        csv_outputs["item_popularity_summary"] = write_frame(
            output_dir, prefix, "item_popularity_summary", item_summary
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "sanity" if args.sanity_limit is not None else "full",
        "sanity_limit_per_file": args.sanity_limit,
        "project_root": str(PROJECT_ROOT),
        "data_root": str(args.data_root),
        "version_root": str(version_root),
        "disk_size_bytes": total_size_bytes(inventory),
        "disk_size": human_size(total_size_bytes(inventory)),
        "interaction_sources": [
            {key: (str(value) if isinstance(value, Path) else value) for key, value in source.items()}
            for source in sources
        ],
        "csv_outputs": csv_outputs,
        "source_counts": dataframe_records(source_counts),
        "policy_counts": dataframe_records(policy_counts),
        "feedback_summary": dataframe_records(feedback),
        "binary_feedback_quality": dataframe_records(binary_quality),
        "watch_time_summary": dataframe_records(watch_time),
        "sequence_summary": dataframe_records(sequence_summary),
        "item_popularity_summary": dataframe_records(item_summary),
        "tab_summary_preview": dataframe_records(tabs.head(50)) if not tabs.is_empty() else [],
        "notes": [
            "Статистика из документации / README не смешивается с вычисленными результатами.",
            "Файлы video statistic features не сканируются этим скриптом, потому что это агрегированные item-level признаки с возможным temporal leakage.",
            "Агрегация item popularity группирует данные по video_id; ее можно пропустить через --skip-item-popularity, если ресурсов кластера недостаточно.",
        ],
    }

    json_path = output_dir / f"{prefix}_summary.json"
    write_json(json_path, make_jsonable(summary))
    print(f"Записано: {json_path}")
    for name, path in sorted(csv_outputs.items()):
        print(f"Записано {name}: {path}")


if __name__ == "__main__":
    main()
