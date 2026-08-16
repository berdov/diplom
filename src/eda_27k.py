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

DOCUMENTED_27K_COUNTS: dict[str, int] = {
    "README standard items": 32_038_725,
    "README standard interactions": 322_278_385,
    "README random interactions": 1_186_059,
    "README users": 27_285,
    "README random items": 7_583,
}

PLAY_RATIO_THRESHOLDS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)


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
                    value.count().alias("non_null_count"),
                    pl.col(column).is_null().sum().alias("missing_count"),
                ]
            )
            .with_columns(
                [
                    (pl.col("positive_count") / pl.col("interactions")).alias("positive_rate"),
                    (pl.col("positive_count") / pl.col("non_null_count")).alias(
                        "positive_rate_non_null"
                    ),
                ]
            )
            .with_columns(pl.lit(column).alias("signal"))
            .select(
                [
                    "policy",
                    "signal",
                    "interactions",
                    "positive_count",
                    "non_null_count",
                    "positive_rate",
                    "positive_rate_non_null",
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


def quantile_exprs(value: Any, prefix: str) -> list[Any]:
    return [
        value.count().alias(f"{prefix}_count"),
        value.mean().alias(f"{prefix}_mean"),
        value.quantile(0.5).alias(f"{prefix}_median"),
        value.quantile(0.75).alias(f"{prefix}_p75"),
        value.quantile(0.9).alias(f"{prefix}_p90"),
        value.quantile(0.95).alias(f"{prefix}_p95"),
        value.quantile(0.99).alias(f"{prefix}_p99"),
        value.quantile(0.999).alias(f"{prefix}_p999"),
        value.max().alias(f"{prefix}_max"),
    ]


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
        exprs.extend(quantile_exprs(value, column))
    return collect_lazy(wt.group_by("policy").agg(exprs).sort("policy"))


def duration_summary(lf: Any, keys: list[str]) -> Any:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if "duration_ms" not in names:
        return pl.DataFrame()

    duration = pl.col("duration_ms").cast(pl.Float64, strict=False)
    non_positive = duration <= 0
    exprs = [
        pl.len().alias("interactions"),
        duration.is_null().sum().alias("duration_null_count"),
        (duration == 0).sum().alias("duration_zero_count"),
        (duration < 0).sum().alias("duration_negative_count"),
        non_positive.sum().alias("duration_non_positive_count"),
    ]
    if "user_id" in names:
        exprs.append(pl.col("user_id").filter(non_positive).n_unique().alias("affected_users"))
    if "video_id" in names:
        exprs.append(pl.col("video_id").filter(non_positive).n_unique().alias("affected_videos"))

    table = lf.group_by(keys).agg(exprs)
    return collect_lazy(
        table.with_columns(
            [
                (pl.col("duration_null_count") / pl.col("interactions")).alias("duration_null_share"),
                (pl.col("duration_zero_count") / pl.col("interactions")).alias("duration_zero_share"),
                (pl.col("duration_negative_count") / pl.col("interactions")).alias("duration_negative_share"),
                (pl.col("duration_non_positive_count") / pl.col("interactions")).alias(
                    "duration_non_positive_share"
                ),
            ]
        ).sort(keys)
    )


def play_ratio_summary(lf: Any) -> Any:
    pl = require_polars()
    wt = with_play_ratio(lf)
    names = set(lazy_schema_names(wt))
    if "play_ratio" not in names:
        return pl.DataFrame()

    value = pl.col("play_ratio").cast(pl.Float64, strict=False)
    exprs = quantile_exprs(value, "play_ratio")
    for threshold in PLAY_RATIO_THRESHOLDS:
        suffix = str(threshold).rstrip("0").rstrip(".").replace(".", "_")
        exprs.append((value > threshold).sum().alias(f"play_ratio_gt_{suffix}_count"))

    table = wt.group_by("policy").agg(exprs)
    share_exprs = []
    for threshold in PLAY_RATIO_THRESHOLDS:
        suffix = str(threshold).rstrip("0").rstrip(".").replace(".", "_")
        share_exprs.append(
            (pl.col(f"play_ratio_gt_{suffix}_count") / pl.col("play_ratio_count")).alias(
                f"play_ratio_gt_{suffix}_share"
            )
        )
    return collect_lazy(table.with_columns(share_exprs).sort("policy"))


def write_temporal_tables(lf: Any, output_dir: Path, prefix: str) -> dict[str, str]:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    outputs: dict[str, str] = {}

    if "date" in names:
        daily_exprs = [pl.len().alias("interactions")]
        if "user_id" in names:
            daily_exprs.append(pl.col("user_id").n_unique().alias("users"))
        if "video_id" in names:
            daily_exprs.append(pl.col("video_id").n_unique().alias("items"))
        for column in ("is_click", "long_view", "is_like"):
            if column in names:
                daily_exprs.append(pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"{column}_rate"))
        daily = collect_lazy(
            lf.group_by(["policy", "date"])
            .agg(daily_exprs)
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


def temporal_period_summary(lf: Any) -> Any:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if "period" not in names:
        return pl.DataFrame()

    exprs = [pl.len().alias("interactions")]
    if "user_id" in names:
        exprs.append(pl.col("user_id").n_unique().alias("users"))
    if "video_id" in names:
        exprs.append(pl.col("video_id").n_unique().alias("items"))
    if "date" in names:
        exprs.extend([pl.col("date").min().alias("date_min"), pl.col("date").max().alias("date_max")])
    for column in ("is_click", "long_view", "is_like"):
        if column in names:
            exprs.append(pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"{column}_rate"))
    if "play_time_ms" in names:
        play_time = pl.col("play_time_ms").cast(pl.Float64, strict=False)
        exprs.extend(
            [
                play_time.mean().alias("play_time_ms_mean"),
                play_time.quantile(0.5).alias("play_time_ms_median"),
            ]
        )
    if "duration_ms" in names:
        duration = pl.col("duration_ms").cast(pl.Float64, strict=False)
        exprs.extend(
            [
                duration.mean().alias("duration_ms_mean"),
                duration.quantile(0.5).alias("duration_ms_median"),
                (duration <= 0).sum().alias("duration_non_positive_count"),
            ]
        )
    return collect_lazy(lf.group_by(["policy", "period"]).agg(exprs).sort(["policy", "period"]))


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
    for column in ("is_click", "long_view", "is_like", "is_hate"):
        if column in names:
            exprs.append(pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"{column}_rate"))
    if "play_time_ms" in names:
        exprs.append(
            pl.col("play_time_ms")
            .cast(pl.Float64, strict=False)
            .quantile(0.5)
            .alias("play_time_ms_median")
        )
    if "duration_ms" in names:
        duration = pl.col("duration_ms").cast(pl.Float64, strict=False)
        exprs.extend(
            [
                (duration <= 0).sum().alias("duration_non_positive_count"),
                (duration == 0).sum().alias("duration_zero_count"),
                (duration < 0).sum().alias("duration_negative_count"),
            ]
        )

    table = collect_lazy(
        lf.group_by(["policy", "tab"])
        .agg(exprs)
        .sort(["policy", "interactions"], descending=[False, True])
    )
    totals = table.group_by("policy").agg(pl.col("interactions").sum().alias("policy_interactions"))
    return table.join(totals, on="policy").with_columns(
        [
            (pl.col("interactions") / pl.col("policy_interactions")).alias("policy_share"),
            (pl.col("duration_non_positive_count") / pl.col("interactions")).alias(
                "duration_non_positive_share"
            )
            if "duration_ms" in names
            else pl.lit(None).alias("duration_non_positive_share"),
        ]
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
        if "date" in names:
            date_col = pl.col("date").cast(pl.Int64, strict=False)
            exprs.extend([date_col.min().alias("first_date"), date_col.max().alias("last_date")])
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


def max_user_sequence_table(by_user: Any, by_policy_user: Any) -> Any:
    pl = require_polars()
    frames = []
    if by_user is not None and not by_user.is_empty():
        frames.append(
            by_user.sort("interactions", descending=True)
            .head(10)
            .with_columns(pl.lit("all").alias("scope"))
        )
    if by_policy_user is not None and not by_policy_user.is_empty():
        for policy, frame in by_policy_user.partition_by("policy", as_dict=True).items():
            policy_label = policy[0] if isinstance(policy, tuple) else policy
            frames.append(
                frame.sort("interactions", descending=True)
                .head(10)
                .with_columns(pl.lit(str(policy_label)).alias("scope"))
            )
    if not frames:
        return pl.DataFrame()
    columns = [
        "scope",
        "policy",
        "user_id",
        "interactions",
        "unique_videos",
        "first_date",
        "last_date",
        "first_time_ms",
        "last_time_ms",
    ]
    table = pl.concat(frames, how="diagonal_relaxed")
    available = [column for column in columns if column in table.columns]
    return table.select(available).sort(["scope", "interactions"], descending=[False, True])


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


def collect_item_counts(lf: Any) -> Any:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if not {"policy", "video_id"}.issubset(names):
        return pl.DataFrame()

    exprs = [pl.len().alias("interactions")]
    if "user_id" in names:
        exprs.append(pl.col("user_id").n_unique().alias("unique_users"))
    if "duration_ms" in names:
        duration = pl.col("duration_ms").cast(pl.Float64, strict=False)
        exprs.extend(
            [
                (duration <= 0).sum().alias("duration_non_positive_count"),
                (duration == 0).sum().alias("duration_zero_count"),
                (duration < 0).sum().alias("duration_negative_count"),
                duration.is_null().sum().alias("duration_null_count"),
            ]
        )
    return collect_lazy(lf.group_by(["policy", "video_id"]).agg(exprs).sort(["policy", "video_id"]))


def gini(values: np.ndarray) -> float | None:
    finite = values.astype(float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    total = finite.sum()
    if total == 0:
        return 0.0
    ordered = np.sort(finite)
    n = ordered.size
    ranks = np.arange(1, n + 1, dtype=float)
    return float(np.sum((2 * ranks - n - 1) * ordered) / (n * total))


def item_popularity_summary_from_counts(item_counts: Any) -> Any:
    pl = require_polars()
    if item_counts is None or item_counts.is_empty():
        return pl.DataFrame()

    summary_exprs = [
        pl.len().alias("items"),
        pl.col("interactions").min().alias("interactions_min"),
        pl.col("interactions").mean().alias("interactions_mean"),
        pl.col("interactions").std().alias("interactions_std"),
        pl.col("interactions").quantile(0.5).alias("interactions_median"),
        pl.col("interactions").quantile(0.75).alias("interactions_p75"),
        pl.col("interactions").quantile(0.9).alias("interactions_p90"),
        pl.col("interactions").quantile(0.95).alias("interactions_p95"),
        pl.col("interactions").quantile(0.99).alias("interactions_p99"),
        pl.col("interactions").quantile(0.999).alias("interactions_p999"),
        pl.col("interactions").max().alias("interactions_max"),
    ]
    if "unique_users" in item_counts.columns:
        summary_exprs.extend(
            [
                pl.col("unique_users").mean().alias("unique_users_mean"),
                pl.col("unique_users").quantile(0.5).alias("unique_users_median"),
                pl.col("unique_users").quantile(0.99).alias("unique_users_p99"),
                pl.col("unique_users").max().alias("unique_users_max"),
            ]
        )

    summary = item_counts.group_by("policy").agg(summary_exprs).sort("policy")
    gini_rows = []
    for policy, frame in item_counts.partition_by("policy", as_dict=True).items():
        policy_label = policy[0] if isinstance(policy, tuple) else policy
        values = frame["interactions"].to_numpy()
        gini_rows.append({"policy": str(policy_label), "interactions_gini": gini(values)})
    gini_frame = pl.DataFrame(gini_rows)
    return summary.join(gini_frame, on="policy", how="left").with_columns(
        (pl.col("interactions_std") / pl.col("interactions_mean")).alias("interactions_cv")
    )


def feature_id_frame(paths: list[Path], id_column: str, infer_schema_length: int) -> Any | None:
    pl = require_polars()
    scans = []
    for path in paths:
        lf = scan_table(path, infer_schema_length=infer_schema_length)
        names = set(lazy_schema_names(lf))
        if id_column in names:
            scans.append(
                lf.select(pl.col(id_column).cast(pl.Int64, strict=False).alias(id_column))
                .drop_nulls()
            )
    if not scans:
        return None
    return collect_lazy(concat_lazy_frames(scans).unique().sort(id_column))


def feature_id_diagnostics(paths: list[Path], id_column: str, infer_schema_length: int) -> dict[str, Any]:
    pl = require_polars()
    scans = []
    for path in paths:
        lf = scan_table(path, infer_schema_length=infer_schema_length)
        names = set(lazy_schema_names(lf))
        if id_column in names:
            scans.append(lf.select(pl.col(id_column).cast(pl.Int64, strict=False).alias(id_column)))
    if not scans:
        return {
            "files": len(paths),
            "rows": None,
            "unique_ids": None,
            "missing_ids": None,
            "duplicate_id_rows": None,
        }
    ids = concat_lazy_frames(scans)
    diagnostics = collect_lazy(
        ids.select(
            [
                pl.len().alias("rows"),
                pl.col(id_column).n_unique().alias("unique_ids"),
                pl.col(id_column).is_null().sum().alias("missing_ids"),
            ]
        )
    ).to_dicts()[0]
    duplicate_rows = collect_lazy(
        ids.drop_nulls()
        .group_by(id_column)
        .agg(pl.len().alias("rows_per_id"))
        .select(
            pl.when(pl.col("rows_per_id") > 1)
            .then(pl.col("rows_per_id") - 1)
            .otherwise(0)
            .sum()
            .alias("duplicate_id_rows")
        )
    ).to_dicts()[0]
    return {"files": len(paths), **diagnostics, **duplicate_rows}


def item_universe_comparison(
    item_counts: Any,
    video_basic_ids: Any | None,
    video_statistic_ids: Any | None,
) -> Any:
    pl = require_polars()
    rows: list[dict[str, Any]] = [
        {"metric": "README standard items", "count": DOCUMENTED_27K_COUNTS["README standard items"]},
    ]
    if item_counts is None or item_counts.is_empty():
        return pl.DataFrame(rows)

    standard_ids = item_counts.filter(pl.col("policy") == "standard").select("video_id").unique()
    random_ids = item_counts.filter(pl.col("policy") == "random").select("video_id").unique()
    rows.extend(
        [
            {"metric": "Unique standard interaction video_id", "count": standard_ids.height},
            {"metric": "Unique random interaction video_id", "count": random_ids.height},
        ]
    )
    if video_basic_ids is not None and not video_basic_ids.is_empty():
        rows.extend(
            [
                {"metric": "Unique video_features_basic video_id", "count": video_basic_ids.height},
                {
                    "metric": "Items in features not in standard logs",
                    "count": video_basic_ids.join(standard_ids, on="video_id", how="anti").height,
                },
                {
                    "metric": "Items in standard logs not in basic features",
                    "count": standard_ids.join(video_basic_ids, on="video_id", how="anti").height,
                },
            ]
        )
    if video_statistic_ids is not None and not video_statistic_ids.is_empty():
        rows.extend(
            [
                {"metric": "Unique video_features_statistic video_id", "count": video_statistic_ids.height},
                {
                    "metric": "Items in statistic features not in standard logs",
                    "count": video_statistic_ids.join(standard_ids, on="video_id", how="anti").height,
                },
                {
                    "metric": "Items in standard logs not in statistic features",
                    "count": standard_ids.join(video_statistic_ids, on="video_id", how="anti").height,
                },
            ]
        )
    return pl.DataFrame(rows)


def video_feature_coverage(item_counts: Any, feature_ids: Any | None, feature_name: str) -> Any:
    pl = require_polars()
    if item_counts is None or item_counts.is_empty() or feature_ids is None or feature_ids.is_empty():
        return pl.DataFrame()

    flag = f"has_{feature_name}"
    joined = item_counts.join(feature_ids.with_columns(pl.lit(True).alias(flag)), on="video_id", how="left")
    joined = joined.with_columns(pl.col(flag).fill_null(False).alias(flag))
    table = joined.group_by("policy").agg(
        [
            pl.col("interactions").sum().alias("interactions"),
            pl.len().alias("unique_interaction_videos"),
            pl.when(pl.col(flag)).then(pl.col("interactions")).otherwise(0).sum().alias(
                "interactions_with_feature"
            ),
            pl.col("video_id").filter(pl.col(flag)).n_unique().alias("unique_interaction_videos_with_feature"),
        ]
    )
    return table.with_columns(
        [
            pl.lit(feature_name).alias("feature_table"),
            (pl.col("interactions_with_feature") / pl.col("interactions")).alias(
                "interaction_coverage_share"
            ),
            (pl.col("unique_interaction_videos_with_feature") / pl.col("unique_interaction_videos")).alias(
                "unique_video_coverage_share"
            ),
        ]
    ).select(
        [
            "feature_table",
            "policy",
            "interactions",
            "interactions_with_feature",
            "interaction_coverage_share",
            "unique_interaction_videos",
            "unique_interaction_videos_with_feature",
            "unique_video_coverage_share",
        ]
    )


def duration_metadata_coverage(item_counts: Any, video_basic_ids: Any | None) -> Any:
    pl = require_polars()
    if (
        item_counts is None
        or item_counts.is_empty()
        or video_basic_ids is None
        or video_basic_ids.is_empty()
        or "duration_non_positive_count" not in item_counts.columns
    ):
        return pl.DataFrame()
    joined = item_counts.join(
        video_basic_ids.with_columns(pl.lit(True).alias("has_video_basic")),
        on="video_id",
        how="left",
    ).with_columns(pl.col("has_video_basic").fill_null(False).alias("has_video_basic"))
    table = joined.group_by(["policy", "has_video_basic"]).agg(
        [
            pl.col("interactions").sum().alias("interactions"),
            pl.col("duration_non_positive_count").sum().alias("duration_non_positive_count"),
            pl.len().alias("unique_videos"),
        ]
    )
    return table.with_columns(
        (pl.col("duration_non_positive_count") / pl.col("interactions")).alias(
            "duration_non_positive_share"
        )
    ).sort(["policy", "has_video_basic"])


def feature_coverage(
    item_counts: Any,
    by_policy_user: Any,
    video_basic_ids: Any | None,
    video_statistic_ids: Any | None,
    user_feature_ids: Any | None,
) -> Any:
    pl = require_polars()
    frames = []
    for feature_name, ids in (
        ("video_basic", video_basic_ids),
        ("video_statistic", video_statistic_ids),
    ):
        coverage = video_feature_coverage(item_counts, ids, feature_name)
        if not coverage.is_empty():
            frames.append(coverage)

    if by_policy_user is not None and not by_policy_user.is_empty() and user_feature_ids is not None:
        flag = "has_user_features"
        joined = by_policy_user.select(["policy", "user_id", "interactions"]).join(
            user_feature_ids.with_columns(pl.lit(True).alias(flag)),
            on="user_id",
            how="left",
        )
        joined = joined.with_columns(pl.col(flag).fill_null(False).alias(flag))
        user_coverage = joined.group_by("policy").agg(
            [
                pl.col("interactions").sum().alias("interactions"),
                pl.len().alias("unique_interaction_users"),
                pl.when(pl.col(flag)).then(pl.col("interactions")).otherwise(0).sum().alias(
                    "interactions_with_feature"
                ),
                pl.col("user_id").filter(pl.col(flag)).n_unique().alias(
                    "unique_interaction_users_with_feature"
                ),
            ]
        )
        user_coverage = user_coverage.with_columns(
            [
                pl.lit("user_features").alias("feature_table"),
                (pl.col("interactions_with_feature") / pl.col("interactions")).alias(
                    "interaction_coverage_share"
                ),
                (
                    pl.col("unique_interaction_users_with_feature")
                    / pl.col("unique_interaction_users")
                ).alias("unique_user_coverage_share"),
            ]
        ).rename(
            {
                "unique_interaction_users": "unique_interaction_entities",
                "unique_interaction_users_with_feature": "unique_interaction_entities_with_feature",
                "unique_user_coverage_share": "unique_entity_coverage_share",
            }
        )
        frames.append(user_coverage)

    normalized_frames = []
    for frame in frames:
        rename_map = {
            source: target
            for source, target in {
                "unique_interaction_videos": "unique_interaction_entities",
                "unique_interaction_videos_with_feature": "unique_interaction_entities_with_feature",
                "unique_video_coverage_share": "unique_entity_coverage_share",
            }.items()
            if source in frame.columns
        }
        normalized = frame.rename(rename_map) if rename_map else frame
        normalized_frames.append(normalized)
    if not normalized_frames:
        return pl.DataFrame()
    return pl.concat(normalized_frames, how="diagonal_relaxed").sort(["feature_table", "policy"])


def duplicate_key_summary(lf: Any) -> Any:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    key_sets = [
        ("user_id+video_id+time_ms", ["user_id", "video_id", "time_ms"]),
        ("user_id+time_ms", ["user_id", "time_ms"]),
    ]
    frames = []
    for key_name, keys in key_sets:
        if not all(key in names for key in keys):
            continue
        grouped = lf.group_by(["policy", *keys]).agg(pl.len().alias("rows_per_key"))
        summary = grouped.group_by("policy").agg(
            [
                pl.len().alias("distinct_keys"),
                (pl.col("rows_per_key") > 1).sum().alias("duplicate_keys"),
                pl.when(pl.col("rows_per_key") > 1)
                .then(pl.col("rows_per_key") - 1)
                .otherwise(0)
                .sum()
                .alias("extra_rows_over_unique_keys"),
                pl.col("rows_per_key").max().alias("max_rows_per_key"),
            ]
        )
        frames.append(summary.with_columns(pl.lit(key_name).alias("key")).select(
            [
                "key",
                "policy",
                "distinct_keys",
                "duplicate_keys",
                "extra_rows_over_unique_keys",
                "max_rows_per_key",
            ]
        ))
    if not frames:
        return pl.DataFrame()
    return collect_lazy(concat_lazy_frames(frames).sort(["key", "policy"]))


def max_user_duplicate_summary(lf: Any, max_user_table: Any) -> Any:
    pl = require_polars()
    names = set(lazy_schema_names(lf))
    if max_user_table is None or max_user_table.is_empty() or not {"user_id", "video_id", "time_ms"}.issubset(names):
        return pl.DataFrame()
    all_rows = max_user_table.filter(pl.col("scope") == "all")
    if all_rows.is_empty():
        return pl.DataFrame()
    user_id = all_rows.sort("interactions", descending=True)["user_id"][0]
    user_lf = lf.filter(pl.col("user_id") == user_id)
    grouped = user_lf.group_by(["video_id", "time_ms"]).agg(pl.len().alias("rows_per_key"))
    summary = collect_lazy(
        grouped.select(
            [
                pl.lit(user_id).alias("user_id"),
                pl.len().alias("distinct_video_time_keys"),
                (pl.col("rows_per_key") > 1).sum().alias("duplicate_video_time_keys"),
                pl.when(pl.col("rows_per_key") > 1)
                .then(pl.col("rows_per_key") - 1)
                .otherwise(0)
                .sum()
                .alias("extra_rows_over_unique_video_time_keys"),
                pl.col("rows_per_key").max().alias("max_rows_per_video_time_key"),
            ]
        )
    )
    return summary


def scalar_from_frame(frame: Any, filters: dict[str, Any], column: str) -> float | int | None:
    if frame is None or frame.is_empty() or column not in frame.columns:
        return None
    sub = frame
    for key, value in filters.items():
        if key not in sub.columns:
            return None
        sub = sub.filter(require_polars().col(key) == value)
    if sub.is_empty():
        return None
    return sub[column][0]


def standard_random_comparison(
    policy_counts: Any,
    feedback: Any,
    watch_time: Any,
    play_ratio: Any,
    sequence_summary: Any,
    item_summary: Any,
) -> Any:
    pl = require_polars()
    rows: list[dict[str, Any]] = []

    def add(metric: str, standard: Any, random: Any) -> None:
        absolute_difference = None
        relative_ratio = None
        if standard is not None and random is not None:
            absolute_difference = float(standard) - float(random)
            if float(random) != 0:
                relative_ratio = float(standard) / float(random)
        rows.append(
            {
                "metric": metric,
                "standard": standard,
                "random": random,
                "absolute_difference_standard_minus_random": absolute_difference,
                "relative_ratio_standard_over_random": relative_ratio,
            }
        )

    for column in ("interactions", "users", "items"):
        add(
            column,
            scalar_from_frame(policy_counts, {"policy": "standard"}, column),
            scalar_from_frame(policy_counts, {"policy": "random"}, column),
        )

    for signal in FEEDBACK_COLUMNS:
        add(
            f"{signal}_rate",
            scalar_from_frame(feedback, {"policy": "standard", "signal": signal}, "positive_rate"),
            scalar_from_frame(feedback, {"policy": "random", "signal": signal}, "positive_rate"),
        )

    for column in ("play_time_ms", "duration_ms"):
        for statistic in ("median", "mean"):
            add(
                f"{column}_{statistic}",
                scalar_from_frame(watch_time, {"policy": "standard"}, f"{column}_{statistic}"),
                scalar_from_frame(watch_time, {"policy": "random"}, f"{column}_{statistic}"),
            )

    for statistic in ("median", "mean"):
        add(
            f"valid_play_ratio_{statistic}",
            scalar_from_frame(play_ratio, {"policy": "standard"}, f"play_ratio_{statistic}"),
            scalar_from_frame(play_ratio, {"policy": "random"}, f"play_ratio_{statistic}"),
        )

    for statistic in ("median", "p95", "p99"):
        add(
            f"interactions_per_user_{statistic}",
            scalar_from_frame(
                sequence_summary,
                {"scope": "standard", "quantity": "interactions_per_user"},
                statistic,
            ),
            scalar_from_frame(
                sequence_summary,
                {"scope": "random", "quantity": "interactions_per_user"},
                statistic,
            ),
        )

    for statistic in ("median", "p95", "p99", "max"):
        add(
            f"item_popularity_interactions_{statistic}",
            scalar_from_frame(item_summary, {"policy": "standard"}, f"interactions_{statistic}"),
            scalar_from_frame(item_summary, {"policy": "random"}, f"interactions_{statistic}"),
        )

    return pl.DataFrame(rows)


def dataframe_records(frame: Any) -> list[dict[str, Any]]:
    if frame is None or frame.is_empty():
        return []
    return frame.to_dicts()


def main() -> None:
    args = parse_args()
    pl = require_polars()
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
    duration_policy = duration_summary(interactions, ["policy"])
    duration_by_tab = duration_summary(interactions, ["policy", "tab"])
    duration_by_date = duration_summary(interactions, ["policy", "date"])
    play_ratio = play_ratio_summary(interactions)
    tabs = tab_summary(interactions)
    temporal_outputs = write_temporal_tables(interactions, output_dir, prefix)
    period_summary = temporal_period_summary(interactions)
    by_user, by_policy_user, sequence_summary = user_activity_tables(interactions)
    max_user_sequence = max_user_sequence_table(by_user, by_policy_user)
    item_counts = None if args.skip_item_popularity else collect_item_counts(interactions)
    item_summary = (
        None
        if args.skip_item_popularity or item_counts is None
        else item_popularity_summary_from_counts(item_counts)
    )

    groups = discover_kuairand_file_groups(version_root)
    video_basic_ids = feature_id_frame(groups["video_basic"], "video_id", args.infer_schema_length)
    video_statistic_ids = feature_id_frame(
        groups["video_statistics"], "video_id", args.infer_schema_length
    )
    user_feature_ids = feature_id_frame(groups["user_features"], "user_id", args.infer_schema_length)
    item_universe = (
        pl.DataFrame()
        if item_counts is None
        else item_universe_comparison(item_counts, video_basic_ids, video_statistic_ids)
    )
    coverage = (
        pl.DataFrame()
        if item_counts is None
        else feature_coverage(
            item_counts,
            by_policy_user,
            video_basic_ids,
            video_statistic_ids,
            user_feature_ids,
        )
    )
    duration_metadata = (
        pl.DataFrame() if item_counts is None else duration_metadata_coverage(item_counts, video_basic_ids)
    )
    duplicates = duplicate_key_summary(interactions)
    max_user_duplicates = max_user_duplicate_summary(interactions, max_user_sequence)
    policy_comparison = standard_random_comparison(
        policy_counts,
        feedback,
        watch_time,
        play_ratio,
        sequence_summary,
        item_summary,
    )

    csv_outputs = {
        "source_counts": write_frame(output_dir, prefix, "source_counts", source_counts),
        "policy_counts": write_frame(output_dir, prefix, "policy_counts", policy_counts),
        "feedback_summary": write_frame(output_dir, prefix, "feedback_summary", feedback),
        "binary_feedback_quality": write_frame(output_dir, prefix, "binary_feedback_quality", binary_quality),
        "watch_time_summary": write_frame(output_dir, prefix, "watch_time_summary", watch_time),
        "duration_summary": write_frame(output_dir, prefix, "duration_summary", duration_policy),
        "duration_by_tab": write_frame(output_dir, prefix, "duration_by_tab", duration_by_tab),
        "duration_by_date": write_frame(output_dir, prefix, "duration_by_date", duration_by_date),
        "play_ratio_summary": write_frame(output_dir, prefix, "play_ratio_summary", play_ratio),
        "tab_summary": write_frame(output_dir, prefix, "tab_summary", tabs),
        "user_activity": write_frame(output_dir, prefix, "user_activity", by_user),
        "policy_user_activity": write_frame(output_dir, prefix, "policy_user_activity", by_policy_user),
        "sequence_summary": write_frame(output_dir, prefix, "sequence_summary", sequence_summary),
        "max_user_sequence": write_frame(output_dir, prefix, "max_user_sequence", max_user_sequence),
        "temporal_period_summary": write_frame(
            output_dir, prefix, "temporal_period_summary", period_summary
        ),
        "duplicate_key_summary": write_frame(output_dir, prefix, "duplicate_key_summary", duplicates),
        "max_user_duplicate_summary": write_frame(
            output_dir, prefix, "max_user_duplicate_summary", max_user_duplicates
        ),
        "item_universe_comparison": write_frame(
            output_dir, prefix, "item_universe_comparison", item_universe
        ),
        "feature_coverage": write_frame(output_dir, prefix, "feature_coverage", coverage),
        "duration_metadata_coverage": write_frame(
            output_dir, prefix, "duration_metadata_coverage", duration_metadata
        ),
        "standard_random_comparison": write_frame(
            output_dir, prefix, "standard_random_comparison", policy_comparison
        ),
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
        "duration_summary": dataframe_records(duration_policy),
        "play_ratio_summary": dataframe_records(play_ratio),
        "sequence_summary": dataframe_records(sequence_summary),
        "item_popularity_summary": dataframe_records(item_summary),
        "item_universe_comparison": dataframe_records(item_universe),
        "feature_coverage": dataframe_records(coverage),
        "duration_metadata_coverage": dataframe_records(duration_metadata),
        "duplicate_key_summary": dataframe_records(duplicates),
        "max_user_sequence": dataframe_records(max_user_sequence),
        "max_user_duplicate_summary": dataframe_records(max_user_duplicates),
        "temporal_period_summary": dataframe_records(period_summary),
        "standard_random_comparison": dataframe_records(policy_comparison),
        "tab_summary_preview": dataframe_records(tabs.head(50)) if not tabs.is_empty() else [],
        "feature_file_diagnostics": {
            "video_basic": feature_id_diagnostics(
                groups["video_basic"], "video_id", args.infer_schema_length
            ),
            "video_statistic": feature_id_diagnostics(
                groups["video_statistics"], "video_id", args.infer_schema_length
            ),
            "user_features": feature_id_diagnostics(
                groups["user_features"], "user_id", args.infer_schema_length
            ),
        },
        "notes": [
            "Статистика из документации / README не смешивается с вычисленными результатами.",
            "Файлы video statistic features не сканируются этим скриптом, потому что это агрегированные item-level признаки с возможным temporal leakage.",
            "Агрегация item popularity группирует данные по video_id; ее можно пропустить через --skip-item-popularity, если ресурсов кластера недостаточно.",
            "Основной feedback positive_rate считается как positive_count / interactions; positive_rate_non_null также сохранён для явной проверки null handling.",
            "play_ratio считается только для duration_ms > 0; значения выше 1 сохраняются как валидные наблюдения для анализа повторного/длинного просмотра.",
        ],
    }

    json_path = output_dir / f"{prefix}_summary.json"
    write_json(json_path, make_jsonable(summary))
    print(f"Записано: {json_path}")
    for name, path in sorted(csv_outputs.items()):
        print(f"Записано {name}: {path}")


if __name__ == "__main__":
    main()
