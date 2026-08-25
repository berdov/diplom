#!/usr/bin/env python3
"""Audit KuaiRand Protocol B behavior targets without training a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_DIR = Path("/home/daryumin/iberdov/diplom/data/processed/protocol_b")
DEFAULT_PROTOCOL_MANIFEST = Path("/home/daryumin/iberdov/diplom/outputs/data/protocol_b_manifest.json")
DEFAULT_SOURCE_LOG = Path(
    "/home/daryumin/iberdov/Corpora/KuaiRand-Pure/KuaiRand-Pure/data/"
    "log_standard_4_08_to_4_21_pure.csv"
)
DEFAULT_OUTPUT_DIR = Path("/home/daryumin/iberdov/diplom/data/processed/protocol_b_multitask")

EXPECTED_USERS = 23_951
EXPECTED_ITEMS = 7_111
EXPECTED_INTERACTIONS = 1_134_420
EXPECTED_SPLITS = {"train": 1_086_518, "validation": 23_951, "test": 23_951}
SPLIT_ORDER = ("train", "validation", "test")

BINARY_TARGETS = (
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)
WATCH_TIME_FIELDS = (
    "play_time_ms",
    "duration_ms",
    "play_ratio",
    "profile_stay_time",
    "comment_stay_time",
)
RAW_BEHAVIOR_FIELDS = (
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
)
SOURCE_AUDIT_FIELDS = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
)
DATASET_COLUMNS = (
    "user_id",
    "item_id",
    "timestamp",
    "source_row_id",
    "split",
    "date",
    "hourmin",
    "tab",
    "is_rand",
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "play_time_ms",
    "duration_ms",
    "play_ratio",
    "profile_stay_time",
    "comment_stay_time",
)

DERIVED_TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "strong_positive": (
        "is_like OR is_follow OR is_comment OR is_forward",
        ("is_like", "is_follow", "is_comment", "is_forward"),
    ),
    "explicit_positive": (
        "is_like OR is_follow OR is_comment OR is_forward OR is_profile_enter",
        ("is_like", "is_follow", "is_comment", "is_forward", "is_profile_enter"),
    ),
    "deep_engagement": (
        "long_view OR is_like OR is_follow OR is_comment OR is_forward OR is_profile_enter",
        (
            "long_view",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_profile_enter",
        ),
    ),
}

FIELD_SEMANTICS: dict[str, str] = {
    "user_id": "ID пользователя. В README KuaiRand-1K описание ошибочно называет его video ID; по смыслу и usage это user ID.",
    "video_id": "ID видео; в Protocol B переименован в item_id.",
    "date": "Дата interaction в формате YYYYMMDD.",
    "hourmin": "Время interaction в компактном формате HHMM/HHSS из датасета.",
    "time_ms": "Timestamp interaction в миллисекундах; в Protocol B поле timestamp.",
    "is_click": "Бинарный post-exposure feedback. В двухколоночном UI означает click; в одноколоночном UI означает valid_play: 1, если play_duration >= video_duration при video_duration <= 7s, иначе play_duration > 7s.",
    "is_like": "Бинарный post-exposure feedback: пользователь нажал like.",
    "is_follow": "Бинарный post-exposure feedback: пользователь подписался на автора.",
    "is_comment": "Бинарный post-exposure feedback: пользователь оставил комментарий.",
    "is_forward": "Бинарный post-exposure feedback: пользователь переслал/поделился видео.",
    "is_hate": "Бинарный post-exposure negative feedback: пользователь отметил видео как нежелательное.",
    "long_view": "Бинарный derived post-exposure feedback: 1, если play_duration >= video_duration при video_duration <= 18s, иначе play_duration >= 18s.",
    "play_time_ms": "Время просмотра пользователем в миллисекундах.",
    "duration_ms": "Длительность видео в миллисекундах.",
    "play_ratio": "Derived audit field: play_time_ms / duration_ms только при duration_ms > 0.",
    "profile_stay_time": "Время пребывания пользователя в профиле автора.",
    "comment_stay_time": "Время пребывания пользователя в секции комментариев.",
    "is_profile_enter": "Бинарный post-exposure feedback: пользователь вошел в профиль автора.",
    "is_rand": "Флаг random intervention; в Protocol B source должен быть 0, так как используется standard log.",
    "tab": "Сценарий/поверхность interaction в приложении, диапазон 0..14.",
}


@dataclass(frozen=True)
class AuditPaths:
    protocol_dir: Path
    protocol_manifest: Path
    source_log: Path
    output_dir: Path
    experiment_dir: Path
    repo_output_dir: Path
    manifest_path: Path
    audit_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST)
    parser.add_argument("--source-log", type=Path, default=DEFAULT_SOURCE_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=PROJECT_ROOT / "experiments" / "multitask_tim4rec")
    parser.add_argument("--repo-output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "data")
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--git-branch", default=None)
    return parser.parse_args()


def require_polars() -> Any:
    try:
        import polars as pl  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Для multitask audit нужен Polars. На cHARISMa используйте "
            "/home/daryumin/iberdov/diplom/.conda/bin/python."
        ) from exc
    return pl


def collect_lazy(lazy_frame: Any) -> Any:
    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:
        return lazy_frame.collect(streaming=True)


def git_value(args: list[str], default: str | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_rows(rows: Iterable[tuple[Any, ...]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(("\t".join("" if value is None else str(value) for value in row) + "\n").encode("utf-8"))
    return h.hexdigest()


def human_size(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}".replace(",", " ")


def fmt_rate(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.{digits}f}%"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_index_expr(split_col: str = "split") -> Any:
    pl = require_polars()
    return (
        pl.when(pl.col(split_col) == "train")
        .then(0)
        .when(pl.col(split_col) == "validation")
        .then(1)
        .when(pl.col(split_col) == "test")
        .then(2)
        .otherwise(9)
        .alias("_split_idx")
    )


def load_protocol_splits(protocol_dir: Path) -> Any:
    pl = require_polars()
    frames = []
    for split in SPLIT_ORDER:
        path = protocol_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Protocol B split не найден: {path}")
        frame = pl.read_parquet(path)
        required = {"user_id", "item_id", "timestamp", "source_row_id"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"В {path} не хватает колонок: {sorted(missing)}")
        frames.append(
            frame.select(["user_id", "item_id", "timestamp", "source_row_id"]).with_columns(
                pl.lit(split).alias("split")
            )
        )
    return pl.concat(frames, how="vertical").with_columns(pl.col("source_row_id").cast(pl.Int64))


def load_source_rows(source_log: Path, columns: tuple[str, ...]) -> tuple[Any, list[dict[str, Any]], list[str]]:
    pl = require_polars()
    if not source_log.exists():
        raise FileNotFoundError(f"Raw source log не найден: {source_log}")
    scan = pl.scan_csv(
        str(source_log),
        infer_schema_length=10_000,
        low_memory=True,
        try_parse_dates=False,
    )
    schema = scan.collect_schema()
    source_columns = list(schema.names())
    selected = [column for column in columns if column in source_columns]
    exprs = [pl.col("source_row_id").cast(pl.Int64)]
    for column in selected:
        if column == "user_id":
            exprs.append(pl.col(column).alias("raw_user_id"))
        elif column == "video_id":
            exprs.append(pl.col(column).alias("raw_item_id"))
        elif column == "time_ms":
            exprs.append(pl.col(column).alias("raw_timestamp"))
        else:
            exprs.append(pl.col(column))
    raw = collect_lazy(scan.with_row_index("source_row_id").select(exprs))
    schema_rows = []
    for name, dtype in zip(source_columns, schema.dtypes()):
        schema_rows.append({"field": name, "source_dtype": str(dtype)})
    return raw, schema_rows, source_columns


def add_play_ratio(frame: Any) -> Any:
    pl = require_polars()
    if "play_time_ms" not in frame.columns or "duration_ms" not in frame.columns:
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("play_ratio"))
    play = pl.col("play_time_ms").cast(pl.Float64, strict=False)
    duration = pl.col("duration_ms").cast(pl.Float64, strict=False)
    return frame.with_columns(
        pl.when(duration > 0).then(play / duration).otherwise(None).alias("play_ratio")
    )


def add_derived_targets(frame: Any) -> Any:
    pl = require_polars()
    exprs = []
    for name, (_, columns) in DERIVED_TARGETS.items():
        exprs.append(
            pl.max_horizontal([pl.col(column).cast(pl.Int8, strict=False).fill_null(0) for column in columns])
            .cast(pl.Int8)
            .alias(name)
        )
    return frame.with_columns(exprs)


def validate_join(protocol: Any, joined: Any) -> dict[str, Any]:
    pl = require_polars()
    rows_expected = protocol.height
    rows_after_join = joined.height
    matched = joined.filter(pl.col("raw_user_id").is_not_null())
    rows_matched = matched.height
    protocol_unique_source_rows = protocol.select(pl.col("source_row_id").n_unique()).item()
    source_row_duplicate_rows = rows_expected - protocol_unique_source_rows
    source_row_duplicate_keys = (
        protocol.group_by("source_row_id").len().filter(pl.col("len") > 1).height
        if source_row_duplicate_rows
        else 0
    )
    user_mismatch = matched.filter(pl.col("user_id") != pl.col("raw_user_id")).height
    item_mismatch = matched.filter(pl.col("item_id") != pl.col("raw_item_id")).height
    timestamp_mismatch = matched.filter(pl.col("timestamp") != pl.col("raw_timestamp")).height
    diagnostics = {
        "rows_expected": rows_expected,
        "rows_after_join": rows_after_join,
        "rows_matched": rows_matched,
        "rows_unmatched": rows_expected - rows_matched,
        "rows_multiple_matched": max(0, rows_after_join - rows_expected),
        "protocol_source_row_duplicate_keys": source_row_duplicate_keys,
        "protocol_source_row_duplicate_extra_rows": source_row_duplicate_rows,
        "user_id_mismatches": user_mismatch,
        "item_id_mismatches": item_mismatch,
        "timestamp_mismatches": timestamp_mismatch,
        "join_is_exact": (
            rows_after_join == rows_expected
            and rows_matched == rows_expected
            and source_row_duplicate_rows == 0
            and user_mismatch == 0
            and item_mismatch == 0
            and timestamp_mismatch == 0
        ),
    }
    if not diagnostics["join_is_exact"]:
        raise RuntimeError(f"Protocol B source join failed: {diagnostics}")
    return diagnostics


def unique_values_string(series: Any, limit: int = 20) -> str:
    values = series.drop_nulls().unique().sort().to_list()
    if len(values) > limit:
        shown = values[:limit]
        return json.dumps(shown, ensure_ascii=False) + f" ... (+{len(values) - limit})"
    return json.dumps(values, ensure_ascii=False)


def classify_field(field: str, frame: Any) -> str:
    if field in BINARY_TARGETS or field == "is_rand":
        return "binary"
    if field in WATCH_TIME_FIELDS:
        return "continuous"
    if field in {"date", "hourmin", "tab"}:
        return "categorical"
    return "identifier"


def possible_target(field: str) -> str:
    if field in BINARY_TARGETS:
        return "yes"
    if field in {"play_time_ms", "play_ratio", "profile_stay_time", "comment_stay_time"}:
        return "possible_regression_or_auxiliary"
    if field == "duration_ms":
        return "source_for_derived_watch_targets"
    return "no"


def leakage_risk(field: str) -> str:
    if field in BINARY_TARGETS or field in WATCH_TIME_FIELDS:
        return "post-exposure label/current interaction: target-only, forbidden as same-row input"
    if field == "is_rand":
        return "constant standard-policy flag in Protocol B; not useful as target"
    if field in {"date", "hourmin", "tab"}:
        return "context field; future use needs time-aware policy review"
    return "identity/linkage field; not a behavior target"


def build_source_field_schema(frame: Any, source_schema_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dtype_by_field = {row["field"]: row["source_dtype"] for row in source_schema_rows}
    rows = []
    for field in SOURCE_AUDIT_FIELDS + ("play_ratio",):
        source_field = {"video_id": "raw_item_id", "time_ms": "raw_timestamp"}.get(field, field)
        materialized = source_field in frame.columns
        available_in_source = field in dtype_by_field
        if materialized:
            series = frame[source_field]
            missing = int(series.null_count())
            unique_count = int(series.n_unique())
            unique_values = unique_values_string(series)
            dtype = str(series.dtype)
        elif field == "play_ratio" and field in frame.columns:
            series = frame[field]
            missing = int(series.null_count())
            unique_count = int(series.n_unique())
            unique_values = unique_values_string(series)
            dtype = str(series.dtype)
        else:
            missing = None
            unique_count = None
            unique_values = ""
            dtype = dtype_by_field.get(field, "")
        rows.append(
            {
                "field": field,
                "dtype": dtype,
                "missing_count": missing,
                "unique_count": unique_count,
                "unique_values": unique_values,
                "field_kind": classify_field(field, frame),
                "available_in_protocol_b_source": bool(available_in_source),
                "possible_target": possible_target(field),
                "possible_leakage_risk": leakage_risk(field),
                "semantics": FIELD_SEMANTICS.get(field, ""),
            }
        )
    return rows


def numeric_summary(series: Any) -> dict[str, Any]:
    import numpy as np

    values = series.cast(float).drop_nulls().to_numpy()
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.5)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def target_stats(frame: Any, binary_fields: tuple[str, ...], continuous_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    analysis_frame = add_derived_targets(frame)
    binary_all = tuple(binary_fields) + tuple(DERIVED_TARGETS)
    scopes = {"full_filtered": analysis_frame}
    for split in SPLIT_ORDER:
        scopes[split] = analysis_frame.filter(analysis_frame["split"] == split)

    for scope, scope_frame in scopes.items():
        for field in binary_all:
            series = scope_frame[field]
            total = scope_frame.height
            missing = int(series.null_count())
            non_null = total - missing
            positives = int(series.cast(int).sum() or 0)
            negatives = non_null - positives
            rows.append(
                {
                    "scope": scope,
                    "field": field,
                    "field_origin": "derived" if field in DERIVED_TARGETS else "raw",
                    "target_kind": "binary",
                    "rows": total,
                    "missing": missing,
                    "non_null": non_null,
                    "unique_count": int(series.n_unique()),
                    "unique_values": unique_values_string(series),
                    "positives": positives,
                    "negatives": negatives,
                    "positive_rate": positives / total if total else None,
                    "positive_rate_non_null": positives / non_null if non_null else None,
                    "negative_positive_ratio": negatives / positives if positives else None,
                    "count": non_null,
                    "mean": None,
                    "median": None,
                    "std": None,
                    "p90": None,
                    "p95": None,
                    "p99": None,
                    "min": None,
                    "max": None,
                }
            )
        for field in continuous_fields:
            series = scope_frame[field]
            summary = numeric_summary(series)
            rows.append(
                {
                    "scope": scope,
                    "field": field,
                    "field_origin": "derived" if field == "play_ratio" else "raw",
                    "target_kind": "continuous",
                    "rows": scope_frame.height,
                    "missing": int(series.null_count()),
                    "non_null": summary["count"],
                    "unique_count": int(series.n_unique()),
                    "unique_values": "",
                    "positives": None,
                    "negatives": None,
                    "positive_rate": None,
                    "positive_rate_non_null": None,
                    "negative_positive_ratio": None,
                    **summary,
                }
            )
    return rows


def target_cooccurrence(frame: Any, binary_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    pl = require_polars()
    analysis_frame = add_derived_targets(frame)
    fields = tuple(binary_fields) + tuple(DERIVED_TARGETS)
    rows: list[dict[str, Any]] = []
    scopes = {"full_filtered": analysis_frame}
    for split in SPLIT_ORDER:
        scopes[split] = analysis_frame.filter(analysis_frame["split"] == split)
    for scope, scope_frame in scopes.items():
        positive_counts = {
            field: int(scope_frame.select(pl.col(field).cast(pl.Int64).sum()).item() or 0)
            for field in fields
        }
        for left in fields:
            left_pos = positive_counts[left]
            for right in fields:
                right_pos = positive_counts[right]
                both = int(
                    scope_frame.select(((pl.col(left) == 1) & (pl.col(right) == 1)).sum()).item()
                    or 0
                )
                union = left_pos + right_pos - both
                rows.append(
                    {
                        "scope": scope,
                        "target_i": left,
                        "target_j": right,
                        "positive_i": left_pos,
                        "positive_j": right_pos,
                        "count_both_positive": both,
                        "p_j_given_i": both / left_pos if left_pos else None,
                        "p_i_given_j": both / right_pos if right_pos else None,
                        "jaccard": both / union if union else None,
                    }
                )
    return rows


def target_correlations(frame: Any, binary_fields: tuple[str, ...], continuous_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    import numpy as np
    import pandas as pd

    analysis_frame = add_derived_targets(frame)
    binary_all = tuple(binary_fields) + tuple(DERIVED_TARGETS)
    fields = list(binary_all) + list(continuous_fields)
    rows: list[dict[str, Any]] = []
    scopes = {"full_filtered": analysis_frame}
    for split in SPLIT_ORDER:
        scopes[split] = analysis_frame.filter(analysis_frame["split"] == split)

    for scope, scope_frame in scopes.items():
        pdf = scope_frame.select(fields).to_pandas()
        pearson = pdf.corr(method="pearson", numeric_only=True)
        spearman = pdf.rank(method="average").corr(method="pearson", numeric_only=True)
        for idx, left in enumerate(binary_all):
            for right in binary_all[idx + 1 :]:
                value = pearson.loc[left, right]
                rows.append(
                    {
                        "scope": scope,
                        "field_a": left,
                        "field_b": right,
                        "correlation_kind": "phi_binary_pearson",
                        "pearson_or_phi": None if np.isnan(value) else float(value),
                        "spearman": None,
                        "n": int(pdf[[left, right]].dropna().shape[0]),
                    }
                )
        for continuous in continuous_fields:
            for binary in binary_all:
                pearson_value = pearson.loc[continuous, binary]
                spearman_value = spearman.loc[continuous, binary]
                rows.append(
                    {
                        "scope": scope,
                        "field_a": continuous,
                        "field_b": binary,
                        "correlation_kind": "continuous_vs_binary",
                        "pearson_or_phi": None if np.isnan(pearson_value) else float(pearson_value),
                        "spearman": None if np.isnan(spearman_value) else float(spearman_value),
                        "n": int(pdf[[continuous, binary]].dropna().shape[0]),
                    }
                )
        for idx, left in enumerate(continuous_fields):
            for right in continuous_fields[idx + 1 :]:
                pearson_value = pearson.loc[left, right]
                spearman_value = spearman.loc[left, right]
                rows.append(
                    {
                        "scope": scope,
                        "field_a": left,
                        "field_b": right,
                        "correlation_kind": "continuous_vs_continuous",
                        "pearson_or_phi": None if np.isnan(pearson_value) else float(pearson_value),
                        "spearman": None if np.isnan(spearman_value) else float(spearman_value),
                        "n": int(pdf[[left, right]].dropna().shape[0]),
                    }
                )
    return rows


def coverage_rows(frame: Any, binary_fields: tuple[str, ...], entity: str) -> list[dict[str, Any]]:
    import numpy as np

    analysis_frame = add_derived_targets(frame).filter(frame["split"] == "train")
    fields = tuple(binary_fields) + tuple(DERIVED_TARGETS)
    total_entities = int(analysis_frame[entity].n_unique())
    rows: list[dict[str, Any]] = []
    for field in fields:
        positive = analysis_frame.filter(analysis_frame[field] == 1)
        if positive.height == 0:
            counts = np.array([], dtype=float)
            positive_entities = 0
        else:
            grouped = positive.group_by(entity).len().select("len")
            counts = grouped["len"].to_numpy()
            positive_entities = int(counts.size)
        label = "users" if entity == "user_id" else "items"
        rows.append(
            {
                "target": field,
                "positive_rows_train": positive.height,
                f"positive_{label}_train": positive_entities,
                f"share_{label}_with_positive_train": positive_entities / total_entities
                if total_entities
                else None,
                f"median_positives_per_{entity}_among_positive": float(np.quantile(counts, 0.5))
                if counts.size
                else None,
                f"p95_positives_per_{entity}_among_positive": float(np.quantile(counts, 0.95))
                if counts.size
                else None,
                f"total_{label}_train": total_entities,
            }
        )
    return rows


def temporal_drift_rows(stats_rows: list[dict[str, Any]], binary_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    fields = tuple(binary_fields) + tuple(DERIVED_TARGETS)
    by_field_scope = {(row["field"], row["scope"]): row for row in stats_rows}
    rows = []
    for field in fields:
        train = safe_float(by_field_scope[(field, "train")]["positive_rate"])
        validation = safe_float(by_field_scope[(field, "validation")]["positive_rate"])
        test = safe_float(by_field_scope[(field, "test")]["positive_rate"])
        values = [value for value in (train, validation, test) if value is not None]
        rows.append(
            {
                "target": field,
                "train_positive_rate": train,
                "validation_positive_rate": validation,
                "test_positive_rate": test,
                "validation_minus_train": validation - train if train is not None and validation is not None else None,
                "test_minus_train": test - train if train is not None and test is not None else None,
                "max_minus_min_across_splits": max(values) - min(values) if values else None,
            }
        )
    return rows


def temporal_by_date_rows(frame: Any, binary_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    pl = require_polars()
    analysis_frame = add_derived_targets(frame)
    fields = tuple(binary_fields) + tuple(DERIVED_TARGETS)
    rows: list[dict[str, Any]] = []
    for field in fields:
        daily = (
            analysis_frame.group_by(["split", "date"])
            .agg([pl.len().alias("rows"), pl.col(field).cast(pl.Int64).sum().alias("positives")])
            .with_columns((pl.col("positives") / pl.col("rows")).alias("positive_rate"))
            .sort(["split", "date"])
        )
        for row in daily.iter_rows(named=True):
            row["target"] = field
            rows.append(row)
    return rows


def watch_time_diagnostics(frame: Any) -> list[dict[str, Any]]:
    import numpy as np

    rows: list[dict[str, Any]] = []
    scopes = {"full_filtered": frame}
    for split in SPLIT_ORDER:
        scopes[split] = frame.filter(frame["split"] == split)
    for scope, scope_frame in scopes.items():
        total = scope_frame.height
        duration = scope_frame["duration_ms"].cast(float)
        play = scope_frame["play_time_ms"].cast(float)
        ratio = scope_frame["play_ratio"].cast(float).drop_nulls().to_numpy()
        rows.append(
            {
                "scope": scope,
                "rows": total,
                "play_time_missing": int(scope_frame["play_time_ms"].null_count()),
                "duration_missing": int(scope_frame["duration_ms"].null_count()),
                "duration_zero": int((duration == 0).sum()),
                "duration_negative": int((duration < 0).sum()),
                "duration_non_positive": int((duration <= 0).sum()),
                "duration_non_positive_rate": int((duration <= 0).sum()) / total if total else None,
                "play_time_negative": int((play < 0).sum()),
                "play_time_zero": int((play == 0).sum()),
                "play_ratio_count": int(ratio.size),
                "play_ratio_missing_or_duration_non_positive": total - int(ratio.size),
                "play_ratio_mean": float(np.mean(ratio)) if ratio.size else None,
                "play_ratio_median": float(np.quantile(ratio, 0.5)) if ratio.size else None,
                "play_ratio_p90": float(np.quantile(ratio, 0.9)) if ratio.size else None,
                "play_ratio_p95": float(np.quantile(ratio, 0.95)) if ratio.size else None,
                "play_ratio_p99": float(np.quantile(ratio, 0.99)) if ratio.size else None,
                "play_ratio_max": float(np.max(ratio)) if ratio.size else None,
                "play_ratio_gt_1": int((ratio > 1).sum()) if ratio.size else 0,
                "play_ratio_gt_1_rate": float((ratio > 1).sum() / ratio.size) if ratio.size else None,
                "play_ratio_gt_2": int((ratio > 2).sum()) if ratio.size else 0,
                "play_ratio_gt_2_rate": float((ratio > 2).sum() / ratio.size) if ratio.size else None,
                "play_ratio_gt_5": int((ratio > 5).sum()) if ratio.size else 0,
                "play_ratio_gt_5_rate": float((ratio > 5).sum() / ratio.size) if ratio.size else None,
                "play_ratio_gt_10": int((ratio > 10).sum()) if ratio.size else 0,
                "play_ratio_gt_10_rate": float((ratio > 10).sum() / ratio.size) if ratio.size else None,
            }
        )
    return rows


def dataset_fingerprint(frame: Any, output_dir: Path) -> dict[str, Any]:
    pl = require_polars()
    split_counts = {split: frame.filter(frame["split"] == split).height for split in SPLIT_ORDER}
    users = int(frame["user_id"].n_unique())
    items = int(frame["item_id"].n_unique())
    sorted_identity = frame.with_columns(split_index_expr()).sort(
        ["_split_idx", "user_id", "timestamp", "source_row_id", "item_id"]
    )
    identity_hash = sha256_rows(
        (
            row["user_id"],
            row["item_id"],
            row["timestamp"],
            row["split"],
        )
        for row in sorted_identity.select(["user_id", "item_id", "timestamp", "split"]).iter_rows(named=True)
    )
    source_identity_hash = sha256_rows(
        (
            row["source_row_id"],
            row["user_id"],
            row["item_id"],
            row["timestamp"],
            row["split"],
        )
        for row in sorted_identity.select(
            ["source_row_id", "user_id", "item_id", "timestamp", "split"]
        ).iter_rows(named=True)
    )
    return {
        "users": users,
        "items": items,
        "interactions": frame.height,
        "split_counts": split_counts,
        "matches_expected": {
            "users": users == EXPECTED_USERS,
            "items": items == EXPECTED_ITEMS,
            "interactions": frame.height == EXPECTED_INTERACTIONS,
            "splits": split_counts == EXPECTED_SPLITS,
            "all": users == EXPECTED_USERS
            and items == EXPECTED_ITEMS
            and frame.height == EXPECTED_INTERACTIONS
            and split_counts == EXPECTED_SPLITS,
        },
        "identity_hash_user_item_timestamp_split": identity_hash,
        "identity_hash_source_row_user_item_timestamp_split": source_identity_hash,
        "output_dir": str(output_dir),
    }


def file_inventory(root: Path) -> list[dict[str, Any]]:
    files = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path),
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "size": human_size(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return files


def write_dataset(frame: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    full = frame.select(DATASET_COLUMNS).with_columns(split_index_expr()).sort(
        ["_split_idx", "user_id", "timestamp", "source_row_id", "item_id"]
    ).drop("_split_idx")
    full.write_parquet(output_dir / "full_filtered.parquet", compression="zstd")
    for split in SPLIT_ORDER:
        part = full.filter(full["split"] == split).sort(["user_id", "timestamp", "source_row_id", "item_id"])
        part.write_parquet(output_dir / f"{split}.parquet", compression="zstd")


def row_for(stats_rows: list[dict[str, Any]], field: str, scope: str) -> dict[str, Any]:
    for row in stats_rows:
        if row["field"] == field and row["scope"] == scope:
            return row
    raise KeyError((field, scope))


def coverage_for(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    for row in rows:
        if row["target"] == field:
            return row
    raise KeyError(field)


def cooccur_value(rows: list[dict[str, Any]], left: str, right: str, scope: str = "train") -> dict[str, Any]:
    for row in rows:
        if row["scope"] == scope and row["target_i"] == left and row["target_j"] == right:
            return row
    raise KeyError((scope, left, right))


def corr_value(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    scope: str = "train",
    kind: str | None = None,
) -> dict[str, Any] | None:
    for row in rows:
        same_pair = {row["field_a"], row["field_b"]} == {left, right}
        kind_ok = kind is None or row["correlation_kind"] == kind
        if row["scope"] == scope and same_pair and kind_ok:
            return row
    return None


def build_audit_markdown(
    manifest: dict[str, Any],
    source_schema: list[dict[str, Any]],
    stats: list[dict[str, Any]],
    cooccurrence: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    user_coverage: list[dict[str, Any]],
    item_coverage: list[dict[str, Any]],
    drift: list[dict[str, Any]],
    watch: list[dict[str, Any]],
) -> str:
    recommended = ["is_click", "long_view", "is_like", "is_profile_enter"]
    richer_with_negative = ["is_click", "long_view", "is_like", "is_profile_enter", "is_hate"]
    balanced_derived = ["is_click", "long_view", "explicit_positive", "is_hate"]
    minimal = ["is_click", "long_view", "is_like"]
    def field_table() -> list[str]:
        lines = [
            "| field | dtype | missing | unique values | available in source | kind | possible target | leakage risk |",
            "| --- | --- | ---: | --- | --- | --- | --- | --- |",
        ]
        for row in source_schema:
            if row["field"] not in SOURCE_AUDIT_FIELDS + ("play_ratio",):
                continue
            lines.append(
                "| `{field}` | `{dtype}` | {missing} | `{unique}` | {available} | {kind} | {target} | {risk} |".format(
                    field=row["field"],
                    dtype=row["dtype"],
                    missing=fmt_int(row["missing_count"]),
                    unique=str(row["unique_values"]).replace("|", "/"),
                    available=row["available_in_protocol_b_source"],
                    kind=row["field_kind"],
                    target=row["possible_target"],
                    risk=row["possible_leakage_risk"].replace("|", "/"),
                )
            )
        return lines

    def rate_table(fields: list[str]) -> list[str]:
        lines = [
            "| target | train positive rate | validation positive rate | test positive rate | train positives | train positive users | train positive items | missing train |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for field in fields:
            train = row_for(stats, field, "train")
            val = row_for(stats, field, "validation")
            test = row_for(stats, field, "test")
            uc = coverage_for(user_coverage, field)
            ic = coverage_for(item_coverage, field)
            lines.append(
                f"| `{field}` | {fmt_rate(train['positive_rate'])} | {fmt_rate(val['positive_rate'])} | "
                f"{fmt_rate(test['positive_rate'])} | {fmt_int(train['positives'])} | "
                f"{fmt_rate(uc['share_users_with_positive_train'])} | "
                f"{fmt_rate(ic['share_items_with_positive_train'])} | {fmt_int(train['missing'])} |"
            )
        return lines

    def user_coverage_table(fields: list[str]) -> list[str]:
        lines = [
            "| target | positive rows train | positive users train | share users with positive | median positives/user | p95 positives/user |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for field in fields:
            row = coverage_for(user_coverage, field)
            lines.append(
                f"| `{field}` | {fmt_int(row['positive_rows_train'])} | "
                f"{fmt_int(row['positive_users_train'])} | "
                f"{fmt_rate(row['share_users_with_positive_train'])} | "
                f"{safe_float(row['median_positives_per_user_id_among_positive']):.1f} | "
                f"{safe_float(row['p95_positives_per_user_id_among_positive']):.1f} |"
            )
        return lines

    def item_coverage_table(fields: list[str]) -> list[str]:
        lines = [
            "| target | positive rows train | positive items train | share items with positive | median positives/item | p95 positives/item |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for field in fields:
            row = coverage_for(item_coverage, field)
            lines.append(
                f"| `{field}` | {fmt_int(row['positive_rows_train'])} | "
                f"{fmt_int(row['positive_items_train'])} | "
                f"{fmt_rate(row['share_items_with_positive_train'])} | "
                f"{safe_float(row['median_positives_per_item_id_among_positive']):.1f} | "
                f"{safe_float(row['p95_positives_per_item_id_among_positive']):.1f} |"
            )
        return lines

    def drift_table(fields: list[str]) -> list[str]:
        by_target = {row["target"]: row for row in drift}
        lines = [
            "| target | train | validation | test | test-train | max-min |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for field in fields:
            row = by_target[field]
            lines.append(
                f"| `{field}` | {fmt_rate(row['train_positive_rate'])} | "
                f"{fmt_rate(row['validation_positive_rate'])} | "
                f"{fmt_rate(row['test_positive_rate'])} | "
                f"{fmt_rate(row['test_minus_train'])} | "
                f"{fmt_rate(row['max_minus_min_across_splits'])} |"
            )
        return lines

    def option_lines(name: str, fields: list[str]) -> list[str]:
        lines = [f"### {name}", ""]
        for field in fields:
            train = row_for(stats, field, "train")
            uc = coverage_for(user_coverage, field)
            ic = coverage_for(item_coverage, field)
            origin = train["field_origin"]
            definition = f"; formula: `{DERIVED_TARGETS[field][0]}`" if field in DERIVED_TARGETS else ""
            lines.append(
                f"- `{field}` ({origin}{definition}): train rate {fmt_rate(train['positive_rate'])}, "
                f"users+ {fmt_rate(uc['share_users_with_positive_train'])}, "
                f"items+ {fmt_rate(ic['share_items_with_positive_train'])}."
            )
        lines.append("")
        return lines

    def top_cooccurrence_lines() -> list[str]:
        pairs = [
            ("long_view", "is_click"),
            ("is_like", "is_click"),
            ("is_profile_enter", "is_click"),
            ("is_hate", "is_click"),
            ("is_follow", "is_like"),
            ("is_comment", "is_click"),
            ("is_forward", "is_click"),
            ("explicit_positive", "is_like"),
        ]
        lines = ["| condition | probability | count both | jaccard |", "| --- | ---: | ---: | ---: |"]
        for left, right in pairs:
            row = cooccur_value(cooccurrence, right, left, "train")
            lines.append(
                f"| P(`{left}`=1 / `{right}`=1) | {fmt_rate(row['p_j_given_i'])} | "
                f"{fmt_int(row['count_both_positive'])} | {safe_float(row['jaccard']):.4f} |"
            )
        return lines

    train_watch = next(row for row in watch if row["scope"] == "train")
    full_watch = next(row for row in watch if row["scope"] == "full_filtered")
    click_long = cooccur_value(cooccurrence, "is_click", "long_view", "train")
    like_click = cooccur_value(cooccurrence, "is_click", "is_like", "train")
    hate_click = cooccur_value(cooccurrence, "is_click", "is_hate", "train")
    ratio_long_corr = corr_value(correlations, "play_ratio", "long_view", "train", "continuous_vs_binary")
    ratio_click_corr = corr_value(correlations, "play_ratio", "is_click", "train", "continuous_vs_binary")

    lines = [
        "# Multitask KuaiRand Protocol B audit",
        "",
        "## Цель",
        "",
        "Подготовить первый data/audit слой для будущей TiM4Rec-based multitask архитектуры без обучения новой модели. Проверка выполняется именно на 1 134 420 interactions Protocol B, а не на full KuaiRand-27K EDA.",
        "",
        "## Источник данных",
        "",
        f"- Protocol B parquet: `{manifest['source_protocol_b']['remote_path']}`.",
        f"- Raw source log: `{manifest['source_raw_files'][0]['path']}`.",
        "- Используется ранний KuaiRand-Pure standard log; random interactions в Protocol B не входят.",
        "- Семантика полей взята из README KuaiRand, локальная копия: `data/KuaiRand-1K/README.md`.",
        "",
        "## Связь с существующим Protocol B",
        "",
        "- Split не пересоздавался: train/validation/test прочитаны из существующих Protocol B parquet.",
        "- Каждая строка Protocol B сохраняет `source_row_id`, нулевой номер строки в raw CSV.",
        "- Join выполнен по `source_row_id`; `(user_id, item_id, timestamp)` использовался только как проверка соответствия.",
        f"- Exact row identity hash `user_id,item_id,timestamp,split`: `{manifest['dataset_fingerprint']['identity_hash_user_item_timestamp_split']}`.",
        "",
        "## Доступные behavior labels",
        "",
        *field_table(),
        "",
        "## Семантика labels",
        "",
        "- `is_click` является post-exposure бинарным feedback. В single-column UI это derived `valid_play`, поэтому он не является чистым click во всех сценариях.",
        "- `long_view` является derived post-exposure target из watch-time и duration: threshold 18 секунд или полная длительность для коротких видео.",
        "- `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_profile_enter` являются явными positive engagement actions после показа.",
        "- `is_hate` является negative feedback и не смешивается с positive engagement.",
        "- `play_ratio` в audit вычислен как `play_time_ms / duration_ms` только при `duration_ms > 0`; в raw CSV отдельной колонки `play_ratio` нет.",
        "",
        "## Статистика train/validation/test",
        "",
        *rate_table(list(BINARY_TARGETS) + list(DERIVED_TARGETS)),
        "",
        "## Class imbalance",
        "",
        "- `is_click` и `long_view` достаточно частые и стабильные по split.",
        "- `is_like` и `is_profile_enter` редкие, но имеют заметную user/item coverage и отражают разные действия.",
        "- `is_follow`, `is_comment`, `is_forward`, `is_hate` сильно разрежены; отдельные heads для них в первом эксперименте рискованны.",
        "- `explicit_positive` и `strong_positive` могут снизить sparse-проблему, но это derived targets, поэтому их надо явно фиксировать в protocol/config.",
        "",
        "## User coverage",
        "",
        *user_coverage_table(list(BINARY_TARGETS) + list(DERIVED_TARGETS)),
        "",
        "## Item coverage",
        "",
        "Покрытие item positives считается только по TRAIN, так как будущая модель должна учиться без validation/test labels.",
        "",
        *item_coverage_table(list(BINARY_TARGETS) + list(DERIVED_TARGETS)),
        "",
        "## Temporal distribution",
        "",
        *drift_table(list(BINARY_TARGETS) + list(DERIVED_TARGETS)),
        "",
        "- `is_click` и `long_view` выше в validation/test примерно на 2.2-2.4 п.п. относительно train.",
        "- `is_profile_enter` немного снижается к test; sparse labels дают более шумные split-level rates.",
        "",
        "## Co-occurrence",
        "",
        *top_cooccurrence_lines(),
        "",
        f"- В train P(`long_view`=1 / `is_click`=1) = {fmt_rate(click_long['p_j_given_i'])}. Это показывает сильную, но не полную связь consumption и click/valid_play.",
        f"- В train P(`is_like`=1 / `is_click`=1) = {fmt_rate(like_click['p_j_given_i'])}; like несет более редкий explicit-positive сигнал.",
        f"- В train P(`is_hate`=1 / `is_click`=1) = {fmt_rate(hate_click['p_j_given_i'])}; negative feedback не надо объединять с positive engagement.",
        "",
        "## Correlations",
        "",
        f"- Spearman(`play_ratio`, `long_view`) в train = {safe_float(ratio_long_corr['spearman']):.4f}; это ожидаемо, потому что `long_view` derived из watch-time/duration.",
        f"- Spearman(`play_ratio`, `is_click`) в train = {safe_float(ratio_click_corr['spearman']):.4f}; click/valid_play также связан с watch-time, но не тождественен long_view.",
        "- Correlation не интерпретируется как причинность. Полная матрица сохранена в `target_correlations.csv`.",
        "",
        "## Rare behaviors",
        "",
        "- `is_follow`, `is_comment`, `is_forward` лучше не брать отдельными heads в `multitask_tim4rec_001`: они слишком sparse для первого устойчивого запуска.",
        "- Их можно включить через прозрачный aggregate `strong_positive` или `explicit_positive`, либо оставить для второго этапа после базовой multitask проверки.",
        "- `is_hate` тоже sparse, но семантически отдельный negative-preference сигнал; его стоит держать как option, а не смешивать с positive labels.",
        "",
        "## Negative feedback",
        "",
        f"- `is_hate` в train: {fmt_rate(row_for(stats, 'is_hate', 'train')['positive_rate'])}, positives {fmt_int(row_for(stats, 'is_hate', 'train')['positives'])}, positive users {fmt_rate(coverage_for(user_coverage, 'is_hate')['share_users_with_positive_train'])}.",
        "- Для первого stable multitask run это слишком sparse как обязательная head, но поле подготовлено и валидно как будущая negative-preference task.",
        "- `is_hate` нельзя объединять с `strong_positive` или `explicit_positive`: это отдельная семантика.",
        "",
        "## Watch-time variables",
        "",
        f"- В full filtered `duration_ms <= 0`: {fmt_int(full_watch['duration_non_positive'])} строк ({fmt_rate(full_watch['duration_non_positive_rate'])}).",
        f"- В train `duration_ms <= 0`: {fmt_int(train_watch['duration_non_positive'])} строк ({fmt_rate(train_watch['duration_non_positive_rate'])}).",
        f"- В train `play_ratio > 1`: {fmt_int(train_watch['play_ratio_gt_1'])} строк среди valid ratio ({fmt_rate(train_watch['play_ratio_gt_1_rate'])}).",
        "- `play_ratio > 1` не помечается как ошибка: возможны пересмотры/повторы. Для regression head разумнее анализировать `log1p(play_time_ms)` или clipped `play_ratio`, но не в первом data-only шаге.",
        "",
        "## Возможные derived targets",
        "",
        "- `strong_positive = is_like OR is_follow OR is_comment OR is_forward`.",
        "- `explicit_positive = is_like OR is_follow OR is_comment OR is_forward OR is_profile_enter`.",
        "- `deep_engagement = long_view OR is_like OR is_follow OR is_comment OR is_forward OR is_profile_enter`.",
        "- Derived targets не материализованы как обязательные labels будущей модели; они посчитаны для выбора первого experiment.",
        "",
        "## Candidate task sets",
        "",
        *option_lines("OPTION A - minimal", minimal),
        *option_lines("OPTION B - balanced derived", balanced_derived),
        *option_lines("OPTION C - richer raw", richer_with_negative),
        "",
        "## Рекомендуемый первый multitask набор",
        "",
        "Для `multitask_tim4rec_001` рекомендуется начать с raw targets:",
        "",
        *[f"- `{field}`" for field in recommended],
        "",
        "Причина: это 4 задачи без спорной preprocessing-логики, с разными behavioral meanings и приемлемой плотностью. `is_hate` лучше оставить как заранее подготовленный negative option для следующего запуска или ablation, потому что он слишком sparse для первого stability check.",
        "",
        "## Предварительные behavior experts",
        "",
        "- Interest/exposure response: `is_click`.",
        "- Consumption/watch-time: `long_view`, позднее `log1p(play_time_ms)` или clipped `play_ratio`.",
        "- Positive engagement: `is_like`, `is_profile_enter`, позднее aggregate `explicit_positive`.",
        "- Social/amplification: `is_follow`, `is_comment`, `is_forward` как отдельная группа после sparse-aware настройки.",
        "- Negative preference: `is_hate` отдельно от positive heads.",
        "",
        "## Leakage policy",
        "",
        "- Labels текущего candidate interaction не используются как input features для предсказания этого же interaction.",
        "- Historical behavior предыдущих interactions можно использовать позже только после time-aware sequence construction.",
        "- Full-period item statistics и target-derived aggregates запрещены как inputs без перестроения по train-window.",
        "- `duration_ms` может быть item/context field, но любые derived watch targets из текущего interaction являются target-only.",
        "",
        "## Dataset preparation",
        "",
        f"- Multitask dataset path: `{manifest['remote_paths']['multitask_dataset_dir']}`.",
        "- Сохранены `train.parquet`, `validation.parquet`, `test.parquet`, `full_filtered.parquet`.",
        "- В строках сохранены `user_id`, `item_id`, `timestamp`, `source_row_id`, `split`, raw behavior labels и watch-time поля.",
        "",
        "## Join validation",
        "",
        f"- rows expected: {fmt_int(manifest['join_diagnostics']['rows_expected'])}.",
        f"- rows matched: {fmt_int(manifest['join_diagnostics']['rows_matched'])}.",
        f"- unmatched: {fmt_int(manifest['join_diagnostics']['rows_unmatched'])}.",
        f"- multiple matched: {fmt_int(manifest['join_diagnostics']['rows_multiple_matched'])}.",
        f"- source_row duplicate extra rows внутри Protocol B: {fmt_int(manifest['join_diagnostics']['protocol_source_row_duplicate_extra_rows'])}.",
        "- Hidden many-to-many joins отсутствуют, потому что join key - stable raw row index.",
        "",
        "## Ограничения",
        "",
        "- Аудит фиксирует labels для KuaiRand-Pure Protocol B, а не для full KuaiRand-27K.",
        "- `is_click` и `long_view` частично derived из watch behavior и UI-specific semantics, поэтому они не являются независимыми чистыми actions.",
        "- Continuous watch-time targets требуют отдельной нормализации/clipping политики перед обучением.",
        "",
        "## Следующий эксперимент",
        "",
        "Следующий шаг - реализовать минимальный TiM4Rec multitask head поверх того же sequence backbone для targets `is_click`, `long_view`, `is_like`, `is_profile_enter`. Не менять split и не открывать новые evaluation protocol variants.",
        "",
    ]
    return "\n".join(lines)


def build_config_text(manifest: dict[str, Any]) -> str:
    recommended = ["is_click", "long_view", "is_like", "is_profile_enter"]
    return "\n".join(
        [
            "experiment: multitask_tim4rec",
            "stage: target_audit_only",
            "protocol: B",
            "",
            "source:",
            f"  protocol_b_dir: {manifest['source_protocol_b']['remote_path']}",
            f"  protocol_b_manifest: {manifest['source_protocol_b']['manifest_path']}",
            f"  raw_log: {manifest['source_raw_files'][0]['path']}",
            "  join_key: source_row_id",
            "",
            "output:",
            f"  multitask_dataset_dir: {manifest['remote_paths']['multitask_dataset_dir']}",
            "  manifest: outputs/data/protocol_b_multitask_manifest.json",
            "  audit: experiments/multitask_tim4rec/AUDIT.md",
            "",
            "recommended_first_version_targets:",
            *[f"  - {target}" for target in recommended],
            "",
            "raw_binary_targets:",
            *[f"  - {target}" for target in BINARY_TARGETS],
            "",
            "watch_time_fields:",
            *[f"  - {field}" for field in WATCH_TIME_FIELDS],
            "",
            "derived_target_candidates:",
            *[f"  {name}: {definition}" for name, (definition, _) in DERIVED_TARGETS.items()],
            "",
            "leakage_policy:",
            "  current_interaction_labels_as_inputs: forbidden",
            "  historical_behavior_context_for_future_experiments: allowed_after_time_aware_construction",
            "  full_period_item_statistics_as_inputs: forbidden_without_train_window_rebuild",
            "",
            "training:",
            "  run_training: false",
            "  run_optuna: false",
            "  implement_architecture: false",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    paths = AuditPaths(
        protocol_dir=args.protocol_dir.expanduser().resolve(),
        protocol_manifest=args.protocol_manifest.expanduser().resolve(),
        source_log=args.source_log.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        experiment_dir=args.experiment_dir.expanduser().resolve(),
        repo_output_dir=args.repo_output_dir.expanduser().resolve(),
        manifest_path=(args.repo_output_dir / "protocol_b_multitask_manifest.json").expanduser().resolve(),
        audit_path=(args.experiment_dir / "AUDIT.md").expanduser().resolve(),
    )
    paths.experiment_dir.mkdir(parents=True, exist_ok=True)
    paths.repo_output_dir.mkdir(parents=True, exist_ok=True)
    (paths.experiment_dir / "runs").mkdir(parents=True, exist_ok=True)

    pl = require_polars()
    protocol_manifest = load_json(paths.protocol_manifest)
    protocol = load_protocol_splits(paths.protocol_dir)
    raw, source_schema_rows, source_columns = load_source_rows(paths.source_log, SOURCE_AUDIT_FIELDS)
    joined = protocol.join(raw, on="source_row_id", how="left")
    join_diagnostics = validate_join(protocol, joined)
    joined = add_play_ratio(joined)

    selected = joined.select(
        [
            "user_id",
            "item_id",
            "timestamp",
            "source_row_id",
            "split",
            "date",
            "hourmin",
            "tab",
            "is_rand",
            *BINARY_TARGETS,
            "play_time_ms",
            "duration_ms",
            "play_ratio",
            "profile_stay_time",
            "comment_stay_time",
        ]
    )
    write_dataset(selected, paths.output_dir)
    output_full = pl.read_parquet(paths.output_dir / "full_filtered.parquet")

    fingerprint = dataset_fingerprint(output_full, paths.output_dir)
    if not fingerprint["matches_expected"]["all"]:
        raise RuntimeError(f"Multitask dataset fingerprint mismatch: {fingerprint}")

    source_schema = build_source_field_schema(joined, source_schema_rows)
    stats = target_stats(selected, BINARY_TARGETS, WATCH_TIME_FIELDS)
    cooccurrence = target_cooccurrence(selected, BINARY_TARGETS)
    correlations = target_correlations(selected, BINARY_TARGETS, WATCH_TIME_FIELDS)
    user_coverage = coverage_rows(selected, BINARY_TARGETS, "user_id")
    item_coverage = coverage_rows(selected, BINARY_TARGETS, "item_id")
    drift = temporal_drift_rows(stats, BINARY_TARGETS)
    temporal_by_date = temporal_by_date_rows(selected, BINARY_TARGETS)
    watch = watch_time_diagnostics(selected)

    compact_paths = {
        "source_field_schema": paths.experiment_dir / "source_field_schema.csv",
        "target_statistics": paths.experiment_dir / "target_statistics.csv",
        "target_cooccurrence": paths.experiment_dir / "target_cooccurrence.csv",
        "target_correlations": paths.experiment_dir / "target_correlations.csv",
        "target_user_coverage": paths.experiment_dir / "target_user_coverage.csv",
        "target_item_coverage": paths.experiment_dir / "target_item_coverage.csv",
        "target_temporal_drift": paths.experiment_dir / "target_temporal_drift.csv",
        "target_temporal_by_date": paths.experiment_dir / "target_temporal_by_date.csv",
        "watch_time_diagnostics": paths.experiment_dir / "watch_time_diagnostics.csv",
    }
    write_csv(compact_paths["source_field_schema"], source_schema)
    write_csv(compact_paths["target_statistics"], stats)
    write_csv(compact_paths["target_cooccurrence"], cooccurrence)
    write_csv(compact_paths["target_correlations"], correlations)
    write_csv(compact_paths["target_user_coverage"], user_coverage)
    write_csv(compact_paths["target_item_coverage"], item_coverage)
    write_csv(compact_paths["target_temporal_drift"], drift)
    write_csv(compact_paths["target_temporal_by_date"], temporal_by_date)
    write_csv(compact_paths["watch_time_diagnostics"], watch)

    protocol_files = file_inventory(paths.protocol_dir)
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "multitask_tim4rec",
        "stage": "target_audit_only",
        "no_training_performed": True,
        "git": {
            "branch": args.git_branch or git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": args.git_commit or git_value(["git", "rev-parse", "HEAD"]),
            "working_directory": str(PROJECT_ROOT),
        },
        "source_protocol_b": {
            "manifest_path": str(paths.protocol_manifest),
            "manifest_sha256": sha256_file(paths.protocol_manifest),
            "remote_path": str(paths.protocol_dir),
            "fingerprint": protocol_manifest.get("filtered_stats", {}),
            "split_stats": protocol_manifest.get("split_stats", {}),
            "files": protocol_files,
        },
        "source_raw_files": [
            {
                "role": "KuaiRand-Pure early standard log",
                "path": str(paths.source_log),
                "size_bytes": paths.source_log.stat().st_size,
                "size": human_size(paths.source_log.stat().st_size),
                "sha256": sha256_file(paths.source_log),
                "schema_columns": source_columns,
            }
        ],
        "join_strategy": {
            "key": "source_row_id",
            "description": "Protocol B parquet stores zero-based raw CSV row index; raw CSV is scanned with the same zero-based row index and joined one-to-one.",
            "not_used_as_primary_key": ["user_id+item_id", "user_id+item_id+timestamp"],
            "reason": "Protocol B contains duplicate user/item/timestamp keys; source_row_id is the stable row identity.",
        },
        "join_diagnostics": join_diagnostics,
        "target_fields": {
            "raw_binary": list(BINARY_TARGETS),
            "watch_time": list(WATCH_TIME_FIELDS),
            "raw_behavior_fields_in_dataset": list(RAW_BEHAVIOR_FIELDS),
        },
        "derived_target_definitions": {
            name: {"formula": definition, "source_fields": list(columns)}
            for name, (definition, columns) in DERIVED_TARGETS.items()
        },
        "recommended_first_version_targets": ["is_click", "long_view", "is_like", "is_profile_enter"],
        "dataset_fingerprint": fingerprint,
        "remote_paths": {
            "multitask_dataset_dir": str(paths.output_dir),
            "train": str(paths.output_dir / "train.parquet"),
            "validation": str(paths.output_dir / "validation.parquet"),
            "test": str(paths.output_dir / "test.parquet"),
            "full_filtered": str(paths.output_dir / "full_filtered.parquet"),
        },
        "files": file_inventory(paths.output_dir),
        "compact_outputs": {key: str(path) for key, path in compact_paths.items()},
    }

    paths.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_text = build_audit_markdown(
        manifest,
        source_schema,
        stats,
        cooccurrence,
        correlations,
        user_coverage,
        item_coverage,
        drift,
        watch,
    )
    paths.audit_path.write_text(audit_text, encoding="utf-8")
    (paths.experiment_dir / "config.yaml").write_text(build_config_text(manifest), encoding="utf-8")

    run_summary = {
        "run_id": "target_audit_001",
        "status": "completed",
        "created_at_utc": manifest["created_at_utc"],
        "no_training_performed": True,
        "join_diagnostics": join_diagnostics,
        "dataset_fingerprint": fingerprint,
        "recommended_first_version_targets": manifest["recommended_first_version_targets"],
        "manifest_path": str(paths.manifest_path),
        "audit_path": str(paths.audit_path),
    }
    run_path = paths.experiment_dir / "runs" / "target_audit_001.json"
    notes_path = paths.experiment_dir / "runs" / "target_audit_001_notes.md"
    run_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    notes_path.write_text(
        "# target_audit_001\n\n"
        "Data-only audit завершен. Модель не обучалась, split Protocol B не менялся.\n\n"
        f"- Manifest: `{paths.manifest_path}`\n"
        f"- Audit: `{paths.audit_path}`\n"
        f"- Dataset: `{paths.output_dir}`\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "no_training_performed": True,
                "join_exact": join_diagnostics["join_is_exact"],
                "fingerprint_matches": fingerprint["matches_expected"]["all"],
                "manifest_path": str(paths.manifest_path),
                "audit_path": str(paths.audit_path),
                "output_dir": str(paths.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
