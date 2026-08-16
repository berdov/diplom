"""Utilities for KuaiRand exploratory data analysis.

The helpers in this module avoid loading full interaction logs into memory.
Heavy tabular operations are expressed as Polars lazy queries and collected
only after aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # Polars is installed on cHARISMa, but may be absent on a local laptop.
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    pl = None  # type: ignore[assignment]


INTERACTION_COLUMNS: tuple[str, ...] = (
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

FEEDBACK_COLUMNS: tuple[str, ...] = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "is_profile_enter",
)

WATCH_TIME_COLUMNS: tuple[str, ...] = ("play_time_ms", "duration_ms")


def require_polars() -> Any:
    """Return the Polars module or raise an actionable error."""

    if pl is None:
        raise ModuleNotFoundError(
            "Polars is required for KuaiRand EDA. It is installed in the "
            "cHARISMa environment at /home/daryumin/iberdov/diplom/.conda."
        )
    return pl


def human_size(num_bytes: int | float | None) -> str:
    """Format a byte count with binary units."""

    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def version_roots(data_root: Path) -> dict[str, Path]:
    """Return canonical KuaiRand version paths under ``data_root``."""

    return {
        "Pure": data_root / "KuaiRand-Pure" / "KuaiRand-Pure",
        "1K": data_root / "KuaiRand-1K" / "KuaiRand-1K",
        "27K": data_root / "KuaiRand-27K" / "KuaiRand-27K",
    }


def classify_dataset_file(path: Path) -> str:
    """Classify a KuaiRand file by name into a compact EDA category."""

    name = path.name.lower()
    suffix = path.suffix.lower()

    if name in {"readme", "readme.md", "license", "license.txt"}:
        return "README / documentation"
    if suffix == ".py" or "load_data" in name or "loader" in name:
        return "loader script"
    if "log_standard" in name:
        return "interaction log: standard"
    if "log_random" in name:
        return "interaction log: random"
    if "user" in name and "feature" in name:
        return "user features"
    if "video" in name and "basic" in name:
        return "video basic features"
    if "video" in name and ("stat" in name or "statistic" in name):
        return "video statistic features"
    if suffix == ".csv":
        return "other CSV table"
    if suffix == ".parquet":
        return "other parquet table"
    return "other"


def iter_files(root: Path, max_depth: int = 3) -> Iterable[Path]:
    """Yield files under ``root`` up to a shallow relative depth."""

    root = Path(root)
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and len(path.relative_to(root).parts) <= max_depth:
            yield path


def dataset_inventory(root: Path, max_depth: int = 3) -> list[dict[str, Any]]:
    """Build a lightweight inventory for a KuaiRand version directory."""

    root = Path(root)
    if not root.exists():
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(iter_files(root, max_depth=max_depth)):
        size_bytes = path.stat().st_size
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "file_name": path.name,
                "suffix": path.suffix.lower() or "(none)",
                "category": classify_dataset_file(path),
                "size_bytes": size_bytes,
                "size": human_size(size_bytes),
            }
        )
    return rows


def total_size_bytes(inventory_rows: Sequence[Mapping[str, Any]]) -> int:
    """Return the summed file size from inventory rows."""

    return int(sum(int(row.get("size_bytes", 0) or 0) for row in inventory_rows))


def files_matching(
    root: Path,
    include_tokens: Sequence[str],
    exclude_tokens: Sequence[str] = (),
    suffixes: Sequence[str] = (".csv", ".parquet"),
) -> list[Path]:
    """Find dataset files whose normalized names contain all include tokens."""

    root = Path(root)
    if not root.exists():
        return []

    include = tuple(token.lower() for token in include_tokens)
    exclude = tuple(token.lower() for token in exclude_tokens)
    suffix_set = {suffix.lower() for suffix in suffixes}

    matches: list[Path] = []
    for path in iter_files(root, max_depth=3):
        name = path.name.lower()
        if path.suffix.lower() not in suffix_set:
            continue
        if all(token in name for token in include) and not any(
            token in name for token in exclude
        ):
            matches.append(path)
    return sorted(matches)


def first_matching_file(
    root: Path,
    include_tokens: Sequence[str],
    exclude_tokens: Sequence[str] = (),
    suffixes: Sequence[str] = (".csv", ".parquet"),
) -> Path | None:
    """Return the first matching file, or ``None`` when absent."""

    matches = files_matching(root, include_tokens, exclude_tokens, suffixes)
    return matches[0] if matches else None


def discover_kuairand_files(root: Path) -> dict[str, Path | None]:
    """Discover common KuaiRand tables without hard-coding version suffixes."""

    return {
        "standard_early": first_matching_file(
            root, ("log_standard", "4_08", "4_21")
        ),
        "standard_late": first_matching_file(
            root, ("log_standard", "4_22", "5_08")
        ),
        "random": first_matching_file(root, ("log_random", "4_22", "5_08")),
        "user_features": first_matching_file(root, ("user", "feature")),
        "video_basic": first_matching_file(root, ("video", "basic")),
        "video_statistics": first_matching_file(root, ("video", "stat")),
    }


def discover_kuairand_file_groups(root: Path) -> dict[str, list[Path]]:
    """Discover all common KuaiRand table files grouped by logical role."""

    return {
        "standard_early": files_matching(root, ("log_standard", "4_08", "4_21")),
        "standard_late": files_matching(root, ("log_standard", "4_22", "5_08")),
        "random": files_matching(root, ("log_random", "4_22", "5_08")),
        "user_features": files_matching(root, ("user", "feature")),
        "video_basic": files_matching(root, ("video", "basic")),
        "video_statistics": files_matching(root, ("video", "stat")),
    }


def scan_table(path: Path, **scan_kwargs: Any) -> Any:
    """Create a lazy scan for CSV or parquet tables."""

    polars = require_polars()
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return polars.scan_parquet(str(path), **scan_kwargs)
    if suffix == ".csv":
        kwargs = {
            "infer_schema_length": 10_000,
            "low_memory": True,
            "try_parse_dates": False,
        }
        kwargs.update(scan_kwargs)
        return polars.scan_csv(str(path), **kwargs)
    raise ValueError(f"Unsupported table format: {path}")


def collect_lazy(lazy_frame: Any) -> Any:
    """Collect a Polars LazyFrame with streaming when the installed version supports it."""

    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:  # Older Polars versions used a boolean flag.
        return lazy_frame.collect(streaming=True)


def lazy_schema_names(lazy_frame: Any) -> list[str]:
    """Return column names from a Polars LazyFrame without materializing rows."""

    if hasattr(lazy_frame, "collect_schema"):
        return list(lazy_frame.collect_schema().names())
    return list(lazy_frame.schema.keys())


def concat_lazy_frames(lazy_frames: Sequence[Any]) -> Any:
    """Concatenate lazy frames while tolerating minor schema differences."""

    polars = require_polars()
    if not lazy_frames:
        raise ValueError("No LazyFrames were provided for concatenation.")
    if len(lazy_frames) == 1:
        return lazy_frames[0]
    try:
        return polars.concat(list(lazy_frames), how="diagonal_relaxed")
    except Exception:
        return polars.concat(list(lazy_frames), how="vertical_relaxed")


def available_columns(lazy_frame: Any, columns: Sequence[str]) -> list[str]:
    """Return requested columns that exist in a lazy frame."""

    names = set(lazy_schema_names(lazy_frame))
    return [column for column in columns if column in names]


def safe_percentiles(
    values: Sequence[float] | np.ndarray,
    percentiles: Sequence[float] = (0, 25, 50, 75, 90, 95, 99, 100),
) -> dict[str, float | None]:
    """Calculate percentiles after dropping non-finite values."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {f"p{int(p)}": None for p in percentiles}
    return {f"p{int(p)}": float(np.percentile(array, p)) for p in percentiles}


def numeric_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    """Return compact numeric summary statistics for a vector."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def ensure_output_dir(path: Path) -> Path:
    """Create and return an output directory."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
