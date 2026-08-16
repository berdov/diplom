"""Вспомогательные функции для exploratory data analysis KuaiRand.

Функции в этом модуле не загружают полные логи взаимодействий в память.
Тяжелые табличные операции описываются как lazy-запросы Polars и собираются
только после агрегации.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # Polars установлен на cHARISMa, но может отсутствовать локально.
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
    """Вернуть модуль Polars или выбросить ошибку с понятным действием."""

    if pl is None:
        raise ModuleNotFoundError(
            "Для KuaiRand EDA нужен Polars. Он установлен в окружении cHARISMa "
            "по пути /home/daryumin/iberdov/diplom/.conda."
        )
    return pl


def human_size(num_bytes: int | float | None) -> str:
    """Отформатировать размер в байтах через бинарные единицы."""

    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def version_roots(data_root: Path) -> dict[str, Path]:
    """Вернуть канонические пути версий KuaiRand внутри ``data_root``."""

    return {
        "Pure": data_root / "KuaiRand-Pure" / "KuaiRand-Pure",
        "1K": data_root / "KuaiRand-1K" / "KuaiRand-1K",
        "27K": data_root / "KuaiRand-27K" / "KuaiRand-27K",
    }


def classify_dataset_file(path: Path) -> str:
    """Классифицировать файл KuaiRand по имени в компактную EDA-категорию."""

    name = path.name.lower()
    suffix = path.suffix.lower()

    if name in {"readme", "readme.md", "license", "license.txt"}:
        return "README / документация"
    if suffix == ".py" or "load_data" in name or "loader" in name:
        return "скрипт загрузки"
    if "log_standard" in name:
        return "лог взаимодействий: standard"
    if "log_random" in name:
        return "лог взаимодействий: random"
    if "user" in name and "feature" in name:
        return "признаки пользователей"
    if "video" in name and "basic" in name:
        return "базовые признаки видео"
    if "video" in name and ("stat" in name or "statistic" in name):
        return "статистические признаки видео"
    if suffix == ".csv":
        return "другая CSV-таблица"
    if suffix == ".parquet":
        return "другая parquet-таблица"
    return "другое"


def iter_files(root: Path, max_depth: int = 3) -> Iterable[Path]:
    """Итерироваться по файлам внутри ``root`` до заданной глубины."""

    root = Path(root)
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and len(path.relative_to(root).parts) <= max_depth:
            yield path


def dataset_inventory(root: Path, max_depth: int = 3) -> list[dict[str, Any]]:
    """Собрать легкую инвентаризацию директории версии KuaiRand."""

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
    """Вернуть суммарный размер файлов по строкам inventory."""

    return int(sum(int(row.get("size_bytes", 0) or 0) for row in inventory_rows))


def files_matching(
    root: Path,
    include_tokens: Sequence[str],
    exclude_tokens: Sequence[str] = (),
    suffixes: Sequence[str] = (".csv", ".parquet"),
) -> list[Path]:
    """Найти файлы датасета, чьи нормализованные имена содержат все include-токены."""

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
    """Вернуть первый подходящий файл или ``None``, если его нет."""

    matches = files_matching(root, include_tokens, exclude_tokens, suffixes)
    return matches[0] if matches else None


def discover_kuairand_files(root: Path) -> dict[str, Path | None]:
    """Найти основные таблицы KuaiRand без жесткой привязки к суффиксам версий."""

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
    """Найти основные таблицы KuaiRand и сгруппировать их по логической роли."""

    return {
        "standard_early": files_matching(root, ("log_standard", "4_08", "4_21")),
        "standard_late": files_matching(root, ("log_standard", "4_22", "5_08")),
        "random": files_matching(root, ("log_random", "4_22", "5_08")),
        "user_features": files_matching(root, ("user", "feature")),
        "video_basic": files_matching(root, ("video", "basic")),
        "video_statistics": files_matching(root, ("video", "stat")),
    }


def scan_table(path: Path, **scan_kwargs: Any) -> Any:
    """Создать lazy scan для CSV или parquet-таблицы."""

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
    raise ValueError(f"Неподдерживаемый формат таблицы: {path}")


def collect_lazy(lazy_frame: Any) -> Any:
    """Собрать Polars LazyFrame в streaming-режиме, если установленная версия это поддерживает."""

    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:  # В старых версиях Polars использовался булев флаг.
        return lazy_frame.collect(streaming=True)


def lazy_schema_names(lazy_frame: Any) -> list[str]:
    """Вернуть имена колонок Polars LazyFrame без материализации строк."""

    if hasattr(lazy_frame, "collect_schema"):
        return list(lazy_frame.collect_schema().names())
    return list(lazy_frame.schema.keys())


def concat_lazy_frames(lazy_frames: Sequence[Any]) -> Any:
    """Склеить lazy frames с учетом небольших различий схемы."""

    polars = require_polars()
    if not lazy_frames:
        raise ValueError("Для конкатенации не передано ни одного LazyFrame.")
    if len(lazy_frames) == 1:
        return lazy_frames[0]
    try:
        return polars.concat(list(lazy_frames), how="diagonal_relaxed")
    except Exception:
        return polars.concat(list(lazy_frames), how="vertical_relaxed")


def available_columns(lazy_frame: Any, columns: Sequence[str]) -> list[str]:
    """Вернуть запрошенные колонки, которые существуют в lazy frame."""

    names = set(lazy_schema_names(lazy_frame))
    return [column for column in columns if column in names]


def safe_percentiles(
    values: Sequence[float] | np.ndarray,
    percentiles: Sequence[float] = (0, 25, 50, 75, 90, 95, 99, 100),
) -> dict[str, float | None]:
    """Посчитать перцентили после удаления нечисловых и бесконечных значений."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {f"p{int(p)}": None for p in percentiles}
    return {f"p{int(p)}": float(np.percentile(array, p)) for p in percentiles}


def numeric_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    """Вернуть компактную числовую сводку для вектора."""

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
    """Создать и вернуть выходную директорию."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Записать JSON со стабильным форматированием."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
