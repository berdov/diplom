#!/usr/bin/env python3
"""Подготовка KuaiRand Protocol B для sequential recommendation.

Protocol B повторяет схему SSD4Rec / TiM4Rec для KuaiRand-Pure:

* исходный лог: KuaiRand-Pure standard за 2022-04-08 -- 2022-04-21;
* поля взаимодействий: user_id, video_id как item_id, time_ms как timestamp;
* итеративный 5-core по пользователям и объектам в стиле RecBole;
* хронологический leave-one-out split: train / validation / test;
* максимальная длина последовательности в configs: 50.

Скрипт намеренно не использует random logs и 322M-row KuaiRand-27K standard logs:
это другой benchmark protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable


EXPECTED_USERS = 23_951
EXPECTED_ITEMS = 7_111
EXPECTED_INTERACTIONS = 1_134_420
DEFAULT_DATA_ROOT = Path("/home/daryumin/iberdov/Corpora")
DEFAULT_PROJECT_ROOT = Path("/home/daryumin/iberdov/diplom")


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    timestamp: int
    source_row_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Корень данных, где лежит KuaiRand-Pure/KuaiRand-Pure.",
    )
    parser.add_argument(
        "--source-log",
        type=Path,
        default=None,
        help="Явный путь к исходному CSV. По умолчанию берётся ранний KuaiRand-Pure standard log.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "data" / "processed" / "protocol_b",
        help="Удалённый/persistent каталог для processed datasets.",
    )
    parser.add_argument(
        "--repo-output-dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "outputs" / "data",
        help="Каталог для компактных manifest/stats, которые можно коммитить.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "reports" / "kuairand_protocol_b_data_report.md",
        help="Путь к Markdown-отчёту с валидацией.",
    )
    parser.add_argument("--min-core", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--sanity-limit", type=int, default=None)
    parser.add_argument(
        "--git-commit",
        default=None,
        help="Git commit кода препроцессинга. По умолчанию git rev-parse HEAD.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Не писать Parquet; полезно только для локального smoke test без зависимостей.",
    )
    return parser.parse_args()


def source_log_path(data_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return (
        data_root.expanduser().resolve()
        / "KuaiRand-Pure"
        / "KuaiRand-Pure"
        / "data"
        / "log_standard_4_08_to_4_21_pure.csv"
    )


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def git_commit(default: str | None = None) -> str | None:
    if default:
        return default
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root_from_file(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        return pa, pq
    except ImportError as exc:
        raise RuntimeError(
            "Для Parquet outputs нужен pyarrow. Запустите в .conda на cHARISMa "
            "или передайте --no-parquet для smoke test."
        ) from exc


def read_source(path: Path, sanity_limit: int | None = None) -> tuple[list[Interaction], dict[str, object]]:
    rows: list[Interaction] = []
    missing_id_rows = 0
    invalid_timestamp_rows = 0
    is_rand_values: set[str] = set()
    date_min: int | None = None
    date_max: int | None = None
    required = {"user_id", "video_id", "time_ms", "date", "is_rand"}

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV без заголовка: {path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"В CSV {path} не хватает обязательных колонок: {sorted(missing)}")

        for source_row_id, row in enumerate(reader):
            if sanity_limit is not None and len(rows) >= sanity_limit:
                break
            try:
                user_raw = row["user_id"]
                item_raw = row["video_id"]
                time_raw = row["time_ms"]
                if user_raw == "" or item_raw == "":
                    missing_id_rows += 1
                    continue
                timestamp = int(float(time_raw))
                user_id = int(user_raw)
                item_id = int(item_raw)
            except Exception:
                invalid_timestamp_rows += 1
                continue

            if "date" in row and row["date"] != "":
                date_value = int(row["date"])
                date_min = date_value if date_min is None else min(date_min, date_value)
                date_max = date_value if date_max is None else max(date_max, date_value)
            is_rand_values.add(row.get("is_rand", ""))
            rows.append(Interaction(user_id, item_id, timestamp, source_row_id))

    stats = {
        "source_log": str(path),
        "source_size_bytes": path.stat().st_size,
        "source_size": human_size(path.stat().st_size),
        "rows_read": len(rows),
        "sanity_limit": sanity_limit,
        "missing_id_rows_skipped": missing_id_rows,
        "invalid_timestamp_rows_skipped": invalid_timestamp_rows,
        "date_min": date_min,
        "date_max": date_max,
        "is_rand_values": sorted(is_rand_values),
    }
    return rows, stats


def count_stats(rows: list[Interaction]) -> dict[str, object]:
    user_counts = Counter(row.user_id for row in rows)
    item_counts = Counter(row.item_id for row in rows)
    exact_counts = Counter((row.user_id, row.item_id, row.timestamp) for row in rows)
    user_time_counts = Counter((row.user_id, row.timestamp) for row in rows)
    timestamps = [row.timestamp for row in rows]
    return {
        "interactions": len(rows),
        "users": len(user_counts),
        "items": len(item_counts),
        "user_min_interactions": min(user_counts.values()) if user_counts else 0,
        "item_min_interactions": min(item_counts.values()) if item_counts else 0,
        "user_max_interactions": max(user_counts.values()) if user_counts else 0,
        "item_max_interactions": max(item_counts.values()) if item_counts else 0,
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
        "exact_duplicate_keys": sum(1 for count in exact_counts.values() if count > 1),
        "exact_duplicate_extra_rows": sum(count - 1 for count in exact_counts.values() if count > 1),
        "user_timestamp_duplicate_keys": sum(1 for count in user_time_counts.values() if count > 1),
        "user_timestamp_duplicate_extra_rows": sum(
            count - 1 for count in user_time_counts.values() if count > 1
        ),
        "user_timestamp_max_rows_per_key": max(user_time_counts.values()) if user_time_counts else 0,
    }


def numeric_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p999": None,
            "max": None,
        }
    sorted_values = sorted(values)

    def percentile(q: float) -> float:
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        pos = (len(sorted_values) - 1) * q
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return float(sorted_values[lower])
        weight = pos - lower
        return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)

    return {
        "count": len(sorted_values),
        "min": int(sorted_values[0]),
        "mean": float(mean(sorted_values)),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "p999": percentile(0.999),
        "max": int(sorted_values[-1]),
    }


def iterative_k_core(rows: list[Interaction], min_core: int) -> tuple[list[Interaction], list[dict[str, int]]]:
    active = bytearray([1]) * len(rows)
    iterations: list[dict[str, int]] = []
    iteration = 0

    while True:
        user_counts: Counter[int] = Counter()
        item_counts: Counter[int] = Counter()
        for is_active, row in zip(active, rows):
            if is_active:
                user_counts[row.user_id] += 1
                item_counts[row.item_id] += 1

        bad_users = {user for user, count in user_counts.items() if count < min_core}
        bad_items = {item for item, count in item_counts.items() if count < min_core}
        if not bad_users and not bad_items:
            break

        dropped = 0
        for idx, is_active in enumerate(active):
            if not is_active:
                continue
            row = rows[idx]
            if row.user_id in bad_users or row.item_id in bad_items:
                active[idx] = 0
                dropped += 1

        iteration += 1
        iterations.append(
            {
                "iteration": iteration,
                "bad_users": len(bad_users),
                "bad_items": len(bad_items),
                "dropped_interactions": dropped,
                "remaining_interactions": int(sum(active)),
            }
        )

    return [row for is_active, row in zip(active, rows) if is_active], iterations


def chronological_split(rows: list[Interaction]) -> dict[str, list[Interaction]]:
    ordered = sorted(rows, key=lambda row: (row.user_id, row.timestamp, row.source_row_id))
    splits = {"train": [], "validation": [], "test": []}
    start = 0
    while start < len(ordered):
        end = start + 1
        user_id = ordered[start].user_id
        while end < len(ordered) and ordered[end].user_id == user_id:
            end += 1
        user_rows = ordered[start:end]
        if len(user_rows) < 3:
            raise ValueError(f"User {user_id} has only {len(user_rows)} interactions after filtering")
        splits["train"].extend(user_rows[:-2])
        splits["validation"].append(user_rows[-2])
        splits["test"].append(user_rows[-1])
        start = end
    return splits


def write_interaction_parquet(path: Path, rows: list[Interaction]) -> None:
    pa, pq = ensure_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "user_id": [row.user_id for row in rows],
            "item_id": [row.item_id for row in rows],
            "timestamp": [row.timestamp for row in rows],
            "source_row_id": [row.source_row_id for row in rows],
        }
    )
    pq.write_table(table, path, compression="zstd")


def write_sequence_parquet(path: Path, rows: list[Interaction]) -> dict[str, object]:
    pa, pq = ensure_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row.user_id, row.timestamp, row.source_row_id))
    user_ids: list[int] = []
    item_sequences: list[list[int]] = []
    timestamp_sequences: list[list[int]] = []
    train_item_sequences: list[list[int]] = []
    train_timestamp_sequences: list[list[int]] = []
    validation_item_ids: list[int] = []
    validation_timestamps: list[int] = []
    test_item_ids: list[int] = []
    test_timestamps: list[int] = []
    sequence_lengths: list[int] = []
    train_lengths: list[int] = []

    start = 0
    while start < len(ordered):
        end = start + 1
        user_id = ordered[start].user_id
        while end < len(ordered) and ordered[end].user_id == user_id:
            end += 1
        user_rows = ordered[start:end]
        items = [row.item_id for row in user_rows]
        timestamps = [row.timestamp for row in user_rows]
        user_ids.append(user_id)
        item_sequences.append(items)
        timestamp_sequences.append(timestamps)
        train_item_sequences.append(items[:-2])
        train_timestamp_sequences.append(timestamps[:-2])
        validation_item_ids.append(items[-2])
        validation_timestamps.append(timestamps[-2])
        test_item_ids.append(items[-1])
        test_timestamps.append(timestamps[-1])
        sequence_lengths.append(len(items))
        train_lengths.append(len(items) - 2)
        start = end

    table = pa.table(
        {
            "user_id": pa.array(user_ids, type=pa.int64()),
            "item_sequence": pa.array(item_sequences, type=pa.list_(pa.int64())),
            "timestamp_sequence": pa.array(timestamp_sequences, type=pa.list_(pa.int64())),
            "train_item_sequence": pa.array(train_item_sequences, type=pa.list_(pa.int64())),
            "train_timestamp_sequence": pa.array(
                train_timestamp_sequences, type=pa.list_(pa.int64())
            ),
            "validation_item_id": pa.array(validation_item_ids, type=pa.int64()),
            "validation_timestamp": pa.array(validation_timestamps, type=pa.int64()),
            "test_item_id": pa.array(test_item_ids, type=pa.int64()),
            "test_timestamp": pa.array(test_timestamps, type=pa.int64()),
            "sequence_length": pa.array(sequence_lengths, type=pa.int64()),
            "train_length": pa.array(train_lengths, type=pa.int64()),
        }
    )
    pq.write_table(table, path, compression="zstd")
    return {
        "sequence_length": numeric_summary(sequence_lengths),
        "train_length": numeric_summary(train_lengths),
    }


def write_mapping_parquet(path: Path, values: Iterable[int], source_name: str, mapped_name: str) -> None:
    pa, pq = ensure_pyarrow()
    unique_values = sorted(set(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            source_name: pa.array(unique_values, type=pa.int64()),
            mapped_name: pa.array(list(range(1, len(unique_values) + 1)), type=pa.int64()),
        }
    )
    pq.write_table(table, path, compression="zstd")


def write_recbole_inter(path: Path, rows: list[Interaction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row.user_id, row.timestamp, row.source_row_id))
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for row in ordered:
            fh.write(f"{row.user_id}\t{row.item_id}\t{row.timestamp}\n")


def write_recbole_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# KuaiRand Protocol B: sequential benchmark SSD4Rec / TiM4Rec.
dataset: kuairand
MAX_ITEM_LIST_LENGTH: 50

USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
load_col:
  inter: [user_id, item_id, timestamp]

user_inter_num_interval: "[5,inf)"
item_inter_num_interval: "[5,inf)"
rm_dup_inter: ~
train_neg_sample_args: ~

eval_args:
  split: {'LS': 'valid_and_test'}
  order: TO
  group_by: user
  mode: full

metrics: ['Hit', 'NDCG', 'MRR']
valid_metric: NDCG@10
eval_batch_size: 4096
topk: [10, 20, 50]
""",
        encoding="utf-8",
    )


def validation_summary(
    filtered_rows: list[Interaction],
    splits: dict[str, list[Interaction]],
    expected_users: int,
    expected_items: int,
    expected_interactions: int,
) -> dict[str, object]:
    filtered_users = {row.user_id for row in filtered_rows}
    filtered_items = {row.item_id for row in filtered_rows}
    train_users = {row.user_id for row in splits["train"]}
    validation_users = {row.user_id for row in splits["validation"]}
    test_users = {row.user_id for row in splits["test"]}
    split_rows_total = sum(len(part) for part in splits.values())

    leakage_violations = 0
    by_user: dict[int, dict[str, int]] = {}
    for split_name, part in splits.items():
        for row in part:
            record = by_user.setdefault(row.user_id, {})
            key = f"{split_name}_max" if split_name == "train" else f"{split_name}_time"
            if split_name == "train":
                record[key] = max(record.get(key, row.timestamp), row.timestamp)
            else:
                record[key] = row.timestamp
    for record in by_user.values():
        train_max = record.get("train_max")
        validation_time = record.get("validation_time")
        test_time = record.get("test_time")
        if train_max is None or validation_time is None or test_time is None:
            leakage_violations += 1
        elif train_max > validation_time or validation_time > test_time:
            leakage_violations += 1

    stats = count_stats(filtered_rows)
    return {
        "expected_fingerprint": {
            "users": expected_users,
            "items": expected_items,
            "interactions": expected_interactions,
        },
        "fingerprint_matches_expected": {
            "users": stats["users"] == expected_users,
            "items": stats["items"] == expected_items,
            "interactions": stats["interactions"] == expected_interactions,
            "all": (
                stats["users"] == expected_users
                and stats["items"] == expected_items
                and stats["interactions"] == expected_interactions
            ),
        },
        "split_rows_total": split_rows_total,
        "split_rows_match_filtered": split_rows_total == len(filtered_rows),
        "train_users": len(train_users),
        "validation_users": len(validation_users),
        "test_users": len(test_users),
        "validation_users_missing_from_train": len(validation_users - train_users),
        "test_users_missing_from_train": len(test_users - train_users),
        "users_outside_filtered_universe": sum(
            1
            for part in splits.values()
            for row in part
            if row.user_id not in filtered_users
        ),
        "items_outside_filtered_universe": sum(
            1
            for part in splits.values()
            for row in part
            if row.item_id not in filtered_items
        ),
        "temporal_leakage_violations": leakage_violations,
        "min_train_interactions_per_user": min(Counter(row.user_id for row in splits["train"]).values())
        if splits["train"]
        else 0,
    }


def file_inventory(output_dir: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    metadata_files = {"manifest.json", "stats.json", "validation_summary.csv"}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.relative_to(output_dir).as_posix() not in metadata_files:
            files.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(output_dir)),
                    "size_bytes": path.stat().st_size,
                    "size": human_size(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return files


def write_validation_csv(path: Path, manifest: dict[str, object]) -> None:
    rows = [
        ("raw_interactions", manifest["raw_stats"]["interactions"]),
        ("raw_users", manifest["raw_stats"]["users"]),
        ("raw_items", manifest["raw_stats"]["items"]),
        ("filtered_interactions", manifest["filtered_stats"]["interactions"]),
        ("filtered_users", manifest["filtered_stats"]["users"]),
        ("filtered_items", manifest["filtered_stats"]["items"]),
        ("train_interactions", manifest["split_stats"]["train"]["interactions"]),
        ("validation_interactions", manifest["split_stats"]["validation"]["interactions"]),
        ("test_interactions", manifest["split_stats"]["test"]["interactions"]),
        (
            "fingerprint_matches_expected",
            manifest["validation"]["fingerprint_matches_expected"]["all"],
        ),
        ("temporal_leakage_violations", manifest["validation"]["temporal_leakage_violations"]),
        (
            "validation_users_missing_from_train",
            manifest["validation"]["validation_users_missing_from_train"],
        ),
        ("test_users_missing_from_train", manifest["validation"]["test_users_missing_from_train"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def format_int(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.3f}"
    return str(value)


def build_report(manifest: dict[str, object]) -> str:
    raw = manifest["raw_stats"]
    filtered = manifest["filtered_stats"]
    split = manifest["split_stats"]
    validation = manifest["validation"]
    seq = manifest["sequence_stats"]["sequence_length"]
    files = manifest["files"]
    tim = manifest["protocol_sources"]["tim4rec"]

    lines = [
        "# Отчёт по данным KuaiRand Protocol B",
        "",
        "Отчёт построен по фактическому полному препроцессингу на cHARISMa. Большие обработанные датасеты сохранены во внешнем каталоге, в Git попадают только код, конфигурация, manifest, checksums и компактные stats.",
        "",
        "## 1. Протокол из источников",
        "",
        f"- Статья SSD4Rec: {manifest['protocol_sources']['ssd4rec']['paper_url']}.",
        f"- Статья TiM4Rec: {manifest['protocol_sources']['tim4rec']['paper_url']}.",
        f"- Официальный репозиторий TiM4Rec: {manifest['protocol_sources']['tim4rec']['repo_url']}.",
        "- SSD4Rec: бенчмарк KuaiRand имеет контрольный fingerprint `23,951 users / 7,111 items / 1,134,420 interactions`, использует leave-one-out разбиение по SASRec и `MAX_ITEM_LIST_LENGTH=50` для KuaiRand.",
        "- TiM4Rec: сортировка по timestamp, минимум 5 interactions для users/items, тот же fingerprint `23,951 / 7,111 / 1,134,420`.",
        f"- Официальный config TiM4Rec `{tim['kuairand_config_path']}` задаёт `MAX_ITEM_LIST_LENGTH=50`, `load_col=[user_id,item_id,timestamp]`, `user_inter_num_interval=[5,inf)`, `item_inter_num_interval=[5,inf)`, `train_neg_sample_args=~`, но не задаёт явный `eval_args`.",
        f"- Поэтому для TiM4Rec применяется стандартная sequential-настройка RecBole 1.2.0: `{tim['recbole_sequential_default_eval_args']}`.",
        "- Канонический вариант B в этом репозитории: совместимый с SSD4Rec/SASRec/TiM4Rec хронологический leave-one-out по раннему standard log из KuaiRand-Pure.",
        "",
        "## 2. Исходные данные",
        "",
        f"- Исходный лог: `{manifest['source']['source_log']}`.",
        f"- Прочитано строк: {format_int(raw['interactions'])}.",
        f"- Исходные users/items: {format_int(raw['users'])} / {format_int(raw['items'])}.",
        f"- Диапазон дат: `{manifest['source']['date_min']}`-`{manifest['source']['date_max']}`.",
        f"- Значения `is_rand`: `{manifest['source']['is_rand_values']}`. Random logs в Protocol B не используются.",
        "",
        "## 3. Фильтрация",
        "",
        f"- Правило фильтрации: итеративный RecBole-style {manifest['filtering']['min_core']}-core по users и items.",
        "- Дубликаты interactions не удаляются, что соответствует default `rm_dup_inter: ~` в RecBole.",
        f"- Итераций k-core: {len(manifest['filtering']['iterations'])}.",
        f"- Итоговые users/items/interactions: {format_int(filtered['users'])} / {format_int(filtered['items'])} / {format_int(filtered['interactions'])}.",
        f"- Минимум interactions на user/item после фильтрации: {filtered['user_min_interactions']} / {filtered['item_min_interactions']}.",
        f"- Совпадение с ожидаемым fingerprint: `{validation['fingerprint_matches_expected']['all']}`.",
        "",
        "## 4. Разбиение",
        "",
        f"- Правило разрешения равных timestamp: `{manifest['split']['tie_breaking_rule']}`.",
        f"- Interactions в train: {format_int(split['train']['interactions'])}.",
        f"- Interactions в validation: {format_int(split['validation']['interactions'])}.",
        f"- Interactions в test: {format_int(split['test']['interactions'])}.",
        f"- Users в train/validation/test: {format_int(validation['train_users'])} / {format_int(validation['validation_users'])} / {format_int(validation['test_users'])}.",
        f"- Длина последовательности median/p95/max: {seq['median']:.1f} / {seq['p95']:.1f} / {format_int(seq['max'])}.",
        "",
        "## 5. Валидация",
        "",
        f"- Лишние строки exact duplicate после фильтрации: {format_int(filtered['exact_duplicate_extra_rows'])}; протокол их не удаляет.",
        f"- Лишние строки user+timestamp duplicate после фильтрации: {format_int(filtered['user_timestamp_duplicate_extra_rows'])}.",
        f"- Нарушения временного порядка: {format_int(validation['temporal_leakage_violations'])}.",
        f"- Validation users без истории в train: {format_int(validation['validation_users_missing_from_train'])}.",
        f"- Test users без истории в train: {format_int(validation['test_users_missing_from_train'])}.",
        f"- Split полностью покрывает filtered rows: `{validation['split_rows_match_filtered']}`.",
        "",
        "## 6. Воспроизводимость",
        "",
        f"- Commit кода препроцессинга: `{manifest['git']['preprocessing_code_commit']}`.",
        f"- Скрипт: `{manifest['preprocessing_script']}`.",
        f"- Slurm-скрипт: `slurm/prepare_protocol_b.sh`.",
        f"- Путь хранения на кластере: `{manifest['storage']['remote_path']}`.",
        f"- Путь manifest: `{manifest['storage']['manifest_path']}`.",
        "",
        "Команда sanity-прогона:",
        "",
        "```bash",
        "python src/prepare_kuairand_protocol_b.py --sanity-limit 10000 --output-dir data/processed/protocol_b_sanity --repo-output-dir outputs/data_sanity --report-path reports/kuairand_protocol_b_data_report_sanity.md",
        "```",
        "",
        "Команда полного Slurm-прогона:",
        "",
        "```bash",
        "sbatch slurm/prepare_protocol_b.sh",
        "```",
        "",
        "## 7. Файлы",
        "",
        "| relative_path | размер | sha256 |",
        "| --- | --- | --- |",
    ]
    for file_info in files:
        lines.append(
            f"| `{file_info['relative_path']}` | {file_info['size']} | `{file_info['sha256']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def copy_if_different(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source_path = source_log_path(args.data_root, args.source_log)
    output_dir = args.output_dir.expanduser().resolve()
    repo_output_dir = args.repo_output_dir.expanduser().resolve()
    report_path = args.report_path.expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Исходный log не найден: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    repo_output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Исходный log: {source_path}")
    print(f"Каталог output: {output_dir}")
    rows, source_stats = read_source(source_path, args.sanity_limit)
    raw_stats = count_stats(rows)
    print(f"Исходные rows/users/items: {raw_stats['interactions']} / {raw_stats['users']} / {raw_stats['items']}")

    filtered_rows, kcore_iterations = iterative_k_core(rows, args.min_core)
    filtered_stats = count_stats(filtered_rows)
    print(
        "После фильтрации rows/users/items: "
        f"{filtered_stats['interactions']} / {filtered_stats['users']} / {filtered_stats['items']}"
    )

    splits = chronological_split(filtered_rows)
    split_stats = {name: count_stats(part) for name, part in splits.items()}
    validation = validation_summary(
        filtered_rows,
        splits,
        EXPECTED_USERS if args.sanity_limit is None else filtered_stats["users"],
        EXPECTED_ITEMS if args.sanity_limit is None else filtered_stats["items"],
        EXPECTED_INTERACTIONS if args.sanity_limit is None else filtered_stats["interactions"],
    )

    parquet_files: list[Path] = []
    if not args.no_parquet:
        write_interaction_parquet(output_dir / "full_filtered.parquet", filtered_rows)
        parquet_files.append(output_dir / "full_filtered.parquet")
        for split_name, split_rows in splits.items():
            write_interaction_parquet(output_dir / f"{split_name}.parquet", split_rows)
            parquet_files.append(output_dir / f"{split_name}.parquet")
        sequence_stats = write_sequence_parquet(output_dir / "sequences.parquet", filtered_rows)
        parquet_files.append(output_dir / "sequences.parquet")
        write_mapping_parquet(
            output_dir / "user_id_mapping.parquet",
            (row.user_id for row in filtered_rows),
            "user_id",
            "user_index",
        )
        write_mapping_parquet(
            output_dir / "item_id_mapping.parquet",
            (row.item_id for row in filtered_rows),
            "item_id",
            "item_index",
        )
        parquet_files.extend([output_dir / "user_id_mapping.parquet", output_dir / "item_id_mapping.parquet"])
    else:
        sequence_lengths = list(Counter(row.user_id for row in filtered_rows).values())
        sequence_stats = {
            "sequence_length": numeric_summary(sequence_lengths),
            "train_length": numeric_summary([length - 2 for length in sequence_lengths]),
        }

    recbole_dir = output_dir / "recbole" / "kuairand"
    write_recbole_inter(recbole_dir / "kuairand.inter", filtered_rows)
    write_recbole_config(output_dir / "recbole" / "kuairand_protocol_b.yaml")

    manifest: dict[str, object] = {
        "protocol_name": "KuaiRand Protocol B: SSD4Rec / TiM4Rec sequential benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_script": "src/prepare_kuairand_protocol_b.py",
        "git": {
            "preprocessing_code_commit": git_commit(args.git_commit),
            "working_directory": str(project_root_from_file()),
        },
        "source": source_stats,
        "protocol_sources": {
            "ssd4rec": {
                "paper_url": "https://arxiv.org/html/2409.01192v1",
                "repo_url": "https://github.com/ZhangYifeng1995/SSD4Rec",
                "dataset": "KuaiRand-Pure",
                "paper_fingerprint": {
                    "users": EXPECTED_USERS,
                    "items": EXPECTED_ITEMS,
                    "interactions": EXPECTED_INTERACTIONS,
                },
                "paper_split": "leave-one-out policy following SASRec",
                "max_item_list_length": args.max_seq_len,
            },
            "tim4rec": {
                "paper_url": "https://arxiv.org/html/2409.16182v1",
                "repo_url": "https://github.com/AlwaysFHao/TiM4Rec",
                "kuairand_config_path": "config/config4kuai_64d.yaml",
                "explicit_eval_args_in_config": False,
                "recbole_version_in_readme": "1.2.0",
                "recbole_sequential_default_eval_args": {
                    "split": {"LS": "valid_and_test"},
                    "order": "TO",
                    "group_by": "user",
                    "mode": {"valid": "full", "test": "full"},
                },
            },
        },
        "filtering": {
            "min_core": args.min_core,
            "implementation": "итеративный RecBole-style k-core по user_id и item_id",
            "remove_duplicates": False,
            "rm_dup_inter": None,
            "iterations": kcore_iterations,
        },
        "split": {
            "name": "chronological leave-one-out",
            "train": "все interactions пользователя, кроме двух последних",
            "validation": "предпоследний interaction пользователя",
            "test": "последний interaction пользователя",
            "tie_breaking_rule": "сортировка по user_id, timestamp, source_row_id; source_row_id - нулевая позиция строки данных в исходном CSV",
            "random_seed": None,
            "max_item_list_length": args.max_seq_len,
        },
        "raw_stats": raw_stats,
        "filtered_stats": filtered_stats,
        "split_stats": split_stats,
        "sequence_stats": sequence_stats,
        "validation": validation,
        "storage": {
            "remote_path": str(output_dir),
            "manifest_path": str(repo_output_dir / "protocol_b_manifest.json"),
            "stats_path": str(repo_output_dir / "protocol_b_validation_summary.csv"),
            "report_path": str(report_path),
            "parquet_compression": None if args.no_parquet else "zstd",
        },
    }

    manifest_path = output_dir / "manifest.json"
    stats_json_path = output_dir / "stats.json"
    validation_csv_path = output_dir / "validation_summary.csv"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stats_json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_validation_csv(validation_csv_path, manifest)

    manifest["files"] = file_inventory(output_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stats_json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_validation_csv(validation_csv_path, manifest)

    report = build_report(manifest)
    report_path.write_text(report, encoding="utf-8")

    repo_manifest_path = repo_output_dir / "protocol_b_manifest.json"
    repo_stats_path = repo_output_dir / "protocol_b_validation_summary.csv"
    repo_stats_json_path = repo_output_dir / "protocol_b_stats.json"
    copy_if_different(manifest_path, repo_manifest_path)
    copy_if_different(validation_csv_path, repo_stats_path)
    copy_if_different(stats_json_path, repo_stats_json_path)

    print(f"Manifest: {manifest_path}")
    print(f"Копия manifest для Git: {repo_manifest_path}")
    print(f"Отчёт: {report_path}")
    print(
        "Совпадение с ожидаемым fingerprint: "
        f"{manifest['validation']['fingerprint_matches_expected']['all']}"
    )


if __name__ == "__main__":
    main()
