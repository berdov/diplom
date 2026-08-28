#!/usr/bin/env python
"""Build MOO benchmark report, raw summary CSV and optional registry rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments" / "moo_8families"
CONFIG = EXPERIMENT_DIR / "config.yaml"
REPORT = EXPERIMENT_DIR / "BENCHMARK_REPORT.md"
SUMMARY_CSV = EXPERIMENT_DIR / "runs" / "summary.csv"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
REGISTRY = ROOT / "experiments" / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--update-registry", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_runs_dir(config: dict[str, Any]) -> Path:
    path = Path(config["run"]["local_runs_dir"])
    return path if path.is_absolute() else ROOT / path


def run_paths(config: dict[str, Any]) -> list[Path]:
    paths = []
    runs_dir = local_runs_dir(config)
    for method, spec in config["methods"].items():
        if method == "pcgrad":
            paths.append(runs_dir / f"{spec['historical_run_id']}.json")
            continue
        for key in ("smoke_run_id", "sanity_run_id"):
            if key in spec:
                paths.append(runs_dir / f"{spec[key]}.json")
    return paths


def metric(record: dict[str, Any], key: str) -> float | None:
    if not record:
        return None
    return record.get("metrics", {}).get(key)


def aux_metric(record: dict[str, Any], target: str, key: str) -> float | None:
    aux = record.get("auxiliary_validation") or {}
    return None if target not in aux else aux[target].get(key)


def summary_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for path in run_paths(config):
        if not path.exists():
            continue
        payload = load_json(path)
        stage = payload.get("stage")
        if stage == "historical":
            method = payload["method"]
            best = payload["validation"]["best"]
        elif stage == "sanity":
            method = payload["method"]
            best = payload["validation"]["best"]
        else:
            method = payload.get("method", {})
            best = {}
        rows.append(
            {
                "run_id": payload["run_id"],
                "stage": stage,
                "status": payload.get("status"),
                "family": method.get("family"),
                "method": method.get("name"),
                "implementation": method.get("implementation_name"),
                "representative_fidelity": method.get("representative_fidelity"),
                "exact_method_reproduction": method.get("exact_method_reproduction"),
                "solution_type": method.get("solution_type"),
                "best_epoch": payload.get("best_epoch") or payload.get("historical", {}).get("best_epoch"),
                "actual_epochs": payload.get("training", {}).get("actual_epochs") or payload.get("historical", {}).get("actual_epochs"),
                "HR@5": metric(best, "HR@5"),
                "HR@10": metric(best, "HR@10"),
                "HR@20": metric(best, "HR@20"),
                "HR@50": metric(best, "HR@50"),
                "NDCG@5": metric(best, "NDCG@5"),
                "NDCG@10": metric(best, "NDCG@10"),
                "NDCG@20": metric(best, "NDCG@20"),
                "NDCG@50": metric(best, "NDCG@50"),
                "click_bce": aux_metric(best, "is_click", "bce_loss"),
                "long_bce": aux_metric(best, "long_view", "bce_loss"),
                "like_bce": aux_metric(best, "is_like", "bce_loss"),
                "profile_bce": aux_metric(best, "is_profile_enter", "bce_loss"),
                "wall_time_sec": payload.get("cost", {}).get("wall_time_sec") or payload.get("runtime", {}).get("total_sec"),
                "peak_vram_bytes": payload.get("cost", {}).get("peak_vram_bytes") or payload.get("gpu", {}).get("peak_allocated_bytes"),
                "model_count": payload.get("cost", {}).get("model_count"),
                "backward_passes_per_batch": payload.get("cost", {}).get("backward_passes_per_batch"),
                "test_evaluated": payload.get("test_safety", {}).get("test_evaluated"),
                "test_evaluation_count": payload.get("test_evaluation_count"),
                "git_commit": payload.get("git", {}).get("commit"),
                "source_json": str(path.relative_to(ROOT)),
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "stage",
        "status",
        "family",
        "method",
        "implementation",
        "representative_fidelity",
        "exact_method_reproduction",
        "solution_type",
        "best_epoch",
        "actual_epochs",
        "HR@5",
        "HR@10",
        "HR@20",
        "HR@50",
        "NDCG@5",
        "NDCG@10",
        "NDCG@20",
        "NDCG@50",
        "click_bce",
        "long_bce",
        "like_bce",
        "profile_bce",
        "wall_time_sec",
        "peak_vram_bytes",
        "model_count",
        "backward_passes_per_batch",
        "test_evaluated",
        "test_evaluation_count",
        "git_commit",
        "source_json",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metric(rows: list[dict[str, Any]], key: str, filename: str, title: str) -> None:
    selected = [row for row in rows if row.get("stage") in {"sanity", "historical"} and row.get(key) not in (None, "")]
    if not selected:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = [row["run_id"] for row in selected]
    values = [float(row[key]) for row in selected]
    plt.figure(figsize=(max(7, len(labels) * 1.0), 4))
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def write_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# MOO 8 Families Benchmark Report",
        "",
        "Отчёт сгенерирован из `experiments/moo_8families/runs/*.json`.",
        "",
        "## Ranking-Oriented Point",
        "",
        "| family | method | implementation | fidelity | exact | run | stage | HR@10 | NDCG@10 | best epoch | test eval |",
        "|---|---|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("stage") not in {"sanity", "historical"}:
            continue
        lines.append(
            f"| {row['family']} | {row['method']} | {row.get('implementation') or ''} | "
            f"{row.get('representative_fidelity') or ''} | {row.get('exact_method_reproduction')} | "
            f"`{row['run_id']}` | {row['stage']} | "
            f"{fmt(row['HR@10'])} | {fmt(row['NDCG@10'])} | {row['best_epoch'] or ''} | {row['test_evaluation_count']} |"
        )
    lines += [
        "",
        "## Compute Cost",
        "",
        "| run | wall sec | peak VRAM GB | model count | backward passes/batch |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("stage") not in {"sanity", "historical"}:
            continue
        peak = row.get("peak_vram_bytes")
        peak_gb = None if peak in (None, "") else float(peak) / 1024**3
        lines.append(
            f"| `{row['run_id']}` | {fmt(row['wall_time_sec'], 1)} | {fmt(peak_gb, 3)} | "
            f"{row.get('model_count') or ''} | {row.get('backward_passes_per_batch') or ''} |"
        )
    lines += [
        "",
        "## Raw Data",
        "",
        f"- Summary CSV: `experiments/moo_8families/runs/summary.csv`.",
        f"- NDCG plot: `experiments/moo_8families/figures/validation_ndcg10.png`.",
        f"- Cost plot: `experiments/moo_8families/figures/wall_time_sec.png`.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def update_registry(rows: list[dict[str, Any]]) -> None:
    if not REGISTRY.exists():
        raise RuntimeError(f"Missing registry: {REGISTRY}")
    with REGISTRY.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        existing = list(reader)
    existing_ids = {row["run_id"] for row in existing}
    additions = []
    for row in rows:
        if row.get("stage") not in {"sanity", "historical"} or row["run_id"] in existing_ids:
            continue
        additions.append(
            {
                "record_type": "sanity" if row["stage"] == "sanity" else "experiment_validation_only",
                "source": "ours",
                "run_id": row["run_id"],
                "model": "MultitaskTiM4Rec",
                "model_variant": f"moo_8families_{row['method']}",
                "dataset": "KuaiRand",
                "protocol": "B",
                "split": "validation",
                "evaluation": "full_7111_items",
                "status": row["status"],
                "parent_run": "multitask_tim4rec_tuned_001",
                "seed": "2026",
                "train_candidates": "full_sequence",
                "item_universe": "7111",
                "HR@5": row["HR@5"],
                "HR@10": row["HR@10"],
                "HR@20": row["HR@20"],
                "HR@50": row["HR@50"],
                "NDCG@5": row["NDCG@5"],
                "NDCG@10": row["NDCG@10"],
                "NDCG@20": row["NDCG@20"],
                "NDCG@50": row["NDCG@50"],
                "best_epoch": row["best_epoch"],
                "actual_epochs": row["actual_epochs"],
                "validation_ndcg10": row["NDCG@10"],
                "test_evaluation_count": row["test_evaluation_count"],
                "git_commit": row["git_commit"],
                "notes_path": row["source_json"].replace(".json", "_notes.md"),
                "test_used": "no",
                "source_json": row["source_json"],
            }
        )
    if not additions:
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        for addition in additions:
            writer.writerow({field: addition.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    rows = summary_rows(config)
    write_summary_csv(rows)
    plot_metric(rows, "NDCG@10", "validation_ndcg10.png", "Validation NDCG@10")
    plot_metric(rows, "wall_time_sec", "wall_time_sec.png", "Wall Time")
    if args.write_report:
        write_report(rows)
    if args.update_registry:
        update_registry(rows)
    print(json.dumps({"rows": len(rows), "summary_csv": str(SUMMARY_CSV)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
