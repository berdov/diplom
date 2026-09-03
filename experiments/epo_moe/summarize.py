#!/usr/bin/env python
"""Build compact EPO + MoE summaries and report tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "epo_moe" / "configs" / "epo_moe.yaml"
SUMMARY = ROOT / "experiments" / "epo_moe" / "summary.json"
REPORT = ROOT / "reports" / "EPO_MOE_BENCHMARK.md"
RESULTS_CSV = ROOT / "experiments" / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--summary-json", default=str(SUMMARY))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--update-results-csv", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def format_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return fmt(value)
    return str(value)


def canonical_metric_key(key: str) -> str:
    metric, at, cutoff = str(key).partition("@")
    if not at:
        return str(key)
    normalized = {"hit": "HR", "hr": "HR", "recall": "Recall", "ndcg": "NDCG"}.get(metric.lower(), metric)
    return f"{normalized}@{cutoff}"


def canonical_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        canonical_metric_key(key): float(value)
        for key, value in metrics.items()
        if "@" in str(key) and value not in (None, "")
    }


def load_run(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    if payload.get("status") != "completed":
        return None
    return payload


def run_metrics(run: Mapping[str, Any] | None) -> dict[str, Any]:
    if not run:
        return {}
    validation = run.get("validation") or {}
    point = validation.get("ranking_operating_point") or validation.get("best") or {}
    return dict(point.get("metrics") or {})


def test_metrics(run: Mapping[str, Any] | None) -> dict[str, Any]:
    if not run:
        return {}
    return canonical_metrics(run.get("final_test_metrics") or {})


def model_params(run: Mapping[str, Any] | None) -> int | None:
    if not run:
        return None
    per_model = ((run.get("model_parameters") or {}).get("per_model") or [{}])[0]
    return None if not per_model else int(per_model["trainable"])


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def compact_run_ref(path: Path, run: Mapping[str, Any] | None) -> dict[str, Any]:
    if not run:
        return {"status": "missing", "source_json": relative_path(path)}
    metrics = test_metrics(run) if run.get("stage") == "final_test" else run_metrics(run)
    return {
        "status": run.get("status"),
        "record_type": run.get("record_type"),
        "run_id": run.get("run_id"),
        "run_key": run.get("run_key"),
        "stage": run.get("stage"),
        "source_json": relative_path(path),
        "git_commit": (run.get("git") or {}).get("commit"),
        "best_epoch": run.get("best_epoch") or ((run.get("validation_run") or {}).get("best_epoch")),
        "actual_epochs": (run.get("training") or {}).get("actual_epochs"),
        "architecture": run.get("architecture"),
        "metrics": metrics,
        "test_evaluation_count": run.get("test_evaluation_count", 0),
    }


def best_core_run(runs: Mapping[str, Mapping[str, Any] | None]) -> tuple[str | None, Mapping[str, Any] | None]:
    complete = [(key, run) for key, run in runs.items() if run and key != "m0"]
    if not complete:
        return None, None
    return max(complete, key=lambda item: float(run_metrics(item[1]).get("NDCG@10", -1.0)))


def paper_metrics(config: Mapping[str, Any]) -> dict[str, float]:
    return {
        "HR@10": 0.1109,
        "HR@20": 0.1774,
        "HR@50": 0.3202,
        "NDCG@10": 0.0611,
        "NDCG@20": 0.0779,
        "NDCG@50": 0.1060,
    }


def existing_test_run(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    candidates = [
        payload.get("final_test_metrics"),
        payload.get("test_metrics"),
        payload.get("metrics"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and "NDCG@10" in candidate:
            return canonical_metrics(candidate)
        if isinstance(candidate, dict) and "ndcg@10" in candidate:
            return canonical_metrics(candidate)
    raise KeyError(f"No final test metrics in {path}")


def table_rows(config: Mapping[str, Any], runs: Mapping[str, Mapping[str, Any] | None], final_tests: Mapping[str, Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    paper = paper_metrics(config)
    tim = existing_test_run(project_path(config["source"]["tim4rec_reproduction"]))
    epo = config["exact_epo_baseline"]["validation_metrics"]
    best_key, best_run = best_core_run(runs)
    best_test = final_tests.get(best_key or "") if best_key else None
    best_val = run_metrics(best_run)
    best_test_metrics = test_metrics(best_test)
    paper_ndcg = paper["NDCG@10"]

    return [
        {
            "Model": "TiM4Rec paper",
            "MoE": "no",
            "Experts": "",
            "Params": "",
            "Validation HR@10": "",
            "Validation NDCG@10": "",
            "Test HR@10": paper["HR@10"],
            "Test NDCG@10": paper["NDCG@10"],
            "Delta vs paper": 0.0,
            "Status": "published benchmark",
        },
        {
            "Model": "Our TiM4Rec reproduction",
            "MoE": "no",
            "Experts": "",
            "Params": "",
            "Validation HR@10": "",
            "Validation NDCG@10": "",
            "Test HR@10": tim.get("HR@10"),
            "Test NDCG@10": tim.get("NDCG@10"),
            "Delta vs paper": None if tim.get("NDCG@10") is None else tim["NDCG@10"] - paper_ndcg,
            "Status": "existing TEST reproduction",
        },
        {
            "Model": "Our TiM4Rec + multitask + EPO",
            "MoE": "no",
            "Experts": 0,
            "Params": config["exact_epo_baseline"]["trainable_params_per_solution"],
            "Validation HR@10": epo["HR@10"],
            "Validation NDCG@10": epo["NDCG@10"],
            "Test HR@10": "",
            "Test NDCG@10": "",
            "Delta vs paper": "",
            "Status": "validation-only baseline unless separately tested",
        },
        {
            "Model": "Our TiM4Rec + multitask + EPO + MoE",
            "MoE": "yes" if best_run else "",
            "Experts": "" if not best_run else best_run["architecture"]["num_experts"],
            "Params": model_params(best_run),
            "Validation HR@10": best_val.get("HR@10"),
            "Validation NDCG@10": best_val.get("NDCG@10"),
            "Test HR@10": best_test_metrics.get("HR@10"),
            "Test NDCG@10": best_test_metrics.get("NDCG@10"),
            "Delta vs paper": None if not best_test_metrics else best_test_metrics.get("NDCG@10") - paper_ndcg,
            "Status": "frozen final TEST" if best_test_metrics else "validation selection",
        },
    ]


def markdown_table(rows: list[Mapping[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            values.append(format_cell(row.get(column, "")))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def validation_rows(runs: Mapping[str, Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    rows = []
    for key, run in runs.items():
        if not run:
            continue
        metrics = run_metrics(run)
        rows.append(
            {
                "Run": key,
                "Run ID": run["run_id"],
                "Experts": run["architecture"]["num_experts"],
                "HR@10": metrics.get("HR@10"),
                "HR@20": metrics.get("HR@20"),
                "HR@50": metrics.get("HR@50"),
                "NDCG@10": metrics.get("NDCG@10"),
                "NDCG@20": metrics.get("NDCG@20"),
                "NDCG@50": metrics.get("NDCG@50"),
                "Best epoch": run.get("best_epoch"),
                "Actual epochs": run.get("training", {}).get("actual_epochs"),
                "Params": model_params(run),
                "Test evals": run.get("test_evaluation_count"),
            }
        )
    return rows


def routing_rows(runs: Mapping[str, Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    rows = []
    for key, run in runs.items():
        if not run or not run.get("routing"):
            continue
        routing = run["routing"]["summary"]
        if int(run["architecture"]["num_experts"]) == 0:
            rows.append({"Run": key, "Preference": "", "Task": "", "Dominant": "", "Share": "", "Entropy": "", "Collapse": "no_moe"})
            continue
        for model in routing["models"]:
            route = model["routing"]
            collapse = route["collapse"]
            for task, stats in route["per_task"].items():
                rows.append(
                    {
                        "Run": key,
                        "Preference": model["preference_id"],
                        "Task": task,
                        "Dominant": stats["dominant_expert"],
                        "Share": stats["dominant_share"],
                        "Entropy": stats["gate_entropy_mean"],
                        "Collapse": collapse["severe_collapse"],
                    }
                )
    return rows


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# EPO + MoE Benchmark",
        "",
        "KuaiRand Protocol B. Architecture selection uses validation only; TEST is used only after the frozen EPO+MoE configuration.",
        "",
        "## Main Table",
        "",
    ]
    lines += markdown_table(
        summary["main_table"],
        [
            "Model",
            "MoE",
            "Experts",
            "Params",
            "Validation HR@10",
            "Validation NDCG@10",
            "Test HR@10",
            "Test NDCG@10",
            "Delta vs paper",
            "Status",
        ],
    )
    lines += ["", "## Validation Runs", ""]
    lines += markdown_table(
        summary["validation_table"],
        ["Run", "Run ID", "Experts", "HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50", "Best epoch", "Actual epochs", "Params", "Test evals"],
    )
    lines += ["", "## Routing Diagnostics", ""]
    lines += markdown_table(summary["routing_table"], ["Run", "Preference", "Task", "Dominant", "Share", "Entropy", "Collapse"])
    lines += [
        "",
        "## Test Disclosure",
        "",
        f"TEST evaluations recorded for this EPO+MoE line: `{summary['test_evaluation_count']}`.",
        "No architecture, seed, learning-rate, dropout, task-set or EPO tuning is performed after TEST.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def update_results_csv(summary: Mapping[str, Any]) -> None:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(RESULTS_CSV)
    rows = list(csv.DictReader(RESULTS_CSV.open("r", encoding="utf-8", newline="")))
    fieldnames = list(rows[0].keys())
    new_rows = []
    for row in summary["registry_rows"]:
        registry = {key: "" for key in fieldnames}
        registry.update(row)
        new_rows.append(registry)
    existing_ids = {row["run_id"] for row in rows}
    rows = [row for row in rows if row["run_id"] not in {new["run_id"] for new in new_rows}]
    rows.extend(new_rows)
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    del existing_ids


def registry_rows(runs: Mapping[str, Mapping[str, Any] | None], final_tests: Mapping[str, Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    rows = []
    for key, run in runs.items():
        if not run:
            continue
        metrics = run_metrics(run)
        rows.append(
            {
                "record_type": "epo_moe_validation",
                "source": "ours",
                "run_id": run["run_id"],
                "model": "MultitaskTiM4Rec",
                "model_variant": f"EPO_MoE_{run['architecture']['num_experts']}_experts",
                "dataset": "KuaiRand",
                "protocol": "B",
                "split": "validation",
                "evaluation": "full_7111_items",
                "status": run["status"],
                "parent_run": "epo_tuning_001_trial_0000",
                "seed": str(run.get("training", {}).get("seed", 2026)),
                "train_candidates": "full_sequence",
                "item_universe": "7111",
                "HR@5": metrics.get("HR@5", ""),
                "HR@10": metrics.get("HR@10", ""),
                "HR@20": metrics.get("HR@20", ""),
                "HR@50": metrics.get("HR@50", ""),
                "Recall@5": metrics.get("Recall@5", ""),
                "Recall@10": metrics.get("Recall@10", ""),
                "Recall@20": metrics.get("Recall@20", ""),
                "Recall@50": metrics.get("Recall@50", ""),
                "NDCG@5": metrics.get("NDCG@5", ""),
                "NDCG@10": metrics.get("NDCG@10", ""),
                "NDCG@20": metrics.get("NDCG@20", ""),
                "NDCG@50": metrics.get("NDCG@50", ""),
                "best_epoch": run.get("best_epoch", ""),
                "actual_epochs": run.get("training", {}).get("actual_epochs", ""),
                "validation_ndcg10": metrics.get("NDCG@10", ""),
                "test_evaluation_count": run.get("test_evaluation_count", 0),
                "git_commit": run.get("git", {}).get("commit", ""),
                "test_used": "no",
                "source_json": f"experiments/epo_moe/runs/{run['run_id']}.json",
            }
        )
    for key, run in final_tests.items():
        if not run:
            continue
        metrics = test_metrics(run)
        rows.append(
            {
                "record_type": "epo_moe_final_test",
                "source": "ours",
                "run_id": run["run_id"],
                "model": "MultitaskTiM4Rec",
                "model_variant": f"EPO_MoE_{run['architecture']['num_experts']}_experts",
                "dataset": "KuaiRand",
                "protocol": "B",
                "split": "test",
                "evaluation": "full_7111_items",
                "status": run["status"],
                "parent_run": run["validation_run"]["run_id"],
                "seed": "2026",
                "train_candidates": "full_sequence",
                "item_universe": "7111",
                "HR@5": metrics.get("HR@5", ""),
                "HR@10": metrics.get("HR@10", ""),
                "HR@20": metrics.get("HR@20", ""),
                "HR@50": metrics.get("HR@50", ""),
                "Recall@5": metrics.get("Recall@5", ""),
                "Recall@10": metrics.get("Recall@10", ""),
                "Recall@20": metrics.get("Recall@20", ""),
                "Recall@50": metrics.get("Recall@50", ""),
                "NDCG@5": metrics.get("NDCG@5", ""),
                "NDCG@10": metrics.get("NDCG@10", ""),
                "NDCG@20": metrics.get("NDCG@20", ""),
                "NDCG@50": metrics.get("NDCG@50", ""),
                "best_epoch": run["validation_run"]["best_epoch"],
                "validation_ndcg10": run["validation_run"]["validation"]["ranking_operating_point"]["metrics"]["NDCG@10"],
                "test_evaluation_count": run.get("test_evaluation_count", 0),
                "git_commit": run.get("git", {}).get("commit", ""),
                "test_used": "yes",
                "source_json": f"experiments/epo_moe/runs/{run['run_id']}.json",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    runs_dir = project_path(config["outputs"]["runs_dir"])
    run_paths = {key: runs_dir / f"{run_cfg['run_id']}.json" for key, run_cfg in config["runs"].items()}
    final_test_paths = {
        key: runs_dir / f"{run_cfg['run_id']}_final_test.json" for key, run_cfg in config["runs"].items()
    }
    runs = {key: load_run(path) for key, path in run_paths.items()}
    final_tests = {key: load_run(path) for key, path in final_test_paths.items()}
    best_key, best_run = best_core_run(runs)
    summary = {
        "record_type": "epo_moe_summary",
        "config": str(Path(args.config)),
        "runs": {key: compact_run_ref(run_paths[key], runs[key]) for key in run_paths},
        "final_tests": {key: compact_run_ref(final_test_paths[key], final_tests[key]) for key in final_test_paths},
        "best_validation_run_key": best_key,
        "best_validation_run_id": None if not best_run else best_run["run_id"],
        "best_validation_NDCG@10": None if not best_run else run_metrics(best_run).get("NDCG@10"),
        "main_table": table_rows(config, runs, final_tests),
        "validation_table": validation_rows(runs),
        "routing_table": routing_rows(runs),
        "test_evaluation_count": sum(int((run or {}).get("test_evaluation_count") or 0) for run in final_tests.values()),
    }
    summary["registry_rows"] = registry_rows(runs, final_tests)
    save_json(Path(args.summary_json), summary)
    write_report(Path(args.report), summary)
    if args.update_results_csv:
        update_results_csv(summary)
    print(json.dumps({"summary": args.summary_json, "report": args.report, "best_key": best_key}, indent=2), flush=True)


if __name__ == "__main__":
    main()
