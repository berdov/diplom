#!/usr/bin/env python3
"""Export compact summary for ltr_xgb_optuna_v1 without touching test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KS = (10, 20, 50)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str], default: str = "unknown") -> str:
    if args == ["rev-parse", "HEAD"] and os.environ.get("LTR_OPTUNA_GIT_COMMIT"):
        return str(os.environ["LTR_OPTUNA_GIT_COMMIT"])
    if args == ["rev-parse", "--abbrev-ref", "HEAD"] and os.environ.get("LTR_OPTUNA_GIT_BRANCH"):
        return str(os.environ["LTR_OPTUNA_GIT_BRANCH"])
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return default


def state_counts(study: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    for name in ["COMPLETE", "FAIL", "PRUNED", "RUNNING", "WAITING"]:
        counts.setdefault(name, 0)
    return counts


def parameter_importance(study: Any, optuna_module: Any) -> tuple[dict[str, Any], str]:
    try:
        return optuna_module.importance.get_param_importances(study), "fanova"
    except Exception as fanova_exc:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                evaluator = optuna_module.importance.PedAnovaImportanceEvaluator()
                importance = optuna_module.importance.get_param_importances(study, evaluator=evaluator)
            return importance, "ped_anova"
        except Exception as ped_exc:
            return {
                "_error": f"fanova failed: {fanova_exc}; ped_anova failed: {ped_exc}",
            }, "unavailable"


def sacct_jobs(job_ids: list[str]) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    fields = [
        "JobID",
        "JobName",
        "Partition",
        "State",
        "ExitCode",
        "Elapsed",
        "Timelimit",
        "NodeList",
        "AllocCPUS",
        "MaxRSS",
        "MaxVMSize",
    ]
    for job_id in job_ids:
        try:
            output = subprocess.check_output(
                ["sacct", "-j", job_id, f"--format={','.join(fields)}", "-P"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception as exc:
            jobs.append({"JobID": job_id, "_error": str(exc)})
            continue
        lines = [line for line in output.splitlines() if line.strip()]
        if len(lines) < 2:
            jobs.append({"JobID": job_id, "_error": "sacct returned no job rows"})
            continue
        header = lines[0].split("|")
        for line in lines[1:]:
            jobs.append(dict(zip(header, line.split("|"))))
    return jobs


def result_for_trial(trial: Any) -> dict[str, Any]:
    result = trial.user_attrs.get("result")
    if not isinstance(result, dict):
        result = {}
    metrics = dict(result.get("validation_metrics") or {})
    if not metrics and trial.value is not None:
        metrics["NDCG@10"] = float(trial.value)
    history = result.get("history") or []
    xgb_train_time = result.get("xgboost_train_time_sec")
    if xgb_train_time is None and history:
        xgb_train_time = sum(float(item.get("train_step_time_sec", 0.0)) for item in history)
    trial_runtime = result.get("trial_runtime_sec", result.get("train_time_sec"))
    return {
        "trial_number": int(trial.number),
        "state": trial.state.name,
        "value": None if trial.value is None else float(trial.value),
        "params": result.get("params") or dict(trial.params),
        "suggested_params": dict(trial.params),
        "validation_metrics": metrics,
        "validation_ndcg10": float(metrics.get("NDCG@10", trial.value if trial.value is not None else 0.0)),
        "validation_hr10": float(metrics.get("HR@10", trial.user_attrs.get("validation_hr10", 0.0))),
        "best_iteration": result.get("best_iteration", trial.user_attrs.get("best_iteration")),
        "best_num_boosted_rounds": result.get("best_num_boosted_rounds"),
        "num_boosted_rounds_trained": result.get("num_boosted_rounds_trained"),
        "number_of_full_ranking_evaluations": len(history),
        "stop_reason": result.get("stop_reason"),
        "xgboost_train_time_sec": None if xgb_train_time is None else float(xgb_train_time),
        "full_ranking_eval_time_sec": result.get("best_full_validation_time_sec"),
        "trial_runtime_sec": None if trial_runtime is None else float(trial_runtime),
        "seed": int(result.get("seed", (result.get("params") or {}).get("seed", 42))),
        "feature_set_version": result.get("feature_set_version"),
        "dataset_fingerprint": result.get("dataset_fingerprint"),
        "git": result.get("git"),
        "test_evaluation_count": int(trial.user_attrs.get("test_evaluation_count", result.get("test_evaluation_count", 0))),
    }


def metric_improvement(best: dict[str, float], baseline: dict[str, float]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in [*(f"NDCG@{k}" for k in KS), *(f"HR@{k}" for k in KS)]:
        ours = float(best[metric])
        base = float(baseline[metric])
        absolute = ours - base
        relative = None if base == 0.0 else absolute / base * 100.0
        result[metric] = {
            "best": ours,
            "baseline": base,
            "absolute_improvement": absolute,
            "relative_improvement_percent": relative,
        }
    return result


def compact_trial_row(rank: int, trial: dict[str, Any]) -> dict[str, Any]:
    metrics = trial["validation_metrics"]
    return {
        "rank": rank,
        "trial": trial["trial_number"],
        "NDCG@10": metrics.get("NDCG@10"),
        "HR@10": metrics.get("HR@10"),
        "NDCG@20": metrics.get("NDCG@20"),
        "HR@20": metrics.get("HR@20"),
        "NDCG@50": metrics.get("NDCG@50"),
        "HR@50": metrics.get("HR@50"),
        "best_iter": trial["best_iteration"],
        "params": trial["suggested_params"],
    }


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| rank | trial | NDCG@10 | HR@10 | NDCG@20 | HR@20 | NDCG@50 | HR@50 | best_iter | params |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        params = ", ".join(f"{key}={value}" for key, value in sorted(row["params"].items()))
        lines.append(
            f"| {row['rank']} | {row['trial']} | {row['NDCG@10']:.6f} | {row['HR@10']:.6f} | "
            f"{row['NDCG@20']:.6f} | {row['HR@20']:.6f} | {row['NDCG@50']:.6f} | "
            f"{row['HR@50']:.6f} | {row['best_iter']} | `{params}` |"
        )
    return lines


def build_notes(summary: dict[str, Any]) -> str:
    best = summary["best_trial"]
    improvement = summary["baseline_comparison"]["improvement"]
    slurm_jobs = summary.get("slurm_jobs") or []
    lines = [
        "# Optuna search 001",
        "",
        "## Study",
        "",
        f"- Study name: `{summary['study']['study_name']}`.",
        f"- Storage: `{summary['study']['storage_path']}`.",
        f"- COMPLETE / RUNNING / FAIL / PRUNED: `{summary['study']['state_counts']['COMPLETE']}` / "
        f"`{summary['study']['state_counts']['RUNNING']}` / `{summary['study']['state_counts']['FAIL']}` / "
        f"`{summary['study']['state_counts']['PRUNED']}`.",
        f"- Optuna: `{summary['environment']['optuna']}`.",
        f"- XGBoost: `{summary['environment']['xgboost']}`.",
        f"- Sampler: `{summary['study']['sampler']}`, seed `{summary['study']['sampler_seed']}`.",
    ]
    if slurm_jobs:
        lines.extend(
            [
                "",
                "## Slurm",
                "",
                "| JobID | JobName | Partition | State | ExitCode | Elapsed | Timelimit | NodeList | AllocCPUS | MaxRSS |",
                "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row in slurm_jobs:
            lines.append(
                f"| {row.get('JobID', '')} | {row.get('JobName', '')} | {row.get('Partition', '')} | "
                f"{row.get('State', '')} | {row.get('ExitCode', '')} | {row.get('Elapsed', '')} | "
                f"{row.get('Timelimit', '')} | {row.get('NodeList', '')} | {row.get('AllocCPUS', '')} | "
                f"{row.get('MaxRSS', '')} |"
            )
    lines.extend(
        [
            "",
            "## Best trial",
            "",
            f"- Trial: `{best['trial_number']}`.",
            f"- Best iteration: `{best['best_iteration']}`.",
            f"- Boosted rounds: `{best['best_num_boosted_rounds']}`.",
            f"- Validation NDCG@10: `{best['validation_metrics']['NDCG@10']:.6f}`.",
            f"- Validation HR@10: `{best['validation_metrics']['HR@10']:.6f}`.",
            f"- Validation NDCG@20: `{best['validation_metrics']['NDCG@20']:.6f}`.",
            f"- Validation HR@20: `{best['validation_metrics']['HR@20']:.6f}`.",
            f"- Validation NDCG@50: `{best['validation_metrics']['NDCG@50']:.6f}`.",
            f"- Validation HR@50: `{best['validation_metrics']['HR@50']:.6f}`.",
            "",
            "## Baseline improvement",
            "",
            "| metric | baseline | best | absolute | relative, % |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric, values in improvement.items():
        rel = values["relative_improvement_percent"]
        rel_text = "n/a" if rel is None else f"{rel:.2f}"
        lines.append(
            f"| {metric} | {values['baseline']:.6f} | {values['best']:.6f} | "
            f"{values['absolute_improvement']:.6f} | {rel_text} |"
        )
    lines.extend(["", "## Top 10", "", *markdown_table(summary["top_10_trials"])])
    if summary.get("parameter_importance"):
        lines.extend(["", "## Parameter importance", ""])
        if "_error" in summary["parameter_importance"]:
            lines.append(f"- Не удалось посчитать: `{summary['parameter_importance']['_error']}`.")
        else:
            lines.append(f"- Method: `{summary.get('parameter_importance_method', 'unknown')}`.")
            lines.append("")
            lines.extend(["| parameter | importance |", "| --- | ---: |"])
            for name, value in summary["parameter_importance"].items():
                lines.append(f"| {name} | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Test safety",
            "",
            f"- Test evaluation count: `{summary['test_safety']['test_evaluation_count']}`.",
            f"- Forbidden test paths loaded: `{summary['test_safety']['forbidden_test_paths_loaded']}`.",
            "- Test metrics отсутствуют в study summary.",
            "- `experiments/results.csv` не обновлялся.",
            "",
            "## Decision",
            "",
            "- Best trial выбран строго по full-ranking validation `NDCG@10`.",
            "- Final test не запускался.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/config.yaml"))
    parser.add_argument("--search-space", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/search_space.yaml"))
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/runs/optuna_search_001.json"))
    parser.add_argument("--output-notes", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/runs/optuna_search_001_notes.md"))
    parser.add_argument("--best-params", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/best_params.yaml"))
    parser.add_argument("--min-complete", type=int, default=40)
    parser.add_argument("--slurm-job-id", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    import optuna
    import xgboost as xgb

    args = parse_args()
    config = load_yaml(project_path(args.config))
    search_space = load_yaml(project_path(args.search_space))
    baseline = load_json(project_path(config["baseline"]["compact_run_json"]))
    storage_path = Path(config["study_storage"])
    storage = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=config["study_name"], storage=storage)

    complete_trials = [
        result_for_trial(trial)
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if len(complete_trials) < int(args.min_complete):
        raise SystemExit(f"Need at least {args.min_complete} COMPLETE trials, got {len(complete_trials)}")
    complete_trials.sort(key=lambda item: item["validation_ndcg10"], reverse=True)
    best = complete_trials[0]
    top_10 = [compact_trial_row(rank, trial) for rank, trial in enumerate(complete_trials[:10], start=1)]

    baseline_validation = baseline["evaluation_summary"]["metrics"]["validation"]["ltr_xgb"]
    improvement = metric_improvement(best["validation_metrics"], baseline_validation)
    counts = state_counts(study)
    total_runtime = sum(float(item.get("trial_runtime_sec") or 0.0) for item in complete_trials)
    full_eval_times = [
        float(item["full_ranking_eval_time_sec"])
        for item in complete_trials
        if item.get("full_ranking_eval_time_sec") is not None
    ]
    test_evaluation_count = sum(int(item.get("test_evaluation_count", 0)) for item in complete_trials)
    forbidden_test_paths: set[str] = set()
    for trial in study.trials:
        result = trial.user_attrs.get("result")
        if isinstance(result, dict):
            forbidden_test_paths.update(result.get("forbidden_test_paths_loaded", []))
            forbidden_test_paths.update(result.get("cache", {}).get("forbidden_test_paths_loaded", []))

    importance, importance_method = parameter_importance(study, optuna)

    db_info = {
        "path": str(storage_path),
        "size_bytes": storage_path.stat().st_size if storage_path.exists() else None,
        "sha256": sha256_file(storage_path) if storage_path.exists() else None,
    }
    summary = {
        "run_id": "optuna_search_001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "optuna": optuna.__version__,
            "xgboost": xgb.__version__,
        },
        "study": {
            "study_name": config["study_name"],
            "storage": storage,
            "storage_path": str(storage_path),
            "storage_db": db_info,
            "sampler": config["sampler"]["name"],
            "sampler_seed": int(config["sampler"]["seed"]),
            "state_counts": counts,
            "total_trials": len(study.trials),
            "complete_trials": len(complete_trials),
            "total_complete_trial_runtime_sec": total_runtime,
            "mean_best_full_validation_time_sec": sum(full_eval_times) / len(full_eval_times) if full_eval_times else None,
        },
        "search_space": search_space,
        "baseline": {
            "run_id": "ltr_xgb_002",
            "validation_metrics": baseline_validation,
            "feature_set_version": best.get("feature_set_version"),
        },
        "best_trial": best,
        "top_10_trials": top_10,
        "all_complete_trials": complete_trials,
        "baseline_comparison": {
            "metric_basis": "validation",
            "improvement": improvement,
        },
        "parameter_importance": importance,
        "parameter_importance_method": importance_method,
        "slurm_jobs": sacct_jobs(args.slurm_job_id),
        "test_safety": {
            "test_evaluation_count": int(test_evaluation_count),
            "forbidden_test_paths_loaded": sorted(forbidden_test_paths),
            "test_metrics_present": False,
            "results_csv_updated": False,
        },
        "decision": {
            "best_trial_selected_by": "max validation full-ranking NDCG@10",
            "final_test_ran": False,
            "ready_for_user_review_before_test": True,
        },
    }
    save_json(Path(args.output_json), summary)
    Path(args.output_notes).write_text(build_notes(summary), encoding="utf-8")
    best_params = {
        "study_name": config["study_name"],
        "storage_path": str(storage_path),
        "trial_number": best["trial_number"],
        "selection_metric": "validation_full_ranking_NDCG@10",
        "selection_value": best["validation_metrics"]["NDCG@10"],
        "validation_metrics": best["validation_metrics"],
        "best_iteration": best["best_iteration"],
        "best_num_boosted_rounds": best["best_num_boosted_rounds"],
        "suggested_params": best["suggested_params"],
        "xgboost_params": best["params"],
        "test_evaluation_count": 0,
    }
    save_yaml(Path(args.best_params), best_params)
    print(json.dumps({"output_json": args.output_json, "best_params": args.best_params, "best_trial": best["trial_number"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
