#!/usr/bin/env python3
"""Locked final test run for the Optuna-selected XGBoost LambdaMART model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "experiments" / "ltr_xgb_baseline"
OPTUNA_DIR = Path(__file__).resolve().parent
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
if str(OPTUNA_DIR) not in sys.path:
    sys.path.insert(0, str(OPTUNA_DIR))

import run_experiment as base  # noqa: E402
import optuna_search  # noqa: E402

RUN_ID = "ltr_xgb_optuna_001"
MODEL_KEY = "ltr_xgb_optuna"
MODEL_LABEL = "XGBoost LambdaMART tuned"
EXPECTED_FEATURE_HASH = "8abcf619f10433225767952859d305ec507d2b00eaca5b2118a79d7f29730a25"
KS = (5, 10, 20, 50)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str], default: str = "unknown") -> str:
    if args == ["rev-parse", "HEAD"]:
        env_commit = os.environ.get("LTR_GIT_COMMIT") or os.environ.get("LTR_OPTUNA_GIT_COMMIT")
        if env_commit:
            return env_commit
    if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
        env_branch = os.environ.get("LTR_GIT_BRANCH") or os.environ.get("LTR_OPTUNA_GIT_BRANCH")
        if env_branch:
            return env_branch
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return default


def slurm_env() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "hostname": socket.gethostname(),
    }


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


def environment_info() -> dict[str, Any]:
    import xgboost as xgb

    packages: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "xgboost": xgb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }
    try:
        import optuna

        packages["optuna"] = optuna.__version__
    except Exception:
        packages["optuna"] = None
    return packages


def ensure_result_absent(results_csv: Path) -> None:
    if not results_csv.exists():
        return
    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("run_id") == RUN_ID:
                raise SystemExit(f"{RUN_ID} already exists in {results_csv}; refusing to repeat locked test")


def metric_names(include_recall: bool = True) -> list[str]:
    names = [*(f"HR@{k}" for k in KS), *(f"NDCG@{k}" for k in KS)]
    if include_recall:
        names.extend(f"Recall@{k}" for k in KS)
    return names


def compact_metrics(metrics: dict[str, Any], include_recall: bool = True) -> dict[str, float]:
    return {name: float(metrics[name]) for name in metric_names(include_recall) if name in metrics}


def assert_hr_recall_equal(metrics: dict[str, Any]) -> None:
    for k in KS:
        if abs(float(metrics[f"HR@{k}"]) - float(metrics[f"Recall@{k}"])) >= 1e-12:
            raise AssertionError(f"HR@{k} != Recall@{k}")


def comparison(best: dict[str, Any], baseline: dict[str, Any], metrics: list[str]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        tuned = float(best[metric])
        base_value = float(baseline[metric])
        absolute = tuned - base_value
        relative = None if base_value == 0.0 else absolute / base_value * 100.0
        ratio = None if base_value == 0.0 else tuned / base_value
        result[metric] = {
            "baseline": base_value,
            "tuned": tuned,
            "absolute_improvement": absolute,
            "relative_improvement_percent": relative,
            "ratio": ratio,
        }
    return result


def rows_by_run_id(results_csv: Path) -> dict[str, dict[str, str]]:
    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        return {row["run_id"]: row for row in csv.DictReader(fh) if row.get("run_id")}


def row_metrics(row: dict[str, str], include_recall: bool = True) -> dict[str, float]:
    return {name: float(row[name]) for name in metric_names(include_recall) if row.get(name) not in {None, ""}}


def build_comparison_tables(results_csv: Path, tuned_test: dict[str, float]) -> dict[str, Any]:
    rows = rows_by_run_id(results_csv)
    required = ["random_002", "mostpop_002", "ltr_xgb_002", "tim4rec_001", "ssd4rec_001"]
    missing = [run_id for run_id in required if run_id not in rows]
    if missing:
        raise AssertionError(f"Missing comparable result rows: {missing}")

    labels = {
        "random_002": "Random",
        "mostpop_002": "MostPopular",
        "ltr_xgb_002": "XGBoost LambdaMART ltr_xgb_002",
        RUN_ID: "XGBoost LambdaMART tuned ltr_xgb_optuna_001",
        "tim4rec_001": "TiM4Rec tim4rec_001",
        "ssd4rec_001": "SSD4Rec ssd4rec_001",
    }
    row_sources: dict[str, dict[str, float]] = {
        "random_002": row_metrics(rows["random_002"], include_recall=False),
        "mostpop_002": row_metrics(rows["mostpop_002"], include_recall=False),
        "ltr_xgb_002": row_metrics(rows["ltr_xgb_002"], include_recall=False),
        RUN_ID: compact_metrics(tuned_test, include_recall=False),
        "tim4rec_001": row_metrics(rows["tim4rec_001"], include_recall=False),
        "ssd4rec_001": row_metrics(rows["ssd4rec_001"], include_recall=False),
    }
    comparable = [{"run_id": run_id, "model": labels[run_id], **row_sources[run_id]} for run_id in row_sources]
    return {
        "test_table": comparable,
        "vs_ltr_xgb_002": comparison(
            tuned_test,
            row_metrics(rows["ltr_xgb_002"]),
            [*(f"HR@{k}" for k in KS), *(f"NDCG@{k}" for k in KS)],
        ),
        "vs_mostpopular": comparison(
            tuned_test,
            row_metrics(rows["mostpop_002"]),
            [*(f"HR@{k}" for k in KS), *(f"NDCG@{k}" for k in KS)],
        ),
        "vs_tim4rec": comparison(tuned_test, row_metrics(rows["tim4rec_001"]), ["HR@10", "NDCG@10"]),
        "vs_ssd4rec": comparison(tuned_test, row_metrics(rows["ssd4rec_001"]), ["HR@10", "NDCG@10"]),
    }


def append_result_row(
    results_csv: Path,
    config: dict[str, Any],
    artifact_root: Path,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    train_time_sec: float,
    inference_time_sec: float,
) -> None:
    ensure_result_absent(results_csv)
    row: dict[str, Any] = {
        "run_id": RUN_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_LABEL,
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "protocol": "B",
        "candidate_protocol": "protocol_b_1pos_100neg_seed42_train",
        "evaluation_protocol": "full_7111_items",
        "train_candidate_protocol": "sampled_100",
        "eval_candidate_protocol": "full_7111_items",
        "item_universe_size": 7111,
        "mask_seen_items": False,
        "protocol_version": "recbole_sequential_full_v1",
        "n_negatives": config["candidate_protocol"]["n_negatives"],
        "seed": config["model"]["model_seed"],
        "features": "|".join(load_yaml(project_path(config["baseline"]["config_path"]))["features"]["names"]),
        "train_time_sec": f"{train_time_sec:.6f}",
        "inference_time_sec": f"{inference_time_sec:.6f}",
        "remote_artifact_path": str(artifact_root),
        "notes": "source=Optuna trial 16; status=completed; locked_params=best_params.yaml; validation_reproduction_passed; test_evaluation_count=1",
    }
    for k in KS:
        for prefix in ["HR", "NDCG", "Recall"]:
            row[f"{prefix}@{k}"] = f"{float(test_metrics[f'{prefix}@{k}']):.12f}"
            row[f"validation_{prefix}@{k}"] = f"{float(validation_metrics[f'{prefix}@{k}']):.12f}"

    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or base.RESULT_COLUMNS)
    missing_columns = [column for column in base.RESULT_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise AssertionError(f"results.csv is missing columns: {missing_columns}")
    with results_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writerow({column: row.get(column, "") for column in fieldnames})


def verify_best_trial(config: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    import optuna

    study = optuna.load_study(study_name=config["study_name"], storage=f"sqlite:///{Path(config['study_storage'])}")
    trial = study.trials[int(best["trial_number"])]
    if study.study_name != best["study_name"]:
        raise AssertionError("best_params study_name does not match Optuna DB")
    if trial.number != 16:
        raise AssertionError("locked final run must use Optuna trial 16")
    if trial.state.name != "COMPLETE":
        raise AssertionError(f"trial 16 is not COMPLETE: {trial.state.name}")
    result = trial.user_attrs.get("result") or {}
    if result.get("objective") != "validation_full_ranking_NDCG@10":
        raise AssertionError("trial 16 objective changed")
    if abs(float(trial.value) - float(best["selection_value"])) >= 1e-15:
        raise AssertionError("trial 16 value differs from best_params selection_value")
    if int(result.get("best_iteration")) != int(best["best_iteration"]):
        raise AssertionError("best_iteration differs between DB and best_params")
    if int(result.get("best_num_boosted_rounds")) != int(best["best_num_boosted_rounds"]):
        raise AssertionError("best_num_boosted_rounds differs between DB and best_params")
    for key, value in best["suggested_params"].items():
        db_value = trial.params[key]
        if isinstance(value, float):
            if not math.isclose(float(db_value), float(value), rel_tol=0.0, abs_tol=1e-15):
                raise AssertionError(f"suggested param mismatch: {key}")
        elif db_value != value:
            raise AssertionError(f"suggested param mismatch: {key}")
    for key, value in best["xgboost_params"].items():
        db_value = result["params"].get(key)
        if isinstance(value, float):
            if not math.isclose(float(db_value), float(value), rel_tol=0.0, abs_tol=1e-15):
                raise AssertionError(f"xgboost param mismatch: {key}")
        elif db_value != value:
            raise AssertionError(f"xgboost param mismatch: {key}")

    counts: dict[str, int] = {}
    for candidate in study.trials:
        counts[candidate.state.name] = counts.get(candidate.state.name, 0) + 1
    return {
        "study_name": study.study_name,
        "storage_path": str(config["study_storage"]),
        "state_counts": counts,
        "failed_trials": [candidate.number for candidate in study.trials if candidate.state.name == "FAIL"],
        "running_trials": [candidate.number for candidate in study.trials if candidate.state.name == "RUNNING"],
        "trial_number": trial.number,
        "trial_state": trial.state.name,
        "trial_value": float(trial.value),
        "objective": result.get("objective"),
        "params_verified_against_best_params_yaml": True,
        "complete_trials_unchanged": int(counts.get("COMPLETE", 0)),
    }


def prepare_artifact_root(config: dict[str, Any]) -> Path:
    root = Path(config["remote_artifact_dir"]) / RUN_ID
    summary_path = root / "metrics" / "final_summary.json"
    if summary_path.exists():
        existing = load_json(summary_path)
        if int(existing.get("test_evaluation_count", 0)) > 0:
            raise SystemExit(f"{summary_path} already contains a test result; refusing to repeat locked test")
    for name in ["model", "rankings", "metrics", "logs", "config"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def train_locked_model(cached: optuna_search.CachedInputs, best: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    import xgboost as xgb

    params = dict(best["xgboost_params"])
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        params["nthread"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    boosted_rounds = int(best["best_num_boosted_rounds"])
    start = time.perf_counter()
    booster = xgb.train(params=params, dtrain=cached.dtrain, num_boost_round=boosted_rounds, evals=[], verbose_eval=False)
    train_time = time.perf_counter() - start
    model_path = artifact_root / "model" / "xgb_lambdamart_tuned.json"
    booster.save_model(model_path)
    return {
        "booster": booster,
        "summary": {
            "xgboost_version": xgb.__version__,
            "params": params,
            "training_decision": "fixed_boosted_rounds_from_locked_optuna_trial",
            "early_stopping_rerun": False,
            "num_boost_round": boosted_rounds,
            "best_iteration_from_optuna": int(best["best_iteration"]),
            "num_boosted_rounds": int(booster.num_boosted_rounds()),
            "train_time_sec": float(train_time),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "feature_importance_gain": booster.get_score(importance_type="gain"),
        },
    }


def evaluate_split_full(
    booster: Any,
    split: str,
    queries: pd.DataFrame,
    cached: optuna_search.CachedInputs,
    config: dict[str, Any],
    artifact_root: Path,
    num_boosted_rounds: int,
) -> dict[str, Any]:
    import xgboost as xgb

    queries = queries.sort_values("query_index").reset_index(drop=True)
    eval_config = config["evaluation"]
    batch_users = int(eval_config["batch_users"])
    topk = int(eval_config["topk"])
    mask_seen_items = bool(eval_config["mask_seen_items"])
    item_ids = cached.item_ids
    item_to_pos = cached.item_to_pos
    if not set(queries["target_item_id"].astype("int64")).issubset(item_to_pos):
        raise AssertionError(f"{split}: target item missing from item universe")

    start = time.perf_counter()
    ranked_batches: list[pd.DataFrame] = []
    rank_batches: list[np.ndarray] = []
    candidate_counts: list[np.ndarray] = []
    target_evaluable_batches: list[np.ndarray] = []
    for start_row in range(0, int(queries.shape[0]), batch_users):
        batch = queries.iloc[start_row : start_row + batch_users].copy()
        batch_size = int(batch.shape[0])
        matrix = base.full_feature_batch(batch, cached.item_features, item_ids, item_to_pos, cached.feature_names)
        dmat = xgb.DMatrix(matrix, feature_names=cached.feature_names)
        predicted = booster.predict(dmat, iteration_range=(0, int(num_boosted_rounds)))
        scores = predicted.reshape(batch_size, item_ids.shape[0]).astype(np.float32, copy=False)
        scores, counts = base.apply_full_masks(scores, batch, item_to_pos, mask_seen_items)
        target_items = batch["target_item_id"].to_numpy(dtype=np.int64)
        target_positions = np.array([item_to_pos[int(item)] for item in target_items], dtype=np.int64)
        target_scores = scores[np.arange(batch_size), target_positions]
        target_evaluable_batches.append(np.isfinite(target_scores))
        candidate_counts.append(counts)
        finite_scores = np.where(np.isfinite(scores), scores, np.finfo(np.float32).min)
        ranked_batch, target_ranks = base.scores_to_topk_frame(batch, finite_scores, item_ids, topk)
        ranked_batches.append(ranked_batch)
        rank_batches.append(target_ranks)

    elapsed = time.perf_counter() - start
    target_ranks = np.concatenate(rank_batches)
    metrics = base.metrics_from_target_ranks(target_ranks)
    assert_hr_recall_equal(metrics)
    ranked = pd.concat(ranked_batches, ignore_index=True)
    ranking_path = artifact_root / "rankings" / f"{MODEL_KEY}_{split}_top{topk}.parquet"
    ranked.to_parquet(ranking_path, index=False)
    split_stats = base.validate_full_eval_split(
        queries,
        item_to_pos,
        candidate_counts,
        target_evaluable_batches,
        mask_seen_items,
    )
    result = {
        "split": split,
        "metrics": metrics,
        "eval_time_sec": float(elapsed),
        "users": int(queries.shape[0]),
        "items": int(item_ids.shape[0]),
        "scores": int(queries.shape[0] * item_ids.shape[0]),
        "batch_users": int(batch_users),
        "topk": int(topk),
        "mask_seen_items": bool(mask_seen_items),
        "ranking_path": str(ranking_path),
        "ranking_sha256": sha256_file(ranking_path),
        "rows_saved": int(ranked.shape[0]),
        "split_stats": split_stats,
    }
    save_json(artifact_root / "metrics" / f"{split}_evaluation_summary.json", result)
    return result


def validation_diff(reproduced: dict[str, float], expected: dict[str, float]) -> dict[str, float]:
    return {name: float(reproduced[name]) - float(expected[name]) for name in metric_names() if name in reproduced}


def validate_reproduction(reproduced: dict[str, float], expected: dict[str, float], tolerance: float) -> dict[str, Any]:
    diffs = validation_diff(reproduced, expected)
    max_abs = max(abs(value) for value in diffs.values())
    return {
        "passed": bool(max_abs < tolerance),
        "tolerance": float(tolerance),
        "max_abs_diff": float(max_abs),
        "diffs": diffs,
    }


def markdown_metric_table(rows: list[dict[str, Any]], metrics: list[str]) -> list[str]:
    lines = [
        "| model | " + " | ".join(metrics) + " |",
        "| --- | " + " | ".join(["---:"] * len(metrics)) + " |",
    ]
    for row in rows:
        values = [str(row["model"])]
        for metric in metrics:
            value = row.get(metric)
            values.append("" if value is None else f"{float(value):.6f}")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def markdown_comparison_table(comp: dict[str, dict[str, float | None]]) -> list[str]:
    lines = [
        "| metric | baseline | tuned | absolute | relative, % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, values in comp.items():
        rel = values["relative_improvement_percent"]
        rel_text = "" if rel is None else f"{rel:.2f}"
        lines.append(
            f"| {metric} | {values['baseline']:.6f} | {values['tuned']:.6f} | "
            f"{values['absolute_improvement']:.6f} | {rel_text} |"
        )
    return lines


def build_notes(summary: dict[str, Any]) -> str:
    best = summary["source_trial"]
    validation = summary["validation_reproduction"]
    final_test = summary["final_test_metrics"]
    comp = summary["comparisons"]
    params = summary["best_params"]["xgboost_params"]
    beats_mostpop_hr10 = comp["vs_mostpopular"]["HR@10"]["absolute_improvement"] > 0
    beats_mostpop_ndcg10 = comp["vs_mostpopular"]["NDCG@10"]["absolute_improvement"] > 0
    mostpop_conclusion = (
        "Tuned XGBoost превосходит MostPopular по HR@10 и NDCG@10."
        if beats_mostpop_hr10 and beats_mostpop_ndcg10
        else "Tuned XGBoost не превосходит MostPopular одновременно по HR@10 и NDCG@10."
    )
    lines = [
        "# XGBoost LambdaMART после Optuna",
        "",
        "## Цель",
        "",
        "Провести один locked-test эксперимент для параметров, выбранных Optuna по full-ranking validation `NDCG@10`.",
        "",
        "## Исходный baseline",
        "",
        "- Baseline: `ltr_xgb_002`.",
        "- Train candidates: `sampled_100`, 1 positive + 100 negatives.",
        "- Validation/test: full-ranking по 7111 items.",
        "",
        "## Optuna search",
        "",
        f"- Study: `{summary['source_study']['study_name']}`.",
        f"- COMPLETE / FAIL / PRUNED / RUNNING: `{summary['source_study']['state_counts'].get('COMPLETE', 0)}` / "
        f"`{summary['source_study']['state_counts'].get('FAIL', 0)}` / "
        f"`{summary['source_study']['state_counts'].get('PRUNED', 0)}` / "
        f"`{summary['source_study']['state_counts'].get('RUNNING', 0)}`.",
        "- Objective: full-ranking validation `NDCG@10`.",
        f"- Best trial: `{best['trial_number']}`.",
        f"- Stale trials: FAIL `{summary['source_study'].get('failed_trials', [])}`, RUNNING `{summary['source_study'].get('running_trials', [])}`.",
        "",
        "## Зафиксированные параметры",
        "",
        "| parameter | value |",
        "| --- | ---: |",
    ]
    for key, value in params.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            f"- Best iteration из Optuna: `{summary['boosted_rounds']['best_iteration']}`.",
            f"- Обучение final model выполнено с фиксированными boosted rounds: `{summary['boosted_rounds']['best_num_boosted_rounds']}`.",
            "- Новый early stopping не запускался.",
            "",
            "## Повторная проверка validation",
            "",
            "| metric | Optuna | reproduced | diff |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in metric_names():
        lines.append(
            f"| {metric} | {summary['validation_optuna_metrics'][metric]:.12f} | "
            f"{validation['metrics'][metric]:.12f} | {validation['diffs'][metric]:.3e} |"
        )
    lines.extend(
        [
            "",
            f"- Tolerance: `{validation['tolerance']}`.",
            f"- Max abs diff: `{validation['max_abs_diff']:.3e}`.",
            f"- Passed: `{validation['passed']}`.",
            "",
            "## Final test",
            "",
            *markdown_metric_table([{"model": RUN_ID, **final_test}], metric_names()),
            "",
            "## Эффект tuning",
            "",
            *markdown_comparison_table(comp["vs_ltr_xgb_002"]),
            "",
            "## Сравнение с MostPopular",
            "",
            *markdown_comparison_table(comp["vs_mostpopular"]),
            "",
            "## Сравнение с TiM4Rec и SSD4Rec",
            "",
            *markdown_metric_table(comp["test_table"], [*(f"HR@{k}" for k in KS), *(f"NDCG@{k}" for k in KS)]),
            "",
            f"- Tuned XGB / TiM4Rec HR@10: `{comp['vs_tim4rec']['HR@10']['ratio']:.4f}`.",
            f"- Tuned XGB / TiM4Rec NDCG@10: `{comp['vs_tim4rec']['NDCG@10']['ratio']:.4f}`.",
            f"- Tuned XGB / SSD4Rec HR@10: `{comp['vs_ssd4rec']['HR@10']['ratio']:.4f}`.",
            f"- Tuned XGB / SSD4Rec NDCG@10: `{comp['vs_ssd4rec']['NDCG@10']['ratio']:.4f}`.",
            "",
            "## Вывод",
            "",
            f"- Tuning улучшил XGBoost по test NDCG@10 на `{comp['vs_ltr_xgb_002']['NDCG@10']['relative_improvement_percent']:.2f}%` "
            f"и HR@10 на `{comp['vs_ltr_xgb_002']['HR@10']['relative_improvement_percent']:.2f}%`.",
            f"- {mostpop_conclusion}",
            "- До TiM4Rec и SSD4Rec остается большой разрыв; последовательные модели используют структуру истории существенно лучше табличного LTR на этих признаках.",
            "- Test был открыт один раз; после test параметры не менялись.",
        ]
    )
    return "\n".join(lines) + "\n"


def refresh_slurm_artifacts(output_json: Path, output_notes: Path, job_ids: list[str]) -> None:
    summary = load_json(output_json)
    summary.setdefault("slurm", {})["sacct"] = sacct_jobs(job_ids)
    save_json(output_json, summary)
    output_notes.write_text(build_notes(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/config.yaml"))
    parser.add_argument("--best-params", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/best_params.yaml"))
    parser.add_argument("--results-csv", default=str(PROJECT_ROOT / "experiments/results.csv"))
    parser.add_argument("--allow-test-evaluation", action="store_true")
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--validation-tolerance", type=float, default=1e-8)
    parser.add_argument("--refresh-slurm", action="store_true")
    parser.add_argument("--slurm-job-id", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(project_path(args.config))
    output_json = project_path(config["compact_runs_dir"]) / f"{RUN_ID}.json"
    output_notes = project_path(config["compact_runs_dir"]) / f"{RUN_ID}_notes.md"
    if args.refresh_slurm:
        refresh_slurm_artifacts(output_json, output_notes, args.slurm_job_id)
        print(json.dumps({"refreshed": str(output_json), "slurm_job_ids": args.slurm_job_id}, ensure_ascii=False))
        return
    if not args.allow_test_evaluation:
        raise SystemExit("Locked test is disabled; pass --allow-test-evaluation for the single final run.")

    results_csv = project_path(args.results_csv)
    ensure_result_absent(results_csv)
    best = load_yaml(project_path(args.best_params))
    source_study = verify_best_trial(config, best)
    baseline_config = load_yaml(project_path(config["baseline"]["config_path"]))
    cached = optuna_search.load_cached_inputs(config, baseline_config)
    if cached.cache_summary["feature_set_version"] != EXPECTED_FEATURE_HASH:
        raise AssertionError("Feature set hash changed")
    if len(cached.feature_names) != 17:
        raise AssertionError(f"Expected 17 features, got {len(cached.feature_names)}")
    expected_fingerprint = {
        "users": int(config["protocol"]["users"]),
        "items": int(config["protocol"]["items"]),
        "interactions": int(config["protocol"]["interactions"]),
        "train": int(config["protocol"]["train_rows"]),
        "validation": int(config["protocol"]["validation_rows"]),
        "test": int(config["protocol"]["test_rows"]),
    }
    if cached.cache_summary["dataset_fingerprint"] != expected_fingerprint:
        raise AssertionError("Protocol B dataset fingerprint changed")

    artifact_root = prepare_artifact_root(config)
    save_yaml(artifact_root / "config" / "config_snapshot.yaml", config)
    save_yaml(artifact_root / "config" / "baseline_config_snapshot.yaml", baseline_config)
    save_yaml(artifact_root / "config" / "best_params_snapshot.yaml", best)
    shutil.copy2(project_path(config["baseline"]["compact_run_json"]), artifact_root / "config" / "ltr_xgb_002.json")

    start_total = time.perf_counter()
    train_result = train_locked_model(cached, best, artifact_root)
    booster = train_result["booster"]
    train_summary = train_result["summary"]
    save_json(artifact_root / "metrics" / "train_summary.json", train_summary)

    validation_eval = evaluate_split_full(
        booster,
        "validation",
        cached.validation_queries,
        cached,
        config,
        artifact_root,
        int(best["best_num_boosted_rounds"]),
    )
    validation_metrics = compact_metrics(validation_eval["metrics"])
    reproduction = validate_reproduction(validation_metrics, compact_metrics(best["validation_metrics"]), float(args.validation_tolerance))
    if not reproduction["passed"]:
        failed = {
            "run_id": RUN_ID,
            "status": "failed_validation_reproduction",
            "source_study": source_study,
            "source_trial": {"trial_number": int(best["trial_number"])},
            "validation_optuna_metrics": compact_metrics(best["validation_metrics"]),
            "validation_reproduced_metrics": validation_metrics,
            "validation_reproduction": reproduction,
            "test_evaluation_count": 0,
            "remote_artifact_path": str(artifact_root),
        }
        save_json(output_json, failed)
        raise SystemExit("Validation reproduction failed; test was not evaluated")

    test_queries_path = Path(config["baseline"]["artifact_dir"]) / "candidates" / "test_queries.parquet"
    test_queries = pd.read_parquet(test_queries_path)
    test_eval = evaluate_split_full(
        booster,
        "test",
        test_queries,
        cached,
        config,
        artifact_root,
        int(best["best_num_boosted_rounds"]),
    )
    test_evaluation_count = 1
    final_test_metrics = compact_metrics(test_eval["metrics"])
    if args.write_results:
        append_result_row(
            results_csv,
            config,
            artifact_root,
            validation_metrics,
            final_test_metrics,
            float(train_summary["train_time_sec"]),
            float(validation_eval["eval_time_sec"] + test_eval["eval_time_sec"]),
        )
    comparisons = build_comparison_tables(results_csv, final_test_metrics)
    runtime = {
        "total_runtime_sec": float(time.perf_counter() - start_total),
        "train_time_sec": float(train_summary["train_time_sec"]),
        "validation_eval_time_sec": float(validation_eval["eval_time_sec"]),
        "test_eval_time_sec": float(test_eval["eval_time_sec"]),
    }
    summary = {
        "run_id": RUN_ID,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
        },
        "environment": environment_info(),
        "slurm": {"environment": slurm_env(), "sacct": sacct_jobs(args.slurm_job_id)},
        "source_study": source_study,
        "source_trial": {
            "trial_number": int(best["trial_number"]),
            "selection_metric": best["selection_metric"],
            "selection_value": float(best["selection_value"]),
        },
        "best_params": best,
        "feature_set_hash": cached.cache_summary["feature_set_version"],
        "feature_names": cached.feature_names,
        "dataset_fingerprint": cached.cache_summary["dataset_fingerprint"],
        "candidate_protocol": {
            "train": config["candidate_protocol"]["train"],
            "validation": config["candidate_protocol"]["validation_objective"],
            "test": "full_7111_items_locked_once",
            "n_negatives": config["candidate_protocol"]["n_negatives"],
            "negative_seed": config["candidate_protocol"]["negative_seed"],
        },
        "model_seed": int(best["xgboost_params"]["seed"]),
        "sampler_seed": int(config["sampler"]["seed"]),
        "boosted_rounds": {
            "best_iteration": int(best["best_iteration"]),
            "best_num_boosted_rounds": int(best["best_num_boosted_rounds"]),
            "early_stopping_rerun": False,
        },
        "model": train_summary,
        "validation_optuna_metrics": compact_metrics(best["validation_metrics"]),
        "validation_reproduced_metrics": validation_metrics,
        "validation_reproduction": {**reproduction, "metrics": validation_metrics},
        "test_evaluation_count": test_evaluation_count,
        "final_test_metrics": final_test_metrics,
        "comparisons": comparisons,
        "runtime": runtime,
        "artifacts": {
            "remote_artifact_path": str(artifact_root),
            "model_path": train_summary["model_path"],
            "validation_ranking_path": validation_eval["ranking_path"],
            "test_ranking_path": test_eval["ranking_path"],
            "validation_summary_path": str(artifact_root / "metrics" / "validation_evaluation_summary.json"),
            "test_summary_path": str(artifact_root / "metrics" / "test_evaluation_summary.json"),
        },
        "remote_artifact_path": str(artifact_root),
        "test_policy": {
            "opened_after_validation_reproduction": True,
            "test_evaluation_count": test_evaluation_count,
            "model_changed_after_test": False,
            "additional_tuning_after_test": False,
        },
    }
    save_json(output_json, summary)
    output_notes.write_text(build_notes(summary), encoding="utf-8")
    save_json(artifact_root / "metrics" / "final_summary.json", summary)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "status": "completed",
                "validation_ndcg10": validation_metrics["NDCG@10"],
                "test_ndcg10": final_test_metrics["NDCG@10"],
                "test_hr10": final_test_metrics["HR@10"],
                "test_evaluation_count": test_evaluation_count,
                "output_json": str(output_json),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
