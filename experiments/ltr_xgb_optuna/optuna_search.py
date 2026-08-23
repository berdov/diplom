#!/usr/bin/env python3
"""Optuna tuning wrapper for the existing ltr_xgb_002 LambdaMART baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "experiments" / "ltr_xgb_baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import run_experiment as base  # noqa: E402


@dataclass
class CachedInputs:
    feature_names: list[str]
    train_df: pd.DataFrame
    dtrain: Any
    validation_queries: pd.DataFrame
    item_features: pd.DataFrame
    item_ids: np.ndarray
    item_to_pos: dict[int, int]
    cache_summary: dict[str, Any]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_value(args: list[str], default: str = "unknown") -> str:
    if args == ["rev-parse", "HEAD"]:
        env_commit = os.environ.get("LTR_OPTUNA_GIT_COMMIT") or os.environ.get("LTR_GIT_COMMIT")
        if env_commit:
            return env_commit
    if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
        env_branch = os.environ.get("LTR_OPTUNA_GIT_BRANCH") or os.environ.get("LTR_GIT_BRANCH")
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


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path}"


def slurm_info() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "hostname": socket.gethostname(),
    }


def environment_info() -> dict[str, Any]:
    import optuna
    import xgboost as xgb

    packages: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "xgboost": xgb.__version__,
        "optuna": optuna.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }
    for name in ["pyarrow", "polars", "sqlalchemy", "alembic"]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = None
    return packages


def validate_config_compatibility(optuna_config: dict[str, Any], baseline_config: dict[str, Any]) -> None:
    if optuna_config["base_run_id"] != "ltr_xgb_002":
        raise AssertionError("Optuna v1 must build on ltr_xgb_002")
    if baseline_config["run_id"] != "ltr_xgb_002":
        raise AssertionError("baseline config is not ltr_xgb_002")
    if baseline_config["train_candidate_protocol"] != "sampled_100":
        raise AssertionError("training candidate protocol changed")
    if baseline_config["eval_candidate_protocol"] != "full_7111_items":
        raise AssertionError("evaluation candidate protocol changed")
    if int(baseline_config["candidate_generation"]["n_negatives"]) != 100:
        raise AssertionError("negative candidate count changed")
    if int(baseline_config["seed"]) != int(optuna_config["model"]["model_seed"]):
        raise AssertionError("model seed changed")
    if bool(optuna_config["test_policy"]["load_test_in_search"]):
        raise AssertionError("search mode must not load test data")
    if bool(optuna_config["test_policy"]["evaluate_test_in_search"]):
        raise AssertionError("search mode must not evaluate test data")


def load_cached_inputs(optuna_config: dict[str, Any], baseline_config: dict[str, Any]) -> CachedInputs:
    import xgboost as xgb

    validate_config_compatibility(optuna_config, baseline_config)
    baseline_json_path = project_path(optuna_config["baseline"]["compact_run_json"])
    baseline_artifact = json.loads(baseline_json_path.read_text(encoding="utf-8"))
    artifact_root = Path(optuna_config["baseline"]["artifact_dir"])

    feature_names = list(baseline_config["features"]["names"])
    if feature_names != baseline_artifact["feature_list"]:
        raise AssertionError("feature list does not match ltr_xgb_002 artifact")

    allowed_paths = {
        "train_features": artifact_root / "features" / "train_features.parquet",
        "validation_queries": artifact_root / "candidates" / "validation_queries.parquet",
        "item_features": artifact_root / "features" / "item_train_popularity.parquet",
    }
    for name, path in allowed_paths.items():
        if "test" in path.name or "test" in name:
            raise AssertionError(f"test path is forbidden in search mode: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Missing cached ltr_xgb_002 artifact: {path}")

    start = time.perf_counter()
    train_df = pd.read_parquet(allowed_paths["train_features"])
    validation_queries = pd.read_parquet(allowed_paths["validation_queries"])
    item_features = pd.read_parquet(allowed_paths["item_features"]).sort_values("candidate_item_id").reset_index(drop=True)
    load_time = time.perf_counter() - start

    expected_group_size = int(baseline_config["candidate_generation"]["n_negatives"]) + 1
    expected_groups = int(train_df["query_index"].nunique())
    group_validation = base.validate_xgb_group_structure(train_df, "train", expected_group_size, expected_groups)

    dtrain = xgb.DMatrix(
        base.feature_matrix(train_df, feature_names),
        label=train_df["label"].to_numpy(dtype=np.float32),
        feature_names=feature_names,
        group=base.group_sizes(train_df),
    )
    item_ids = item_features["candidate_item_id"].to_numpy(dtype=np.int64)
    item_to_pos = {int(item): pos for pos, item in enumerate(item_ids)}
    if int(item_ids.shape[0]) != int(optuna_config["protocol"]["items"]):
        raise AssertionError("item universe size changed")
    if not set(validation_queries["target_item_id"].astype("int64")).issubset(item_to_pos):
        raise AssertionError("validation target item missing from item universe")

    feature_version_payload = {
        "base_run_id": optuna_config["base_run_id"],
        "feature_names": feature_names,
        "candidate_protocol": baseline_artifact["candidate_protocol"],
        "train_candidate_protocol": baseline_artifact["train_candidate_protocol"],
        "eval_candidate_protocol": baseline_artifact["eval_candidate_protocol"],
        "feature_summary": {
            "train": baseline_artifact["feature_summary"]["splits"]["train"]["feature_sha256"],
            "validation": baseline_artifact["feature_summary"]["splits"]["validation"]["feature_sha256"],
        },
        "candidate_summary": {
            "train": baseline_artifact["candidate_summary"]["splits"]["train"]["content_sha256"],
            "validation": baseline_artifact["candidate_summary"]["splits"]["validation"]["content_sha256"],
        },
    }
    cache_summary = {
        "base_run_id": optuna_config["base_run_id"],
        "baseline_git_commit": baseline_artifact["git_commit"],
        "baseline_remote_artifact": str(artifact_root),
        "loaded_paths": {name: str(path) for name, path in allowed_paths.items()},
        "loaded_path_sha256": {name: sha256_file(path) for name, path in allowed_paths.items()},
        "forbidden_test_paths_loaded": [],
        "load_time_sec": float(load_time),
        "group_validation": group_validation,
        "feature_set_version": sha256_json(feature_version_payload),
        "feature_set_payload": feature_version_payload,
        "dataset_fingerprint": baseline_artifact["candidate_summary"]["dataset_fingerprint"],
    }
    return CachedInputs(
        feature_names=feature_names,
        train_df=train_df,
        dtrain=dtrain,
        validation_queries=validation_queries,
        item_features=item_features,
        item_ids=item_ids,
        item_to_pos=item_to_pos,
        cache_summary=cache_summary,
    )


def suggest_params(trial: Any, baseline_config: dict[str, Any], search_space: dict[str, Any], nthread: int) -> dict[str, Any]:
    params = dict(baseline_config["xgboost"])
    params.pop("num_boost_round", None)
    params.pop("early_stopping_rounds", None)
    params["objective"] = "rank:ndcg"
    params["eval_metric"] = "ndcg@10"
    params["tree_method"] = "hist"
    params["seed"] = int(search_space["fixed"]["seed"])
    params["nthread"] = int(nthread)
    params.setdefault("verbosity", 1)

    for name, spec in search_space["parameters"].items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "float_with_zero":
            zero_name = spec.get("zero_choice_name", f"{name}_is_zero")
            if trial.suggest_categorical(zero_name, [True, False]):
                params[name] = 0.0
            else:
                params[name] = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False)),
                )
        else:
            raise ValueError(f"Unknown search-space type for {name}: {kind}")
    return params


def target_ranks_from_topk(scores: np.ndarray, item_ids: np.ndarray, target_items: np.ndarray, topk: int) -> np.ndarray:
    finite_scores = np.where(np.isfinite(scores), scores, np.finfo(np.float32).min)
    kth = min(topk, finite_scores.shape[1]) - 1
    top_unsorted = np.argpartition(-finite_scores, kth=kth, axis=1)[:, :topk]
    top_scores_unsorted = np.take_along_axis(finite_scores, top_unsorted, axis=1)
    top_items_unsorted = item_ids[top_unsorted]
    ranks = np.full(finite_scores.shape[0], np.inf, dtype=np.float64)
    rank_positions = np.arange(1, topk + 1, dtype=np.float64)
    for row_idx in range(top_unsorted.shape[0]):
        order = np.lexsort((top_items_unsorted[row_idx], -top_scores_unsorted[row_idx]))
        sorted_items = top_items_unsorted[row_idx, order]
        hits = sorted_items == target_items[row_idx]
        if bool(hits.any()):
            ranks[row_idx] = rank_positions[int(np.argmax(hits))]
    return ranks


def evaluate_validation_full(
    booster: Any,
    num_boosted_rounds: int,
    cached: CachedInputs,
    optuna_config: dict[str, Any],
) -> dict[str, Any]:
    import xgboost as xgb

    queries = cached.validation_queries.sort_values("query_index").reset_index(drop=True)
    eval_config = optuna_config["evaluation"]
    batch_users = int(eval_config["batch_users"])
    topk = int(eval_config["topk"])
    mask_seen_items = bool(eval_config["mask_seen_items"])
    item_ids = cached.item_ids
    item_to_pos = cached.item_to_pos

    start = time.perf_counter()
    rank_batches: list[np.ndarray] = []
    candidate_counts: list[np.ndarray] = []
    target_evaluable_batches: list[np.ndarray] = []
    for start_row in range(0, int(queries.shape[0]), batch_users):
        batch = queries.iloc[start_row : start_row + batch_users].copy()
        batch_size = int(batch.shape[0])
        matrix = base.full_feature_batch(
            batch,
            cached.item_features,
            item_ids,
            item_to_pos,
            cached.feature_names,
        )
        dmat = xgb.DMatrix(matrix, feature_names=cached.feature_names)
        predicted = booster.predict(dmat, iteration_range=(0, int(num_boosted_rounds)))
        scores = predicted.reshape(batch_size, item_ids.shape[0]).astype(np.float32, copy=False)
        scores, counts = base.apply_full_masks(scores, batch, item_to_pos, mask_seen_items)
        target_items = batch["target_item_id"].to_numpy(dtype=np.int64)
        target_positions = np.array([item_to_pos[int(item)] for item in target_items], dtype=np.int64)
        target_scores = scores[np.arange(batch_size), target_positions]
        target_evaluable_batches.append(np.isfinite(target_scores))
        candidate_counts.append(counts)
        rank_batches.append(target_ranks_from_topk(scores, item_ids, target_items, topk))

    elapsed = time.perf_counter() - start
    target_ranks = np.concatenate(rank_batches)
    metrics = base.metrics_from_target_ranks(target_ranks)
    split_stats = base.validate_full_eval_split(
        queries,
        item_to_pos,
        candidate_counts,
        target_evaluable_batches,
        mask_seen_items,
    )
    return {
        "metrics": metrics,
        "eval_time_sec": float(elapsed),
        "users": int(queries.shape[0]),
        "items": int(item_ids.shape[0]),
        "scores": int(queries.shape[0] * item_ids.shape[0]),
        "batch_users": int(batch_users),
        "topk": int(topk),
        "mask_seen_items": bool(mask_seen_items),
        "split_stats": split_stats,
    }


def train_trial(
    trial: Any,
    params: dict[str, Any],
    cached: CachedInputs,
    optuna_config: dict[str, Any],
    artifact_root: Path,
    max_boost_rounds: int,
    eval_every_rounds: int,
    patience_evals: int,
) -> dict[str, Any]:
    import optuna
    import xgboost as xgb

    trial_dir = artifact_root / "trials" / f"trial_{trial.number:04d}"
    model_dir = artifact_root / "models"
    trial_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    booster = None
    rounds_done = 0
    best_ndcg10 = -1.0
    best_iteration_zero_based = None
    best_num_boosted_rounds = None
    best_metrics: dict[str, float] | None = None
    best_eval: dict[str, Any] | None = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    min_delta = float(optuna_config["model"].get("early_stopping_min_delta", 0.0))

    train_start = time.perf_counter()
    stop_reason = "max_boost_rounds_reached"
    while rounds_done < max_boost_rounds:
        step_rounds = min(eval_every_rounds, max_boost_rounds - rounds_done)
        step_start = time.perf_counter()
        booster = xgb.train(
            params=params,
            dtrain=cached.dtrain,
            num_boost_round=step_rounds,
            evals=[],
            xgb_model=booster,
            verbose_eval=False,
        )
        rounds_done += step_rounds
        step_train_time = time.perf_counter() - step_start

        validation = evaluate_validation_full(booster, rounds_done, cached, optuna_config)
        ndcg10 = float(validation["metrics"]["NDCG@10"])
        improved = ndcg10 > best_ndcg10 + min_delta
        if improved:
            best_ndcg10 = ndcg10
            best_iteration_zero_based = rounds_done - 1
            best_num_boosted_rounds = rounds_done
            best_metrics = dict(validation["metrics"])
            best_eval = validation
            best_model_path = model_dir / f"trial_{trial.number:04d}_best.json"
            booster.save_model(best_model_path)
            no_improvement = 0
        else:
            no_improvement += 1

        record = {
            "num_boosted_rounds": int(rounds_done),
            "iteration_zero_based": int(rounds_done - 1),
            "train_step_time_sec": float(step_train_time),
            "full_validation_time_sec": float(validation["eval_time_sec"]),
            "validation_metrics": validation["metrics"],
            "improved": bool(improved),
        }
        history.append(record)
        trial.report(ndcg10, step=rounds_done)
        if trial.should_prune():
            stop_reason = "optuna_pruned"
            raise optuna.exceptions.TrialPruned()  # type: ignore[name-defined]
        if no_improvement >= patience_evals:
            stop_reason = f"early_stopping_no_full_validation_improvement_{patience_evals}"
            break

    trial_runtime = time.perf_counter() - train_start
    xgboost_train_time = sum(float(item["train_step_time_sec"]) for item in history)
    if booster is None or best_metrics is None or best_eval is None or best_num_boosted_rounds is None:
        raise AssertionError("trial finished without a validation evaluation")
    final_model_path = model_dir / f"trial_{trial.number:04d}_final.json"
    booster.save_model(final_model_path)
    best_model_path = model_dir / f"trial_{trial.number:04d}_best.json"
    if not best_model_path.exists():
        booster.save_model(best_model_path)

    return {
        "trial_number": int(trial.number),
        "state": "COMPLETE",
        "params": params,
        "objective": "validation_full_ranking_NDCG@10",
        "validation_ndcg10": float(best_metrics["NDCG@10"]),
        "validation_hr10": float(best_metrics["HR@10"]),
        "best_iteration": int(best_iteration_zero_based),
        "best_num_boosted_rounds": int(best_num_boosted_rounds),
        "num_boosted_rounds_trained": int(rounds_done),
        "stop_reason": stop_reason,
        "trial_runtime_sec": float(trial_runtime),
        "xgboost_train_time_sec": float(xgboost_train_time),
        "train_time_sec": float(xgboost_train_time),
        "best_full_validation_time_sec": float(best_eval["eval_time_sec"]),
        "validation_metrics": best_metrics,
        "validation_eval": {
            key: value
            for key, value in best_eval.items()
            if key != "metrics"
        },
        "history": history,
        "model_paths": {
            "best": str(best_model_path),
            "final": str(final_model_path),
        },
        "model_sha256": {
            "best": sha256_file(best_model_path),
            "final": sha256_file(final_model_path),
        },
        "seed": int(params.get("seed", 42)),
        "feature_set_version": cached.cache_summary["feature_set_version"],
        "dataset_fingerprint": cached.cache_summary["dataset_fingerprint"],
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
        },
        "forbidden_test_paths_loaded": [],
        "test_evaluation_count": 0,
    }


def build_notes(result: dict[str, Any]) -> str:
    trial = result["smoke_trial"]
    lines = [
        "# Optuna smoke 001",
        "",
        "## Цель",
        "",
        "Проверить pipeline подбора гиперпараметров XGBoost LambdaMART поверх `ltr_xgb_002` без изменения признаков, candidates и evaluation protocol.",
        "",
        "## Test policy",
        "",
        f"- Test evaluation count: `{result['test_evaluation_count']}`.",
        "- Test split не загружался и не использовался в objective, pruning, model selection или отчетной метрике.",
        "- Objective: validation full-ranking `NDCG@10`.",
        "",
        "## Baseline",
        "",
        f"- Base run: `{result['baseline']['run_id']}`.",
        f"- Training candidates: `{result['baseline']['train_candidate_protocol']}`.",
        f"- Validation objective candidates: `{result['baseline']['validation_candidate_protocol']}`.",
        f"- Feature set version: `{result['cache']['feature_set_version']}`.",
        "",
        "## Optuna",
        "",
        f"- Study name: `{result['optuna']['study_name']}`.",
        f"- Storage: `{result['optuna']['storage']}`.",
        f"- Sampler: `{result['optuna']['sampler']}`.",
        f"- Sampler seed: `{result['optuna']['sampler_seed']}`.",
        "",
        "## Smoke trial",
        "",
        f"- Trial number: `{trial['trial_number']}`.",
        f"- Status: `{trial['state']}`.",
        f"- Best iteration: `{trial['best_iteration']}`.",
        f"- Best boosted rounds: `{trial['best_num_boosted_rounds']}`.",
        f"- Validation NDCG@10: `{trial['validation_ndcg10']:.6f}`.",
        f"- Validation HR@10: `{trial['validation_hr10']:.6f}`.",
        f"- Trial runtime: `{trial.get('trial_runtime_sec', trial['train_time_sec']):.2f}` sec.",
        f"- XGBoost train time: `{trial.get('xgboost_train_time_sec', trial['train_time_sec']):.2f}` sec.",
        f"- Best full-ranking validation time: `{trial['best_full_validation_time_sec']:.2f}` sec.",
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(trial["params"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Full-ranking validation cost",
        "",
        f"- Users: `{trial['validation_eval']['users']}`.",
        f"- Items: `{trial['validation_eval']['items']}`.",
        f"- Scores per full validation: `{trial['validation_eval']['scores']}`.",
        f"- Batch users: `{trial['validation_eval']['batch_users']}`.",
        "",
        "## Caching",
        "",
        f"- Reused artifact root: `{result['cache']['baseline_remote_artifact']}`.",
        f"- Forbidden test paths loaded: `{result['cache']['forbidden_test_paths_loaded']}`.",
        "",
        "## Decision",
        "",
        "- Pipeline creates/resumes SQLite Optuna study.",
        "- Trial parameters are passed to XGBoost.",
        "- Model selection uses full-ranking validation, not sampled validation metric.",
        "- This smoke trial is not a final result and is not written to `experiments/results.csv`.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/config.yaml"))
    parser.add_argument("--search-space", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/search_space.yaml"))
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--target-complete", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-one-smoke-trial", action="store_true")
    return parser.parse_args()


def main() -> None:
    import optuna

    args = parse_args()
    optuna_config = load_yaml(Path(args.config))
    search_space = load_yaml(Path(args.search_space))
    baseline_config = load_yaml(project_path(optuna_config["baseline"]["config_path"]))
    artifact_root = Path(optuna_config["remote_artifact_dir"])
    runs_dir = Path(optuna_config["compact_runs_dir"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    smoke = bool(args.smoke)
    if smoke and args.target_complete is not None:
        raise AssertionError("target-complete is not supported in smoke mode")
    n_trials = int(args.n_trials if args.n_trials is not None else (optuna_config["smoke"]["n_trials"] if smoke else 1))
    if smoke and n_trials != 1:
        raise AssertionError("smoke mode must run exactly one trial")

    cached = load_cached_inputs(optuna_config, baseline_config)
    nthread = int(os.environ.get("SLURM_CPUS_PER_TASK", optuna_config["model"]["nthread"]))

    storage_path = Path(optuna_config["study_storage"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = sqlite_url(storage_path)
    sampler_seed = int(optuna_config["sampler"]["seed"])
    study = optuna.create_study(
        study_name=optuna_config["study_name"],
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.NopPruner(),
    )
    trial_count_before = len(study.trials)
    complete_before = sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE)
    if args.target_complete is not None:
        target_complete = int(args.target_complete)
        if target_complete <= 0:
            raise AssertionError("--target-complete must be positive")
        n_trials = max(0, target_complete - complete_before)
        print(
            json.dumps(
                {
                    "study_name": optuna_config["study_name"],
                    "complete_before": complete_before,
                    "target_complete": target_complete,
                    "new_trials_to_run": n_trials,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    max_boost_rounds = int(optuna_config["smoke"]["max_boost_rounds"] if smoke else optuna_config["model"]["max_boost_rounds"])
    eval_every_rounds = int(
        optuna_config["smoke"]["full_validation_eval_every_rounds"]
        if smoke
        else optuna_config["evaluation"]["full_validation_eval_every_rounds"]
    )
    patience_evals = int(
        optuna_config["smoke"]["early_stopping_patience_evals"]
        if smoke
        else optuna_config["model"]["early_stopping_patience_evals"]
    )

    trial_results: dict[int, dict[str, Any]] = {}

    def objective(trial: Any) -> float:
        params = suggest_params(trial, baseline_config, search_space, nthread=nthread)
        result = train_trial(
            trial,
            params,
            cached,
            optuna_config,
            artifact_root,
            max_boost_rounds=max_boost_rounds,
            eval_every_rounds=eval_every_rounds,
            patience_evals=patience_evals,
        )
        trial.set_user_attr("result", result)
        trial.set_user_attr("validation_hr10", result["validation_hr10"])
        trial.set_user_attr("best_iteration", result["best_iteration"])
        trial.set_user_attr("test_evaluation_count", 0)
        trial_results[int(trial.number)] = result
        return float(result["validation_ndcg10"])

    started = time.perf_counter()
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
    total_runtime = time.perf_counter() - started
    reloaded = optuna.load_study(study_name=optuna_config["study_name"], storage=storage)
    if len(reloaded.trials) < trial_count_before + n_trials:
        raise AssertionError("Optuna resume/load check failed")

    new_trials = [trial for trial in reloaded.trials if int(trial.number) >= trial_count_before]
    if smoke:
        if len(new_trials) != 1:
            raise AssertionError(f"Expected exactly one new smoke trial, got {len(new_trials)}")
        trial = new_trials[0]
        trial_result = trial.user_attrs.get("result") or trial_results[int(trial.number)]
        result = {
            "run_id": optuna_config["smoke_run_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": {
                "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
                "commit": git_value(["rev-parse", "HEAD"]),
            },
            "environment": environment_info(),
            "slurm": slurm_info(),
            "baseline": {
                "run_id": optuna_config["base_run_id"],
                "baseline_config": optuna_config["baseline"]["config_path"],
                "train_candidate_protocol": baseline_config["train_candidate_protocol"],
                "validation_candidate_protocol": optuna_config["candidate_protocol"]["validation_objective"],
                "evaluation_protocol": baseline_config["evaluation_protocol"],
                "seed": int(baseline_config["seed"]),
                "features": cached.feature_names,
            },
            "optuna": {
                "version": environment_info()["optuna"],
                "study_name": optuna_config["study_name"],
                "storage": storage,
                "storage_path": str(storage_path),
                "sampler": optuna_config["sampler"]["name"],
                "sampler_seed": sampler_seed,
                "n_trials_requested": n_trials,
                "trial_count_before": int(trial_count_before),
                "trial_count_after": int(len(reloaded.trials)),
                "resume_check": True,
            },
            "search_space": search_space,
            "cache": cached.cache_summary,
            "smoke_trial": trial_result,
            "runtime_sec": float(total_runtime),
            "test_evaluation_count": 0,
            "results_csv_updated": False,
            "future_search_estimate": {
                "trials": 40,
                "per_trial_sec_from_smoke": float(trial_result.get("trial_runtime_sec", trial_result["train_time_sec"])),
                "approx_total_sec": float(trial_result.get("trial_runtime_sec", trial_result["train_time_sec"]) * 40),
                "per_full_validation_sec_from_smoke": float(trial_result["best_full_validation_time_sec"]),
                "bottleneck": "repeated full-ranking validation over 23951 users x 7111 items",
            },
        }
        json_path = runs_dir / f"{optuna_config['smoke_run_id']}.json"
        notes_path = runs_dir / f"{optuna_config['smoke_run_id']}_notes.md"
        save_json(json_path, result)
        notes_path.write_text(build_notes(result), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "study_name": optuna_config["study_name"],
            "storage": storage,
            "n_trials_requested": n_trials,
            "trial_count_before": int(trial_count_before),
            "trial_count_after": int(len(reloaded.trials)),
            "test_evaluation_count": 0,
        }
        save_json(runs_dir / "study_last_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
