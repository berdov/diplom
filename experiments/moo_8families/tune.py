#!/usr/bin/env python
"""Controlled validation-only Optuna tuning for selected MOO families."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import optuna
import yaml
from optuna.trial import TrialState


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPACES = ROOT / "configs" / "moo_tuning_spaces.yaml"
METHODS = ("epo", "gradhv", "cosmos", "pcgrad")
SAMPLED_OVERRIDE_KEYS = {"learning_rate", "weight_decay", "dropout_prob", "head_lr_multiplier"}
DEGENERATE_SPREAD_EPS = 1e-6
FORBIDDEN_TUNING_PARAMS = {
    "lambda_aux",
    "aux_scale",
    "w_click_raw",
    "w_long_view_raw",
    "w_like_raw",
    "w_profile_raw",
    "alpha_common",
    "alpha_rare",
}


def env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default=str(DEFAULT_SPACES))
    parser.add_argument("--method", choices=(*METHODS, "all"), required=True)
    parser.add_argument("--tuning-stage", choices=("stage_a", "stage_b", "full"), default=os.environ.get("MOO_TUNING_STAGE", "stage_a"))
    parser.add_argument("--target-complete", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--fail-stale-running", action="store_true")
    parser.add_argument("--stale-reason", default=os.environ.get("MOO_TUNING_STALE_REASON", "walltime timeout"))
    parser.add_argument("--stale-min-age-hours", type=float, default=env_float("MOO_TUNING_STALE_MIN_AGE_HOURS") or 0.0)
    parser.add_argument("--max-worker-runtime-sec", type=float, default=env_float("MOO_TUNING_MAX_WORKER_RUNTIME_SEC"))
    parser.add_argument("--min-runtime-buffer-sec", type=float, default=env_float("MOO_TUNING_MIN_RUNTIME_BUFFER_SEC"))
    parser.add_argument("--estimated-trial-runtime-sec", type=float, default=env_float("MOO_TUNING_ESTIMATED_TRIAL_RUNTIME_SEC"))
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def assert_tuning_guardrails(spaces: Mapping[str, Any]) -> None:
    protocol = spaces["protocol"]
    test_policy = protocol["test_policy"]
    if any(bool(test_policy[key]) for key in ("load_test_dataset", "create_test_dataloader", "evaluate_test")):
        raise RuntimeError(f"MOO tuning must stay validation-only: {test_policy}")
    if int(test_policy["test_evaluation_count"]) != 0:
        raise RuntimeError(f"test_evaluation_count must be 0: {test_policy}")
    reference = spaces["frozen_parameters"]["evaluation_reference"]
    if [float(value) for value in reference["values"]] != [1.0, 2.0, 2.0, 2.0, 2.0]:
        raise RuntimeError(f"Unexpected Pareto reference for tuning: {reference}")
    if str(reference.get("invalid_reference_policy")) != "raise":
        raise RuntimeError(f"invalid_reference_policy must remain raise: {reference}")

    for method, method_spec in spaces["methods"].items():
        params = set(method_spec["parameters"])
        forbidden = sorted(params & FORBIDDEN_TUNING_PARAMS)
        if forbidden:
            raise RuntimeError(f"{method}: forbidden tuning parameter(s) would change the objective: {forbidden}")
        for name in params:
            spec = param_spec(spaces, method_spec, name)
            if spec.get("type") != "float":
                raise RuntimeError(f"{method}: only float tuning params are supported, got {name}: {spec}")
            current = float(spec["current"])
            low = float(spec["min"])
            high = float(spec["max"])
            if not (low <= current <= high):
                raise RuntimeError(f"{method}: current {name}={current} is outside [{low}, {high}]")


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        value = subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return value or default
    except Exception:
        return default


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path}"


def method_list(requested: str) -> list[str]:
    return list(METHODS) if requested == "all" else [requested]


def param_spec(spaces: Mapping[str, Any], method_spec: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name in spaces["common_parameters"]:
        return spaces["common_parameters"][name]
    return method_spec.get("method_parameters", {})[name]


def baseline_params(spaces: Mapping[str, Any], method_spec: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(param_spec(spaces, method_spec, name)["current"])
        for name in method_spec["parameters"]
    }


def suggest_params(trial: optuna.Trial, spaces: Mapping[str, Any], method_spec: Mapping[str, Any]) -> dict[str, float]:
    params: dict[str, float] = {}
    for name in method_spec["parameters"]:
        spec = param_spec(spaces, method_spec, name)
        if spec.get("type") != "float":
            raise RuntimeError(f"Only float tuning params are supported, got {name}: {spec}")
        params[name] = float(
            trial.suggest_float(
                name,
                float(spec["min"]),
                float(spec["max"]),
                log=str(spec.get("scale", "linear")) == "log",
            )
        )
    return params


def nested_set(target: dict[str, Any], dotted_path: str, value: float) -> None:
    current = target
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = float(value)


def overrides_for_params(spaces: Mapping[str, Any], method_spec: Mapping[str, Any], params: Mapping[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    sampled = {key: float(params[key]) for key in params if key in SAMPLED_OVERRIDE_KEYS}
    method_overrides: dict[str, Any] = {}
    for name, value in params.items():
        if name in SAMPLED_OVERRIDE_KEYS:
            continue
        spec = param_spec(spaces, method_spec, name)
        target_path = spec.get("target_path")
        if not target_path:
            raise RuntimeError(f"Method-specific parameter {name} has no target_path.")
        nested_set(method_overrides, str(target_path), float(value))
    return sampled, method_overrides


def artifact_paths(spaces: Mapping[str, Any], method_spec: Mapping[str, Any], trial_number: int) -> dict[str, Path]:
    study = str(method_spec["study_name"])
    root = project_path(spaces["storage"]["artifact_root"]) / study / f"trial_{trial_number:04d}"
    return {
        "artifact_dir": root / "artifacts",
        "result_json": root / "result.json",
        "notes": root / "notes.md",
        "sampled_overrides": root / "sampled_overrides.json",
        "method_overrides": root / "method_overrides.json",
        "tuning_metadata": root / "tuning_metadata.json",
    }


def train_command(
    *,
    method: str,
    spaces: Mapping[str, Any],
    method_spec: Mapping[str, Any],
    trial: optuna.Trial,
    params: Mapping[str, float],
    allow_cpu: bool,
) -> list[str]:
    paths = artifact_paths(spaces, method_spec, int(trial.number))
    sampled_overrides, method_overrides = overrides_for_params(spaces, method_spec, params)
    metadata = {
        "study_group": spaces["study_group"],
        "study_name": method_spec["study_name"],
        "trial_number": int(trial.number),
        "baseline_trial": int(trial.number) == 0,
        "params": dict(params),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "test_evaluation_count": 0,
    }
    save_json(paths["sampled_overrides"], sampled_overrides)
    save_json(paths["method_overrides"], method_overrides)
    save_json(paths["tuning_metadata"], metadata)
    run_id = f"{method_spec['study_name']}_trial_{trial.number:04d}"
    command = [
        sys.executable,
        "-m",
        "experiments.moo_8families.train",
        "--method",
        method,
        "--stage",
        "tuning",
        "--run-id",
        run_id,
        "--config",
        str(project_path(spaces["source_config"])),
        "--artifact-dir",
        str(paths["artifact_dir"]),
        "--result-json",
        str(paths["result_json"]),
        "--notes",
        str(paths["notes"]),
        "--sampled-params-json",
        str(paths["sampled_overrides"]),
        "--method-overrides-json",
        str(paths["method_overrides"]),
        "--tuning-metadata-json",
        str(paths["tuning_metadata"]),
    ]
    if allow_cpu:
        command.append("--allow-cpu")
    return command


def bce_summary(result: Mapping[str, Any]) -> dict[str, float | None]:
    record = result["validation"]["ranking_operating_point"]
    aux = record.get("auxiliary_validation") or {}
    return {
        "click": None if "is_click" not in aux else float(aux["is_click"]["bce_loss"]),
        "long_view": None if "long_view" not in aux else float(aux["long_view"]["bce_loss"]),
        "like": None if "is_like" not in aux else float(aux["is_like"]["bce_loss"]),
        "profile": None if "is_profile_enter" not in aux else float(aux["is_profile_enter"]["bce_loss"]),
    }


def pairwise_l2_mean(points: list[list[float]]) -> float | None:
    if len(points) <= 1:
        return None
    distances = []
    for left_idx, left in enumerate(points):
        for right in points[left_idx + 1 :]:
            distances.append(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) ** 0.5)
    return float(sum(distances) / len(distances)) if distances else None


def validation_points_summary(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in result["validation"].get("records", []):
        aux = record.get("auxiliary_validation") or {}
        metrics = record.get("metrics") or {}
        rows.append(
            {
                "id": record.get("preference_id", record.get("solution_index")),
                "HR@10": metrics.get("HR@10"),
                "NDCG@10": metrics.get("NDCG@10"),
                "objective_point": record.get("objective_point"),
                "click_BCE": None if "is_click" not in aux else aux["is_click"].get("bce_loss"),
                "long_view_BCE": None if "long_view" not in aux else aux["long_view"].get("bce_loss"),
                "like_BCE": None if "is_like" not in aux else aux["is_like"].get("bce_loss"),
                "profile_BCE": None if "is_profile_enter" not in aux else aux["is_profile_enter"].get("bce_loss"),
            }
        )
    return rows


def method_diagnostic_flags(method: str, result: Mapping[str, Any]) -> dict[str, Any]:
    pareto = result["validation"]["pareto_validation"]
    solution_count = int(pareto.get("solution_count") or 0)
    non_dominated = int(pareto.get("non_dominated_count") or 0)
    spread = pareto.get("spread_l2_mean")
    spread_value = None if spread is None else float(spread)
    degenerate = solution_count > 1 and (non_dominated <= 1 or spread_value is None or spread_value <= DEGENERATE_SPREAD_EPS)
    flags: dict[str, Any] = {
        "pareto_status": "PARETO_DEGENERATE" if degenerate else "PARETO_NON_DEGENERATE",
        "degenerate_spread_eps": DEGENERATE_SPREAD_EPS,
    }
    if method == "cosmos":
        sensitivity = result.get("preference_sensitivity") or {}
        sensitivity_passed = bool(sensitivity.get("output_metric_passed"))
        flags["controllability_status"] = "NON_COLLAPSED" if sensitivity_passed and not degenerate else "COLLAPSED"
    return flags


def training_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    epochs = result.get("training", {}).get("epochs", [])
    last_losses = epochs[-1].get("losses", {}) if epochs else {}
    method_state = last_losses.get("method_state") or {}
    train_hv = None
    if isinstance(method_state, Mapping):
        last_record = method_state.get("last_record") or {}
        train_hv = last_record.get("hypervolume") if isinstance(last_record, Mapping) else None
    return {
        "train_hv_last": train_hv,
        "solver_fallback_count": int(sum((epoch.get("losses") or {}).get("solver_fallback_count", 0) for epoch in epochs)),
        "solver_fallback_by_type": merge_epoch_counts(epochs, "solver_fallback_by_type"),
    }


def merge_epoch_counts(epochs: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for epoch in epochs:
        payload = (epoch.get("losses") or {}).get(key) or {}
        if not isinstance(payload, Mapping):
            continue
        for name, value in payload.items():
            merged[str(name)] = merged.get(str(name), 0) + int(value)
    return merged


def compact_trial_summary(method: str, trial_number: int, params: Mapping[str, float], result: Mapping[str, Any]) -> dict[str, Any]:
    ranking = result["validation"]["ranking_operating_point"]
    metrics = ranking["metrics"]
    pareto = result["validation"]["pareto_validation"]
    sensitivity = result.get("preference_sensitivity")
    validation_points = validation_points_summary(result)
    objective_points = [list(row["objective_point"]) for row in validation_points if row.get("objective_point") is not None]
    return {
        "method": method,
        "trial": int(trial_number),
        "run_id": result["run_id"],
        "state": result["status"],
        "NDCG@10": float(metrics["NDCG@10"]),
        "HR@10": float(metrics["HR@10"]),
        "best_epoch": int(result["best_epoch"]),
        "stop_epoch": int(result["stop_epoch"]),
        "runtime_sec": float(result["runtime"]["total_sec"]),
        "params": dict(params),
        "HV": pareto.get("hypervolume"),
        "non_dominated": pareto.get("non_dominated_count"),
        "spread": pareto.get("spread_l2_mean"),
        "mean_pairwise_objective_distance": pairwise_l2_mean(objective_points),
        "aux_bce": bce_summary(result),
        "validation_points": validation_points,
        "diagnostic_flags": method_diagnostic_flags(method, result),
        "training_diagnostics": training_diagnostics(result),
        "preference_sensitivity": sensitivity,
        "test_evaluation_count": int(result.get("test_evaluation_count", -1)),
        "artifact_dir": result.get("artifact_dir"),
        "git": result.get("git"),
        "gpu": result.get("gpu"),
        "slurm": result.get("slurm"),
    }


def run_trial(
    trial: optuna.Trial,
    *,
    method: str,
    spaces: Mapping[str, Any],
    method_spec: Mapping[str, Any],
    allow_cpu: bool,
) -> float:
    params = baseline_params(spaces, method_spec) if trial.number == 0 else suggest_params(trial, spaces, method_spec)
    command = train_command(method=method, spaces=spaces, method_spec=method_spec, trial=trial, params=params, allow_cpu=allow_cpu)
    if os.environ.get("MOO_TUNING_PRINT_COMMANDS") == "1":
        print(" ".join(command), flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Training command failed with exit code {completed.returncode}: {' '.join(command)}")
    result_path = artifact_paths(spaces, method_spec, int(trial.number))["result_json"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if int(result.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError(f"TEST contamination in {result_path}: test_evaluation_count={result.get('test_evaluation_count')}")
    summary = compact_trial_summary(method, int(trial.number), params, result)
    summary["orchestrator_runtime_sec"] = float(time.monotonic() - started)
    trial.set_user_attr("summary", summary)
    trial.set_user_attr("test_evaluation_count", 0)
    trial.set_user_attr("best_epoch", summary["best_epoch"])
    trial.set_user_attr("HR@10", summary["HR@10"])
    value = float(summary["NDCG@10"])
    if trial.number == 0:
        expected = float(method_spec["current_ndcg10"])
        tolerance = float(spaces["training_protocol"]["baseline_reproduction_tolerance_ndcg10"])
        if abs(value - expected) > tolerance:
            raise RuntimeError(
                f"Baseline trial mismatch for {method}: observed NDCG@10={value}, "
                f"expected={expected}, tolerance={tolerance}. Stop this study and audit pipeline mismatch."
            )
    return value


def complete_count(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)


def target_complete_trials(method_spec: Mapping[str, Any], args: argparse.Namespace) -> int:
    if args.target_complete is not None:
        return int(args.target_complete)
    if args.tuning_stage in {"stage_a", "stage_b"}:
        stage_spec = method_spec.get(args.tuning_stage) or {}
        if "target_complete_trials" in stage_spec:
            return int(stage_spec["target_complete_trials"])
    return int(method_spec["target_complete_trials"])


def state_counts(study: optuna.Study) -> dict[str, int]:
    counts = {state.name: 0 for state in TrialState}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    return counts


def complete_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]


def running_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.RUNNING]


def trial_age_hours(trial: optuna.trial.FrozenTrial) -> float | None:
    started = trial.datetime_start
    if started is None:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return float((datetime.now(timezone.utc) - started).total_seconds() / 3600.0)


def trial_summary_or_fallback(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    summary = trial.user_attrs.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return {
        "trial": int(trial.number),
        "NDCG@10": trial.value,
        "params": dict(trial.params),
        "test_evaluation_count": trial.user_attrs.get("test_evaluation_count"),
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "HR@10": trial.user_attrs.get("HR@10"),
    }


def parameter_importance(study: optuna.Study) -> dict[str, float]:
    if len(complete_trials(study)) < 2:
        return {}
    try:
        return {key: float(value) for key, value in optuna.importance.get_param_importances(study).items()}
    except Exception as exc:
        return {"__error__": repr(exc)}


def objective_distribution(study: optuna.Study) -> dict[str, float | None]:
    values = sorted(float(trial.value) for trial in complete_trials(study))
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}

    def quantile(q: float) -> float:
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
        return values[idx]

    return {
        "min": values[0],
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "max": values[-1],
    }


def runtime_estimate_sec(
    study: optuna.Study,
    method_spec: Mapping[str, Any],
    args: argparse.Namespace,
) -> float | None:
    if args.estimated_trial_runtime_sec is not None:
        return float(args.estimated_trial_runtime_sec)
    runtimes = []
    for trial in complete_trials(study):
        summary = trial_summary_or_fallback(trial)
        runtime = summary.get("runtime_sec") or summary.get("orchestrator_runtime_sec")
        if runtime is not None:
            runtimes.append(float(runtime))
    if runtimes:
        runtimes.sort()
        idx = min(len(runtimes) - 1, max(0, round((len(runtimes) - 1) * 0.9)))
        return float(runtimes[idx])
    runtime_spec = method_spec.get("runtime") or {}
    if "estimated_trial_sec" in runtime_spec:
        return float(runtime_spec["estimated_trial_sec"])
    return None


def enough_time_for_next_trial(started: float, study: optuna.Study, method_spec: Mapping[str, Any], args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    if args.max_worker_runtime_sec is None:
        return True, {"walltime_aware": False}
    estimate = runtime_estimate_sec(study, method_spec, args)
    buffer_sec = float(args.min_runtime_buffer_sec if args.min_runtime_buffer_sec is not None else 1800.0)
    elapsed = float(time.monotonic() - started)
    remaining = float(args.max_worker_runtime_sec) - elapsed
    decision = {
        "walltime_aware": True,
        "elapsed_sec": elapsed,
        "remaining_sec": remaining,
        "estimated_trial_runtime_sec": estimate,
        "min_runtime_buffer_sec": buffer_sec,
    }
    if estimate is None:
        return True, decision | {"reason": "no_estimate_available"}
    allowed = remaining >= estimate + buffer_sec
    return allowed, decision | {"reason": "enough_time" if allowed else "insufficient_time_for_next_trial"}


def summary_from_study(
    method: str,
    spaces: Mapping[str, Any],
    method_spec: Mapping[str, Any],
    study: optuna.Study,
    *,
    target: int,
    tuning_stage: str,
    storage_path: Path,
    stale_handling: Mapping[str, Any] | None = None,
    worker_stop: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]
    ranked = sorted(complete, key=lambda item: float(item.value), reverse=True)
    summaries = [trial_summary_or_fallback(trial) for trial in ranked]
    running = [
        {
            "trial": int(trial.number),
            "datetime_start": None if trial.datetime_start is None else trial.datetime_start.isoformat(),
            "age_hours": trial_age_hours(trial),
            "params": dict(trial.params),
            "user_attrs": dict(trial.user_attrs),
        }
        for trial in running_trials(study)
    ]
    return {
        "method": method,
        "study_group": spaces["study_group"],
        "study_name": method_spec["study_name"],
        "storage_path": str(storage_path),
        "tuning_stage": tuning_stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "state_counts": state_counts(study),
        "target_complete_trials": int(target),
        "remaining_to_target": max(int(target) - len(complete), 0),
        "running_trials": running,
        "best_trial": summaries[0] if summaries else None,
        "top_trials": summaries[:10],
        "parameter_importance": parameter_importance(study),
        "objective_distribution": objective_distribution(study),
        "stage_b_ready": bool(tuning_stage == "stage_b" or len(complete) >= int((method_spec.get("stage_a") or {}).get("target_complete_trials", target))),
        "stale_handling": dict(stale_handling or {}),
        "worker_stop": dict(worker_stop or {}),
        "test_evaluation_count": 0,
    }


def storage_path_for(spaces: Mapping[str, Any], method_spec: Mapping[str, Any]) -> Path:
    return project_path(spaces["storage"]["storage_root"]) / f"{method_spec['study_name']}.db"


def storage_for(spaces: Mapping[str, Any], method_spec: Mapping[str, Any]) -> tuple[optuna.storages.RDBStorage, Path]:
    storage_path = storage_path_for(spaces, method_spec)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_cfg = spaces.get("storage", {})
    kwargs: dict[str, Any] = {
        "engine_kwargs": {
            "connect_args": {
                "timeout": int(storage_cfg.get("sqlite_timeout_sec", 120)),
            }
        }
    }
    if "heartbeat_interval_sec" in storage_cfg:
        kwargs["heartbeat_interval"] = int(storage_cfg["heartbeat_interval_sec"])
    if "grace_period_sec" in storage_cfg:
        kwargs["grace_period"] = int(storage_cfg["grace_period_sec"])
    try:
        storage = optuna.storages.RDBStorage(url=sqlite_url(storage_path), **kwargs)
    except TypeError:
        storage = optuna.storages.RDBStorage(url=sqlite_url(storage_path), engine_kwargs=kwargs["engine_kwargs"])
    return storage, storage_path


def load_existing_study(storage: optuna.storages.RDBStorage, method_spec: Mapping[str, Any], storage_path: Path) -> optuna.Study | None:
    if not storage_path.exists():
        return None
    try:
        return optuna.load_study(study_name=str(method_spec["study_name"]), storage=storage)
    except (KeyError, ValueError):
        return None


def fail_heartbeat_stale_trials(study: optuna.Study) -> dict[str, Any]:
    result = {"attempted": False, "ok": None, "error": None}
    if not hasattr(optuna.storages, "fail_stale_trials"):
        return result
    result["attempted"] = True
    try:
        optuna.storages.fail_stale_trials(study)
        result["ok"] = True
    except Exception as exc:  # pragma: no cover - depends on Optuna storage backend.
        result["ok"] = False
        result["error"] = repr(exc)
    return result


def mark_running_trials_failed(
    study: optuna.Study,
    *,
    min_age_hours: float,
    reason: str,
) -> dict[str, Any]:
    repaired = []
    skipped = []
    for trial in running_trials(study):
        age = trial_age_hours(trial)
        if age is not None and age < float(min_age_hours):
            skipped.append({"trial": int(trial.number), "age_hours": age, "reason": "too_young"})
            continue
        payload = {
            "trial": int(trial.number),
            "datetime_start": None if trial.datetime_start is None else trial.datetime_start.isoformat(),
            "age_hours": age,
            "reason": reason,
        }
        try:
            study.tell(int(trial.number), state=TrialState.FAIL, skip_if_finished=True)
        except TypeError:
            study.tell(int(trial.number), state=TrialState.FAIL)
        repaired.append(payload)
    return {
        "manual_running_fail_requested": True,
        "reason": reason,
        "min_age_hours": float(min_age_hours),
        "repaired": repaired,
        "skipped": skipped,
    }


def ensure_baseline_enqueued(study: optuna.Study, spaces: Mapping[str, Any], method_spec: Mapping[str, Any]) -> None:
    if study.trials:
        return
    study.enqueue_trial(baseline_params(spaces, method_spec))


def assert_baseline_ready_for_search(study: optuna.Study, method: str) -> None:
    trial0 = next((trial for trial in study.trials if trial.number == 0), None)
    if trial0 is None:
        raise RuntimeError(f"{method}: baseline trial 0 is missing; stop and recreate/audit the study.")
    if trial0.state == TrialState.COMPLETE:
        return
    if trial0.state == TrialState.WAITING and len(study.trials) == 1:
        return
    if trial0.state == TrialState.RUNNING:
        raise RuntimeError(f"{method}: baseline trial 0 is already running; avoid concurrent writers for this study.")
    raise RuntimeError(
        f"{method}: baseline trial 0 state is {trial0.state.name}; "
        "do not continue search before reproducing the frozen baseline."
    )


def run_method(method: str, args: argparse.Namespace, spaces: Mapping[str, Any]) -> dict[str, Any]:
    method_spec = spaces["methods"][method]
    storage_path = storage_path_for(spaces, method_spec)
    target = target_complete_trials(method_spec, args)
    if args.status and not args.fail_stale_running:
        if not storage_path.exists():
            return {
                "method": method,
                "study_group": spaces["study_group"],
                "study_name": method_spec["study_name"],
                "storage_path": str(storage_path),
                "tuning_stage": args.tuning_stage,
                "state_counts": {state.name: 0 for state in TrialState},
                "target_complete_trials": target,
                "remaining_to_target": target,
                "best_trial": None,
                "top_trials": [],
                "running_trials": [],
                "parameter_importance": {},
                "objective_distribution": {"min": None, "p25": None, "median": None, "p75": None, "max": None},
                "test_evaluation_count": 0,
                "status": "MISSING_STUDY",
            }
        storage, storage_path = storage_for(spaces, method_spec)
        study = load_existing_study(storage, method_spec, storage_path)
        if study is None:
            return {
                "method": method,
                "study_group": spaces["study_group"],
                "study_name": method_spec["study_name"],
                "storage_path": str(storage_path),
                "tuning_stage": args.tuning_stage,
                "state_counts": {state.name: 0 for state in TrialState},
                "target_complete_trials": target,
                "remaining_to_target": target,
                "best_trial": None,
                "top_trials": [],
                "running_trials": [],
                "parameter_importance": {},
                "objective_distribution": {"min": None, "p25": None, "median": None, "p75": None, "max": None},
                "test_evaluation_count": 0,
                "status": "MISSING_STUDY",
            }
        stale = fail_heartbeat_stale_trials(study)
        return summary_from_study(
            method,
            spaces,
            method_spec,
            study,
            target=target,
            tuning_stage=args.tuning_stage,
            storage_path=storage_path,
            stale_handling=stale,
        )

    storage, storage_path = storage_for(spaces, method_spec)
    sampler = optuna.samplers.TPESampler(seed=int(spaces["training_protocol"]["tuning_seed"]))
    study = optuna.create_study(
        study_name=str(method_spec["study_name"]),
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    stale: dict[str, Any] = {"heartbeat": fail_heartbeat_stale_trials(study)}
    if args.fail_stale_running:
        stale["manual"] = mark_running_trials_failed(
            study,
            min_age_hours=float(args.stale_min_age_hours),
            reason=str(args.stale_reason),
        )
        study = optuna.load_study(study_name=str(method_spec["study_name"]), storage=storage)
        if args.status:
            return summary_from_study(
                method,
                spaces,
                method_spec,
                study,
                target=target,
                tuning_stage=args.tuning_stage,
                storage_path=storage_path,
                stale_handling=stale,
            )
    ensure_baseline_enqueued(study, spaces, method_spec)
    assert_baseline_ready_for_search(study, method)

    remaining = max(target - complete_count(study), 0)
    if args.n_trials is not None:
        remaining = min(remaining, int(args.n_trials))
    if args.summary_only:
        remaining = 0
    worker_started = time.monotonic()
    worker_stop: dict[str, Any] = {}

    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": method,
                    "study": method_spec["study_name"],
                    "storage": str(storage_path),
                    "tuning_stage": args.tuning_stage,
                    "target_complete": target,
                    "complete": complete_count(study),
                    "would_run_trials": remaining,
                    "baseline_params": baseline_params(spaces, method_spec),
                    "runtime_estimate_sec": runtime_estimate_sec(study, method_spec, args),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif remaining > 0:
        trials_started = 0
        while trials_started < remaining and complete_count(study) < target:
            allowed, walltime_decision = enough_time_for_next_trial(worker_started, study, method_spec, args)
            if not allowed:
                worker_stop = walltime_decision
                break
            study.optimize(
                lambda trial: run_trial(
                    trial,
                    method=method,
                    spaces=spaces,
                    method_spec=method_spec,
                    allow_cpu=bool(args.allow_cpu),
                ),
                n_trials=1,
                gc_after_trial=True,
            )
            trials_started += 1
        if not worker_stop:
            worker_stop = {
                "walltime_aware": args.max_worker_runtime_sec is not None,
                "reason": "target_or_n_trials_reached",
                "trials_started": trials_started,
            }

    summary = summary_from_study(
        method,
        spaces,
        method_spec,
        study,
        target=target,
        tuning_stage=args.tuning_stage,
        storage_path=storage_path,
        stale_handling=stale,
        worker_stop=worker_stop,
    )
    summary_path = project_path(spaces["storage"]["summary_root"]) / f"{method_spec['study_name']}_summary.json"
    save_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    spaces = load_yaml(Path(args.spaces))
    assert_tuning_guardrails(spaces)
    summaries = [run_method(method, args, spaces) for method in method_list(args.method)]
    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2, allow_nan=False, default=str), flush=True)


if __name__ == "__main__":
    main()
