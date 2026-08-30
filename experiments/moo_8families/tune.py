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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default=str(DEFAULT_SPACES))
    parser.add_argument("--method", choices=(*METHODS, "all"), required=True)
    parser.add_argument("--target-complete", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
        "artifact_dir": root,
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


def state_counts(study: optuna.Study) -> dict[str, int]:
    counts = {state.name: 0 for state in TrialState}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    return counts


def summary_from_study(method: str, spaces: Mapping[str, Any], method_spec: Mapping[str, Any], study: optuna.Study) -> dict[str, Any]:
    complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]
    ranked = sorted(complete, key=lambda item: float(item.value), reverse=True)
    summaries = [trial.user_attrs.get("summary") or {"trial": trial.number, "NDCG@10": trial.value, "params": trial.params} for trial in ranked]
    return {
        "method": method,
        "study_group": spaces["study_group"],
        "study_name": method_spec["study_name"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "state_counts": state_counts(study),
        "target_complete_trials": int(method_spec["target_complete_trials"]),
        "best_trial": summaries[0] if summaries else None,
        "top_trials": summaries[:10],
        "test_evaluation_count": 0,
    }


def storage_for(spaces: Mapping[str, Any], method_spec: Mapping[str, Any]) -> tuple[optuna.storages.RDBStorage, Path]:
    storage_path = project_path(spaces["storage"]["storage_root"]) / f"{method_spec['study_name']}.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(url=sqlite_url(storage_path))
    return storage, storage_path


def ensure_baseline_enqueued(study: optuna.Study, spaces: Mapping[str, Any], method_spec: Mapping[str, Any]) -> None:
    if study.trials:
        return
    study.enqueue_trial(baseline_params(spaces, method_spec))


def run_method(method: str, args: argparse.Namespace, spaces: Mapping[str, Any]) -> dict[str, Any]:
    method_spec = spaces["methods"][method]
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
    ensure_baseline_enqueued(study, spaces, method_spec)

    target = int(method_spec["target_complete_trials"] if args.target_complete is None else args.target_complete)
    remaining = max(target - complete_count(study), 0)
    if args.n_trials is not None:
        remaining = min(remaining, int(args.n_trials))
    if args.summary_only:
        remaining = 0

    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": method,
                    "study": method_spec["study_name"],
                    "storage": str(storage_path),
                    "target_complete": target,
                    "complete": complete_count(study),
                    "would_run_trials": remaining,
                    "baseline_params": baseline_params(spaces, method_spec),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif remaining > 0:
        study.optimize(
            lambda trial: run_trial(
                trial,
                method=method,
                spaces=spaces,
                method_spec=method_spec,
                allow_cpu=bool(args.allow_cpu),
            ),
            n_trials=remaining,
            gc_after_trial=True,
        )

    summary = summary_from_study(method, spaces, method_spec, study)
    summary["storage_path"] = str(storage_path)
    summary_path = project_path(spaces["storage"]["summary_root"]) / f"{method_spec['study_name']}_summary.json"
    save_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    spaces = load_yaml(Path(args.spaces))
    summaries = [run_method(method, args, spaces) for method in method_list(args.method)]
    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2, allow_nan=False, default=str), flush=True)


if __name__ == "__main__":
    main()
