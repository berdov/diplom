#!/usr/bin/env python
"""Train or smoke one MOO eight-family benchmark method.

Smoke mode is train-only and does not create validation/test dataloaders.
Sanity mode runs 5 validation-only epochs and still keeps test closed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from recbole.config import Config
from recbole.trainer import Trainer
from recbole.utils import init_seed


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from experiments.adaptive_multitask_tim4rec.methods.common import (  # noqa: E402
    ensure_finite_gradients,
    parameter_group_summary,
    shared_parameter_entries,
    task_gradient_vectors,
)
from experiments.moo_8families.evaluation.ranking import evaluate_validation_ranking  # noqa: E402
from experiments.moo_8families.evaluation.objectives import (  # noqa: E402
    gradient_diagnostics,
    scalar_loss_record,
    task_losses,
)
from experiments.moo_8families.evaluation.pareto import (  # noqa: E402
    EVAL_OBJECTIVE_ORDER,
    RANKING_OPERATING_POINT_ID,
    validation_summary_from_records,
)
from experiments.moo_8families.pareto_models.cosmos import COSMOSTiM4Rec  # noqa: E402
from experiments.moo_8families.pareto_models.palora import PaLoRATiM4Rec  # noqa: E402
from experiments.moo_8families.pareto_models.phn import PHNAdapterTiM4Rec  # noqa: E402
from experiments.moo_8families.strategies.base import (  # noqa: E402
    AUX_TARGETS,
    TASK_ORDER,
    finite_scalar_mapping,
    losses_to_vector,
    preference_tensor,
    round_list,
    sha256_file,
    sha256_json,
    tensor_to_float,
    weighted_sum,
)
from experiments.moo_8families.strategies.epo import ExactParetoPreferenceSolver  # noqa: E402
from experiments.moo_8families.strategies.famo import FAMO  # noqa: E402
from experiments.moo_8families.strategies.gradhv import DominatedHypervolume  # noqa: E402
from experiments.moo_8families.strategies.pcgrad_adapter import load_historical_pcgrad  # noqa: E402
from experiments.moo_8families.strategies.preferences import ContinuousPreferenceSampler  # noqa: E402
from experiments.moo_8families.strategies.stch import SmoothTchebycheffScalarizer  # noqa: E402
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS  # noqa: E402
from experiments.multitask_tim4rec.train import (  # noqa: E402
    EXPECTED_FINGERPRINT,
    EXPECTED_IDENTITY_HASH,
    all_gradient_check,
    count_parameters,
    evaluate_auxiliary,
)
from experiments.multitask_tim4rec_optuna.optuna_search import (  # noqa: E402
    build_config,
    create_loaders,
    load_data_bundle,
    load_yaml,
    optimizer_for_trial,
    pos_weight_tensors,
    project_path,
)
from experiments.multitask_tim4rec_optuna.run_locked_tuned import sampled_from_locked_params  # noqa: E402


EXPERIMENT_DIR = ROOT / "experiments" / "moo_8families"
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.yaml"
METHODS = ("stch", "famo", "pcgrad", "epo", "gradhv", "phn", "cosmos", "palora")
TRAIN_METHODS = ("stch", "famo", "epo", "gradhv", "phn", "cosmos", "palora")
CONDITIONAL_METHODS = ("phn", "cosmos", "palora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=("smoke", "sanity", "historical"), default="smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return tensor_to_float(value)
        return {"shape": list(value.shape), "norm": float(torch.linalg.vector_norm(value.detach().float()).cpu().item())}
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")


def write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=json_default) + "\n")


def git_value(args: list[str], default: str = "unknown") -> str:
    env_map = {
        ("rev-parse", "HEAD"): "MOO_GIT_COMMIT",
        ("rev-parse", "--abbrev-ref", "HEAD"): "MOO_GIT_BRANCH",
        ("config", "--get", "remote.origin.url"): "MOO_GIT_REMOTE",
    }
    env_key = env_map.get(tuple(args))
    if env_key and os.environ.get(env_key):
        return str(os.environ[env_key])
    try:
        value = subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return value or default
    except Exception:
        return default


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "recbole": version("recbole"),
        "optuna": version("optuna"),
        "mamba_ssm": version("mamba-ssm"),
        "causal_conv1d": version("causal-conv1d"),
        "numpy": version("numpy"),
        "pyyaml": yaml.__version__,
    }


def slurm_info() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "constraint": os.environ.get("SLURM_JOB_CONSTRAINT"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "hostname": socket.gethostname(),
    }


def gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "device": "cpu"}
    current = torch.cuda.current_device()
    return {
        "available": True,
        "device": "cuda",
        "name": torch.cuda.get_device_name(current),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(current))),
        "device_count": torch.cuda.device_count(),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def source_hashes() -> dict[str, str]:
    relative_paths = [
        "experiments/moo_8families/config.yaml",
        "experiments/moo_8families/preferences.yaml",
        "experiments/moo_8families/train.py",
        "experiments/moo_8families/smoke_test.py",
        "experiments/moo_8families/run_benchmark.py",
        "experiments/moo_8families/build_results.py",
        "experiments/moo_8families/evaluation/objectives.py",
        "experiments/moo_8families/evaluation/pareto.py",
        "experiments/moo_8families/evaluation/ranking.py",
        "experiments/moo_8families/strategies/base.py",
        "experiments/moo_8families/strategies/stch.py",
        "experiments/moo_8families/strategies/famo.py",
        "experiments/moo_8families/strategies/epo.py",
        "experiments/moo_8families/strategies/gradhv.py",
        "experiments/moo_8families/strategies/pcgrad_adapter.py",
        "experiments/moo_8families/strategies/preferences.py",
        "experiments/moo_8families/pareto_models/phn.py",
        "experiments/moo_8families/pareto_models/cosmos.py",
        "experiments/moo_8families/pareto_models/palora.py",
        "experiments/multitask_tim4rec_optuna/optuna_search.py",
        "experiments/multitask_tim4rec_optuna/run_locked_tuned.py",
        "experiments/multitask_tim4rec_optuna/prepare_validation_only.py",
        "slurm/moo_8families.sh",
    ]
    return {
        path: sha256_file(ROOT / path)
        for path in relative_paths
        if (ROOT / path).exists()
    }


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_preferences(path: Path) -> dict[str, Any]:
    payload = load_yaml_file(path)
    if payload["objective_order"] != list(TASK_ORDER):
        raise RuntimeError(f"Preference task order mismatch: {payload['objective_order']}")
    for pref_id, spec in payload["preferences"].items():
        pref = torch.tensor(spec["weights"], dtype=torch.float32)
        normalized = preference_tensor(pref)
        if abs(float(normalized.sum().item()) - 1.0) > 1e-6:
            raise RuntimeError(f"Preference does not sum to one after normalization: {pref_id}")
    return payload


def method_default_run_id(config: Mapping[str, Any], method: str, stage: str) -> str:
    if method == "pcgrad":
        return str(config["methods"]["pcgrad"]["historical_run_id"])
    key = "smoke_run_id" if stage == "smoke" else "sanity_run_id"
    return str(config["methods"][method][key])


def resolve_paths(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[str, Path, Path, Path]:
    run_id = args.run_id or method_default_run_id(config, args.method, args.stage)
    local_runs_dir = project_path(config["run"]["local_runs_dir"])
    result_json = Path(args.result_json) if args.result_json else local_runs_dir / f"{run_id}.json"
    notes = Path(args.notes) if args.notes else local_runs_dir / f"{run_id}_notes.md"
    artifact_root = Path(args.artifact_dir) if args.artifact_dir else Path(config["run"]["artifact_root"]) / run_id
    return run_id, artifact_root, result_json, notes


def assert_output_allowed(paths: Sequence[Path], artifact_dir: Path, allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    for path in paths:
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing artifact: {path}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty artifact dir: {artifact_dir}")


def build_run_config(optuna_config: Mapping[str, Any], artifact_root: Path, sampled: Mapping[str, Any], epochs: int) -> Config:
    overrides = dict(optuna_config["recbole_overrides"])
    overrides.update(
        {
            "checkpoint_dir": str(artifact_root / "recbole_checkpoints"),
            "epochs": int(epochs),
            "stopping_step": int(epochs) + 1,
            "final_test_evaluation_count": 0,
            "test_evaluation_count": 0,
            "learning_rate": float(sampled["learning_rate"]),
            "weight_decay": float(sampled["weight_decay"]),
            "dropout_prob": float(sampled["dropout_prob"]),
            "metrics": ["Hit", "Recall", "NDCG"],
            "topk": [5, 10, 20, 50],
            "valid_metric": "NDCG@10",
            "show_progress": False,
            "log_wandb": False,
        }
    )
    return Config(
        model=MultitaskTiM4Rec,
        config_file_list=[str(project_path(optuna_config["source"]["base_config"]))],
        config_dict=overrides,
    )


def assert_protocol_guards(config: Mapping[str, Any], data: Any, recbole_config: Any) -> None:
    observed = data.validation_only_summary.get("protocol_fingerprint") or data.validation_only_summary.get("dataset_fingerprint")
    expected = {
        "users": int(config["protocol"]["users"]),
        "items": int(config["protocol"]["items"]),
        "interactions": int(config["protocol"]["interactions"]),
        "train": int(config["protocol"]["train"]),
        "validation": int(config["protocol"]["validation"]),
        "test": int(config["protocol"]["test"]),
    }
    if observed != expected:
        raise RuntimeError(f"Protocol B fingerprint mismatch: observed={observed}, expected={expected}")
    identity = data.validation_only_summary.get("identity_hash")
    if identity != str(config["protocol"]["identity_hash"]):
        raise RuntimeError(f"Identity hash mismatch: {identity}")
    if identity != EXPECTED_IDENTITY_HASH:
        raise RuntimeError(f"Code identity hash mismatch: {identity} != {EXPECTED_IDENTITY_HASH}")
    if EXPECTED_FINGERPRINT != expected:
        raise RuntimeError(f"Code fingerprint mismatch: {EXPECTED_FINGERPRINT} != {expected}")
    if tuple(recbole_config["multitask_targets"]) != TARGETS:
        raise RuntimeError(f"Task set changed: {recbole_config['multitask_targets']}")
    if not bool(recbole_config["is_time"]):
        raise RuntimeError("TiM4Rec is_time must remain True.")
    summary = data.validation_only_summary
    if summary.get("forbidden_test_paths_loaded") != []:
        raise RuntimeError(f"Validation-only prep touched test paths: {summary}")
    if bool(summary.get("test_path_passed_to_search")):
        raise RuntimeError(f"Test path passed to validation-only prep: {summary}")
    if int(summary["rows"]["test"]) != 0 or int(summary["test_rows_in_inter_file"]) != 0:
        raise RuntimeError(f"Validation-only RecBole data contains test rows: {summary['rows']}")


def preference_by_id(preferences: Mapping[str, Any], pref_id: str) -> list[float]:
    return [float(value) for value in preferences["preferences"][pref_id]["weights"]]


def preference_records(preferences: Mapping[str, Any], set_id: str) -> list[dict[str, Any]]:
    return [
        {"id": pref_id, "weights": preference_by_id(preferences, pref_id)}
        for pref_id in preferences["sets"][set_id]
    ]


def evaluation_reference_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    reference_config = dict(config["evaluation"]["pareto_reference"])
    objective_order = list(reference_config["objective_order"])
    if objective_order != list(EVAL_OBJECTIVE_ORDER):
        raise RuntimeError(f"Evaluation objective order mismatch: {objective_order} != {list(EVAL_OBJECTIVE_ORDER)}")
    values = [float(value) for value in reference_config["values"]]
    if len(values) != len(EVAL_OBJECTIVE_ORDER):
        raise RuntimeError(f"Evaluation reference must have {len(EVAL_OBJECTIVE_ORDER)} values, got {values}")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"Evaluation reference contains non-finite values: {values}")
    return {
        "objective_order": objective_order,
        "values": values,
        "source": reference_config.get("source"),
        "source_run_id": reference_config.get("source_run_id"),
        "source_json": reference_config.get("source_json"),
        "control_point": reference_config.get("control_point"),
        "margins": reference_config.get("margins"),
        "invalid_reference_policy": reference_config.get("invalid_reference_policy", "raise"),
        "frozen_before_moo_sanity_results": bool(reference_config.get("frozen_before_moo_sanity_results")),
    }


def set_model_preference(model: Any, preference: Sequence[float] | None) -> None:
    if preference is not None and hasattr(model, "set_preference"):
        model.set_preference(preference)


def new_model(method: str, recbole_config: Any, train_dataset: Any, method_config: Mapping[str, Any]) -> Any:
    if method in {"stch", "famo", "epo", "gradhv"}:
        return MultitaskTiM4Rec(recbole_config, train_dataset).to(recbole_config["device"])
    if method == "phn":
        return PHNAdapterTiM4Rec(
            recbole_config,
            train_dataset,
            adapter_hidden_size=int(method_config["adapter_hidden_size"]),
            adapter_scale=float(method_config["adapter_scale"]),
        ).to(recbole_config["device"])
    if method == "cosmos":
        return COSMOSTiM4Rec(
            recbole_config,
            train_dataset,
            preference_hidden_size=int(method_config["preference_hidden_size"]),
        ).to(recbole_config["device"])
    if method == "palora":
        return PaLoRATiM4Rec(
            recbole_config,
            train_dataset,
            rank=int(method_config["rank"]),
            alpha=float(method_config["alpha"]),
            target_modules=list(method_config["target_modules"]),
        ).to(recbole_config["device"])
    raise RuntimeError(f"Unsupported training method: {method}")


def normalized_task_map(vector: Any) -> dict[str, Any]:
    return {task: vector[index] for index, task in enumerate(TASK_ORDER)}


def check_backward(model: Any, shared_entries: list[Any]) -> dict[str, Any]:
    finite = all_gradient_check(model)
    shared = ensure_finite_gradients(shared_entries)
    if not bool(finite["all_finite"]) or not bool(shared["all_finite"]):
        raise RuntimeError(f"Non-finite gradients: model={finite}, shared={shared}")
    return {"model": finite, "shared": shared}


def interaction_from_batch(batch: Any) -> Any:
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def preference_sensitivity_diagnostic(
    *,
    model: Any,
    train_data: Any,
    preferences: Mapping[str, Any],
    p1_id: str,
    p2_id: str,
    tolerance: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    p1 = preference_by_id(preferences, p1_id)
    p2 = preference_by_id(preferences, p2_id)
    try:
        batch = next(iter(train_data))
    except StopIteration as exc:
        raise RuntimeError("Cannot run preference sensitivity diagnostic: train loader is empty.") from exc
    interaction = interaction_from_batch(batch).to(device)

    model.eval()
    with torch.no_grad():
        set_model_preference(model, p1)
        representation_1 = model.shared_representation(interaction).detach().float()
        logits_1 = model.ranking_logits_from_representation(representation_1)
        set_model_preference(model, p2)
        representation_2 = model.shared_representation(interaction).detach().float()
        logits_2 = model.ranking_logits_from_representation(representation_2)

        if getattr(model, "POS_ITEM_ID", None) in interaction.interaction:
            item_ids = interaction[model.POS_ITEM_ID].long()
            score_mode = "positive_item_logits"
            scores_1 = logits_1.gather(1, item_ids.view(-1, 1)).squeeze(1)
            scores_2 = logits_2.gather(1, item_ids.view(-1, 1)).squeeze(1)
        elif getattr(model, "ITEM_ID", None) in interaction.interaction:
            item_ids = interaction[model.ITEM_ID].long()
            score_mode = "candidate_item_logits"
            scores_1 = logits_1.gather(1, item_ids.view(-1, 1)).squeeze(1)
            scores_2 = logits_2.gather(1, item_ids.view(-1, 1)).squeeze(1)
        else:
            score_mode = "full_ranking_logits"
            scores_1 = logits_1
            scores_2 = logits_2

    representation_delta = representation_1 - representation_2
    score_delta = scores_1 - scores_2
    representation_l2 = float(torch.linalg.vector_norm(representation_delta).cpu().item())
    representation_mean_abs = float(representation_delta.abs().mean().cpu().item())
    score_l2 = float(torch.linalg.vector_norm(score_delta).cpu().item())
    score_mean_abs = float(score_delta.abs().mean().cpu().item())
    output_metric_passed = bool(score_mean_abs > float(tolerance) or score_l2 > float(tolerance))
    return {
        "split": "train",
        "batch_source": "first_train_batch_after_smoke_training",
        "batch_examples": int(len(interaction)),
        "p1_id": p1_id,
        "p1": p1,
        "p2_id": p2_id,
        "p2": p2,
        "representation_l2": representation_l2,
        "representation_mean_abs": representation_mean_abs,
        "ranking_score_l2": score_l2,
        "ranking_score_mean_abs": score_mean_abs,
        "output_metric_name": "ranking_score_mean_abs_or_l2",
        "ranking_score_mode": score_mode,
        "tolerance": float(tolerance),
        "output_metric_passed": output_metric_passed,
    }


def compute_normalization_diagnostics(
    model: Any,
    train_data: Any,
    *,
    sampled: Mapping[str, Any],
    pos_weights: Mapping[str, Any],
    batches: int,
    selector: str,
) -> dict[str, Any]:
    model.train()
    sums = {task: 0.0 for task in TASK_ORDER}
    sums_sq = {task: 0.0 for task in TASK_ORDER}
    grad_norm_sums = {task: 0.0 for task in TASK_ORDER}
    cosine_sums: dict[str, dict[str, float]] = {left: {right: 0.0 for right in TASK_ORDER} for left in TASK_ORDER}
    cosine_counts: dict[str, dict[str, int]] = {left: {right: 0 for right in TASK_ORDER} for left in TASK_ORDER}
    pos_sums = {target: 0.0 for target in AUX_TARGETS}
    examples = 0
    seen_batches = 0
    first_gradient_diagnostic = None

    for interaction in train_data:
        interaction = interaction.to(model.device if hasattr(model, "device") else next(model.parameters()).device)
        batch_size = len(interaction)
        losses = task_losses(model, interaction, sampled, pos_weights, loss_scales=None)
        vector = losses_to_vector(losses)
        for index, task in enumerate(TASK_ORDER):
            value = tensor_to_float(vector[index])
            sums[task] += value
            sums_sq[task] += value * value
        for target in AUX_TARGETS:
            pos_sums[target] += tensor_to_float(interaction[target].float().mean()) * batch_size
        diag = gradient_diagnostics(model, losses, selector=selector)
        if first_gradient_diagnostic is None:
            first_gradient_diagnostic = diag
        for task, norm in diag["gradient_norms"].items():
            grad_norm_sums[task] += float(norm)
        for left, row in diag["cosine_matrix"].items():
            for right, value in row.items():
                if value is None:
                    continue
                cosine_sums[left][right] += float(value)
                cosine_counts[left][right] += 1
        examples += batch_size
        seen_batches += 1
        model.zero_grad(set_to_none=True)
        if seen_batches >= batches:
            break

    if seen_batches == 0:
        raise RuntimeError("No train batches for normalization diagnostics.")
    mean_losses = {task: sums[task] / seen_batches for task in TASK_ORDER}
    std_losses = {}
    for task in TASK_ORDER:
        mean = mean_losses[task]
        variance = max(sums_sq[task] / seen_batches - mean * mean, 0.0)
        std_losses[task] = math.sqrt(variance)
    mean_grad_norms = {task: grad_norm_sums[task] / seen_batches for task in TASK_ORDER}
    mean_cosines = {
        left: {
            right: (cosine_sums[left][right] / cosine_counts[left][right] if cosine_counts[left][right] else None)
            for right in TASK_ORDER
        }
        for left in TASK_ORDER
    }
    loss_scales = [max(mean_losses[task], 1e-8) for task in TASK_ORDER]
    reference_point = [1.5 for _task in TASK_ORDER]
    return {
        "batches": seen_batches,
        "examples": examples,
        "task_order": list(TASK_ORDER),
        "mean_loss": mean_losses,
        "std_loss": std_losses,
        "loss_scales": loss_scales,
        "mean_gradient_norm": mean_grad_norms,
        "mean_pairwise_cosine": mean_cosines,
        "positive_rate": {target: pos_sums[target] / max(examples, 1) for target in AUX_TARGETS},
        "effective_pos_weights": dict(sampled["effective_pos_weights"]),
        "reference_point_normalized": reference_point,
        "first_gradient_diagnostic": first_gradient_diagnostic,
        "split": "train",
        "test_access": "none",
    }


def train_single_epoch(
    *,
    method: str,
    model: Any,
    optimizer: Any,
    train_data: Any,
    sampled: Mapping[str, Any],
    pos_weights: Mapping[str, Any],
    loss_scales: Sequence[float],
    method_state: Any,
    max_batches: int | None,
    epoch: int,
    preferences: Mapping[str, Any],
    method_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model.train()
    shared_entries = shared_parameter_entries(model, "all_backbone")
    sums: dict[str, float] = {}
    examples = 0
    batches = 0
    first_diag = None
    last_method_state = None
    preference = None
    if method == "stch":
        preference = preference_by_id(preferences, str(method_config["preference_id"]))

    for batch_idx, interaction in enumerate(train_data):
        interaction = interaction.to(next(model.parameters()).device)
        batch_size = len(interaction)
        optimizer.zero_grad(set_to_none=True)
        if method == "stch":
            set_model_preference(model, preference)
            losses = task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
            scalar = method_state.scalarize(losses["normalized_task_vector"])
            scalar.backward()
            last_method_state = method_state.state_dict()
        elif method == "famo":
            losses = task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
            scalar = method_state.get_weighted_loss(losses["normalized_task_vector"])
            scalar.backward()
        else:
            raise RuntimeError(f"Unsupported single method: {method}")
        if not torch.isfinite(scalar):
            raise RuntimeError(f"Non-finite scalar loss for {method}: {tensor_to_float(scalar)}")
        check_backward(model, shared_entries)
        optimizer.step()
        if method == "famo":
            with torch.no_grad():
                refreshed = task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
            update = method_state.update(refreshed["normalized_task_vector"])
            last_method_state = method_state.state_dict() | {"last_effective_weights": update.effective_weights}
        for key, value in scalar_loss_record(losses).items():
            sums[key] = sums.get(key, 0.0) + value * batch_size
        sums["moo_scalar"] = sums.get("moo_scalar", 0.0) + tensor_to_float(scalar) * batch_size
        examples += batch_size
        batches += 1
        if first_diag is None:
            fresh = task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
            first_diag = gradient_diagnostics(model, fresh, selector="all_backbone")
        if max_batches is not None and batches >= max_batches:
            break
    return summarize_epoch(sums, examples, batches) | {"method_state": last_method_state}, first_diag


def train_epo_epoch(
    *,
    models: list[Any],
    optimizers: list[Any],
    solvers: list[ExactParetoPreferenceSolver],
    train_data: Any,
    sampled: Mapping[str, Any],
    pos_weights: Mapping[str, Any],
    loss_scales: Sequence[float],
    max_batches: int | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sums: dict[str, float] = {}
    solution_sums: list[dict[str, float]] = [{} for _ in models]
    examples = 0
    batches = 0
    first_diag = None
    last_solver_states = []

    for interaction in train_data:
        interaction = interaction.to(next(models[0].parameters()).device)
        batch_size = len(interaction)
        for model, optimizer, solver, sol_sums in zip(models, optimizers, solvers, solution_sums):
            model.train()
            shared_entries = shared_parameter_entries(model, "all_backbone")
            optimizer.zero_grad(set_to_none=True)
            probe = task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
            vectors = task_gradient_vectors(normalized_task_map(probe["normalized_task_vector"]), shared_entries, TASK_ORDER)
            gradient_matrix = torch.stack([vectors[task] for task in TASK_ORDER], dim=0)
            alpha = solver.alpha(probe["normalized_task_vector"].detach(), gradient_matrix)
            optimizer.zero_grad(set_to_none=True)
            losses = task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
            scalar = (alpha.detach() * losses["normalized_task_vector"]).sum()
            scalar.backward()
            if not torch.isfinite(scalar):
                raise RuntimeError(f"Non-finite EPO scalar loss: {tensor_to_float(scalar)}")
            check_backward(model, shared_entries)
            optimizer.step()
            for key, value in scalar_loss_record(losses).items():
                sol_sums[key] = sol_sums.get(key, 0.0) + value * batch_size
                sums[key] = sums.get(key, 0.0) + value * batch_size
            sol_sums["moo_scalar"] = sol_sums.get("moo_scalar", 0.0) + tensor_to_float(scalar) * batch_size
            sums["moo_scalar"] = sums.get("moo_scalar", 0.0) + tensor_to_float(scalar) * batch_size
            if first_diag is None:
                fresh = task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
                first_diag = gradient_diagnostics(model, fresh, selector="all_backbone") | {"epo": solver.state_dict()}
        examples += batch_size
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    denom = max(examples * len(models), 1)
    aggregate = {key: value / denom for key, value in sums.items()}
    aggregate["batches"] = batches
    aggregate["examples"] = examples
    aggregate["solutions"] = [
        summarize_epoch(solution_sums[idx], examples, batches) | {"epo": solvers[idx].state_dict()}
        for idx in range(len(models))
    ]
    last_solver_states = [solver.state_dict() for solver in solvers]
    aggregate["method_state"] = {"solvers": last_solver_states}
    return aggregate, first_diag


def train_gradhv_epoch(
    *,
    models: list[Any],
    optimizers: list[Any],
    hv: DominatedHypervolume,
    train_data: Any,
    sampled: Mapping[str, Any],
    pos_weights: Mapping[str, Any],
    loss_scales: Sequence[float],
    max_batches: int | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    for model in models:
        model.train()
    sums: dict[str, float] = {}
    solution_sums: list[dict[str, float]] = [{} for _ in models]
    examples = 0
    batches = 0
    first_diag = None

    for interaction in train_data:
        interaction = interaction.to(next(models[0].parameters()).device)
        batch_size = len(interaction)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses_by_solution = [
            task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
            for model in models
        ]
        points = torch.stack([losses["normalized_task_vector"] for losses in losses_by_solution], dim=0)
        scalar = hv.loss(points)
        if not torch.isfinite(scalar):
            raise RuntimeError(f"Non-finite GradHV scalar loss: {tensor_to_float(scalar)}")
        scalar.backward()
        for model in models:
            shared_entries = shared_parameter_entries(model, "all_backbone")
            check_backward(model, shared_entries)
        for optimizer in optimizers:
            optimizer.step()
        for idx, losses in enumerate(losses_by_solution):
            for key, value in scalar_loss_record(losses).items():
                solution_sums[idx][key] = solution_sums[idx].get(key, 0.0) + value * batch_size
                sums[key] = sums.get(key, 0.0) + value * batch_size
            solution_sums[idx]["moo_scalar"] = solution_sums[idx].get("moo_scalar", 0.0) + tensor_to_float(scalar) * batch_size
        sums["moo_scalar"] = sums.get("moo_scalar", 0.0) + tensor_to_float(scalar) * batch_size
        if first_diag is None:
            fresh = task_losses(models[0], interaction, sampled, pos_weights, loss_scales=loss_scales)
            first_diag = gradient_diagnostics(models[0], fresh, selector="all_backbone") | {"gradhv": hv.state_dict()}
        examples += batch_size
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    denom = max(examples * len(models), 1)
    aggregate = {key: value / denom for key, value in sums.items()}
    aggregate["batches"] = batches
    aggregate["examples"] = examples
    aggregate["solutions"] = [summarize_epoch(sol_sums, examples, batches) for sol_sums in solution_sums]
    aggregate["method_state"] = hv.state_dict()
    return aggregate, first_diag


def train_conditional_epoch(
    *,
    method: str,
    model: Any,
    optimizer: Any,
    train_data: Any,
    sampled: Mapping[str, Any],
    pos_weights: Mapping[str, Any],
    loss_scales: Sequence[float],
    preference_sampler: ContinuousPreferenceSampler,
    method_config: Mapping[str, Any],
    max_batches: int | None,
    epoch: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model.train()
    shared_entries = shared_parameter_entries(model, "all_backbone")
    sums: dict[str, float] = {}
    examples = 0
    batches = 0
    first_diag = None
    sampled_preferences: list[list[float]] = []

    for batch_idx, interaction in enumerate(train_data):
        del batch_idx
        pref_tensor = preference_sampler.sample_tensor(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        preference = [tensor_to_float(value) for value in pref_tensor]
        sampled_preferences.append(preference)
        set_model_preference(model, preference)
        interaction = interaction.to(next(model.parameters()).device)
        batch_size = len(interaction)
        optimizer.zero_grad(set_to_none=True)
        losses = task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
        if method == "cosmos":
            scalar = model.cosmos_regularized_loss(
                losses["normalized_task_vector"],
                preference,
                lambda_cosine=float(method_config["lambda_cosine"]),
            )
        else:
            scalar = weighted_sum(losses["normalized_task_vector"], preference)
        if not torch.isfinite(scalar):
            raise RuntimeError(f"Non-finite {method} scalar loss: {tensor_to_float(scalar)}")
        scalar.backward()
        check_backward(model, shared_entries)
        optimizer.step()
        for key, value in scalar_loss_record(losses).items():
            sums[key] = sums.get(key, 0.0) + value * batch_size
        sums["moo_scalar"] = sums.get("moo_scalar", 0.0) + tensor_to_float(scalar) * batch_size
        examples += batch_size
        batches += 1
        if first_diag is None:
            fresh = task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)
            first_diag = gradient_diagnostics(model, fresh, selector="all_backbone") | {
                "preference_id": "dirichlet_sample",
                "preference": preference,
            }
        if max_batches is not None and batches >= max_batches:
            break
    summary = summarize_epoch(sums, examples, batches)
    summary["sampled_preferences_head"] = sampled_preferences[:5]
    summary["preference_sampling"] = preference_sampler.diagnostics()
    if hasattr(model, "extra_parameter_summary"):
        summary["method_state"] = model.extra_parameter_summary()
    return summary, first_diag


def summarize_epoch(sums: Mapping[str, float], examples: int, batches: int) -> dict[str, Any]:
    if examples <= 0:
        raise RuntimeError("No training examples.")
    result = {key: float(value) / examples for key, value in sums.items()}
    result["batches"] = int(batches)
    result["examples"] = int(examples)
    if "rank" in result:
        rank = result["rank"]
        result["auxiliary_rank_ratio"] = None if rank == 0 else result.get("weighted_aux_sum", 0.0) / rank
    return result


def save_checkpoint(path: Path, models: list[Any], optimizers: list[Any], epoch: int, best_score: float | None, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_valid_score": best_score,
        "models": [{key: value.detach().cpu() for key, value in model.state_dict().items()} for model in models],
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "metadata": dict(metadata),
    }
    torch.save(payload, path, pickle_protocol=4)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def evaluate_models(
    *,
    method: str,
    models: list[Any],
    trainers: list[Any],
    valid_data: Any,
    train_data: Any,
    topk: Sequence[int],
    pareto_reference_point: Sequence[float],
    preferences: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    records = []
    if method in {"epo", "gradhv"}:
        for idx, model in enumerate(models):
            eval_result = evaluate_validation_ranking(
                trainer=trainers[idx],
                model=model,
                valid_data=valid_data,
                train_data=train_data,
                topk=topk,
            )
            aux = evaluate_auxiliary(model, valid_data, next(model.parameters()).device)
            records.append(
                {
                    "solution_index": idx,
                    "preference_id": None if preferences is None or idx >= len(preferences) else preferences[idx]["id"],
                    "preference": None if preferences is None or idx >= len(preferences) else preferences[idx]["weights"],
                    "metrics": eval_result["metrics"],
                    "auxiliary_validation": aux,
                    "checks": eval_result["checks"],
                }
            )
    elif preferences:
        model = models[0]
        trainer = trainers[0]
        for pref in preferences:
            eval_result = evaluate_validation_ranking(
                trainer=trainer,
                model=model,
                valid_data=valid_data,
                train_data=train_data,
                topk=topk,
                preference=pref["weights"],
            )
            aux = evaluate_auxiliary(model, valid_data, next(model.parameters()).device)
            records.append(
                {
                    "preference_id": pref["id"],
                    "preference": pref["weights"],
                    "metrics": eval_result["metrics"],
                    "auxiliary_validation": aux,
                    "checks": eval_result["checks"],
                }
            )
    else:
        model = models[0]
        trainer = trainers[0]
        eval_result = evaluate_validation_ranking(
            trainer=trainer,
            model=model,
            valid_data=valid_data,
            train_data=train_data,
            topk=topk,
        )
        aux = evaluate_auxiliary(model, valid_data, next(model.parameters()).device)
        records.append(
            {
                "solution_index": 0,
                "metrics": eval_result["metrics"],
                "auxiliary_validation": aux,
                "checks": eval_result["checks"],
            }
        )
    return validation_summary_from_records(
        records,
        method=method,
        reference_point=pareto_reference_point,
        ranking_preference_id=RANKING_OPERATING_POINT_ID,
    )


def build_notes(result: Mapping[str, Any]) -> str:
    lines = [
        f"# {result['run_id']}",
        "",
        "## Safety",
        "",
        "- `test_evaluation_count = 0`.",
        "- Test dataset не загружался, test dataloader не создавался.",
        f"- Stage: `{result['stage']}`.",
        f"- Method: `{result['method']['name']}`.",
        f"- Implementation: `{result['method'].get('implementation_name')}`.",
        f"- Representative fidelity: `{result['method'].get('representative_fidelity')}`.",
        f"- Exact method reproduction: `{result['method'].get('exact_method_reproduction')}`.",
        "",
        "## Dataset",
        "",
        f"- Protocol: `{result['dataset']['protocol']}`.",
        f"- Identity hash: `{result['dataset']['identity_hash_expected']}`.",
        "",
    ]
    if result["stage"] == "smoke":
        lines += [
            "## Smoke",
            "",
            f"- Train batches: `{result['training']['epochs'][0]['losses']['batches']}`.",
            f"- Mean scalar loss: `{result['training']['epochs'][0]['losses'].get('moo_scalar')}`.",
            "- Validation не запускалась.",
            "",
        ]
        if result.get("preference_sensitivity") is not None:
            sensitivity = result["preference_sensitivity"]
            lines += [
                "## Preference Sensitivity",
                "",
                f"- Split: `{sensitivity['split']}`.",
                f"- Preferences: `{sensitivity['p1_id']}` vs `{sensitivity['p2_id']}`.",
                f"- Representation L2: `{sensitivity['representation_l2']}`.",
                f"- Ranking score L2: `{sensitivity['ranking_score_l2']}`.",
                f"- Ranking score mean abs: `{sensitivity['ranking_score_mean_abs']}`.",
                f"- Passed: `{sensitivity['output_metric_passed']}`.",
                "",
            ]
    else:
        primary = result["validation"]["ranking_operating_point"]
        oracle = result["validation"]["oracle_best_validation_point"]
        metrics = primary["metrics"]
        lines += [
            "## Ranking Operating Point",
            "",
            f"- Selection: `{result['validation']['ranking_operating_point_selection']}`.",
            f"- Selection is validation oracle: `{result['validation']['selection_is_validation_oracle']}`.",
            "| HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {metrics['HR@5']:.4f} | {metrics['HR@10']:.4f} | {metrics['HR@20']:.4f} | {metrics['HR@50']:.4f} | "
            f"{metrics['NDCG@5']:.4f} | {metrics['NDCG@10']:.4f} | {metrics['NDCG@20']:.4f} | {metrics['NDCG@50']:.4f} |",
            "",
            "## Oracle Best Validation Point",
            "",
            f"- id: `{oracle.get('preference_id') or oracle.get('solution_index')}`.",
            f"- NDCG@10: `{oracle['metrics']['NDCG@10']:.4f}`.",
            "",
            "## Validation Points",
            "",
            "| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for record in result["validation"]["records"]:
            aux = record["auxiliary_validation"]
            row_id = record.get("preference_id", record.get("solution_index", 0))
            lines.append(
                f"| `{row_id}` | {record['metrics']['HR@10']:.4f} | {record['metrics']['NDCG@10']:.4f} | "
                f"{aux['is_click']['bce_loss']:.4f} | {aux['long_view']['bce_loss']:.4f} | "
                f"{aux['is_like']['bce_loss']:.4f} | {aux['is_profile_enter']['bce_loss']:.4f} |"
            )
        lines.append("")
    lines += [
        "## Cost",
        "",
        f"- Runtime sec: `{result['runtime']['total_sec']:.3f}`.",
        f"- Peak VRAM bytes: `{result['gpu'].get('peak_allocated_bytes')}`.",
        f"- Params: `{result['model_parameters']['total_trainable']}` trainable.",
        "",
    ]
    return "\n".join(lines)


def build_historical_pcgrad_result(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    run_id: str,
    result_json: Path,
    notes: Path,
) -> None:
    historical = load_historical_pcgrad(project_path(config["source"]["pcgrad_historical_json"]))
    evaluation_reference = evaluation_reference_from_config(config)
    validation_record = {
        "solution_index": 0,
        "metrics": historical["best_validation_metrics"],
        "auxiliary_validation": historical["best_auxiliary_metrics"],
        "preference_id": "historical_pcgrad",
    }
    validation_payload = validation_summary_from_records(
        [validation_record],
        method="pcgrad",
        reference_point=evaluation_reference["values"],
        ranking_preference_id=RANKING_OPERATING_POINT_ID,
    )
    result = {
        "run_id": run_id,
        "status": "completed",
        "stage": "historical",
        "record_type": "experiment_validation_only",
        "method": {
            "name": "PCGrad",
            "family": config["methods"]["pcgrad"]["family"],
            "representative": config["methods"]["pcgrad"]["representative"],
            "solution_type": "single",
        },
        "historical": historical,
        "evaluation": {
            "pareto_reference": evaluation_reference,
            "ranking_operating_point_id": RANKING_OPERATING_POINT_ID,
        },
        "validation": validation_payload,
        "test_safety": {
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
            "test_evaluated": False,
            "test_evaluation_count": 0,
        },
        "test_evaluation_count": 0,
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": git_value(["config", "--get", "remote.origin.url"]),
        },
        "source_files": source_hashes(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(result_json, result)
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(
        "\n".join(
            [
                f"# {run_id}",
                "",
                "PCGrad не перезапускался. Использован historical validation-only run `pcgrad_001`.",
                f"Best validation NDCG@10: `{historical['best_validation_metrics']['NDCG@10']}`.",
                "`test_evaluation_count=0`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    moo_config = load_yaml_file(Path(args.config))
    preferences = load_preferences(project_path(moo_config["source"]["preferences"]))
    evaluation_reference = evaluation_reference_from_config(moo_config)
    run_id, artifact_dir, result_json, notes = resolve_paths(args, moo_config)
    partial_json = result_json.with_suffix(".partial.json")
    assert_output_allowed([result_json, notes, partial_json], artifact_dir, args.allow_overwrite)

    if args.method == "pcgrad" or args.stage == "historical":
        if args.method != "pcgrad":
            raise RuntimeError("Historical stage is only valid for PCGrad.")
        build_historical_pcgrad_result(args=args, config=moo_config, run_id=run_id, result_json=result_json, notes=notes)
        return
    if args.method not in TRAIN_METHODS:
        raise RuntimeError(f"Method {args.method} is not trainable in this runner.")

    stage = args.stage
    method_config = moo_config["methods"][args.method]
    epochs = int(args.epochs or (1 if stage == "smoke" else moo_config["run"]["sanity_epochs"]))
    if stage == "sanity" and epochs != int(moo_config["run"]["sanity_epochs"]):
        raise RuntimeError(f"Sanity must run exactly {moo_config['run']['sanity_epochs']} epochs, got {epochs}")
    max_batches = args.max_batches
    if max_batches is None and stage == "smoke":
        max_batches = int(moo_config["run"]["smoke_batches"])

    artifact_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = artifact_dir / "training_log.jsonl"
    start = time.monotonic()
    run_started = datetime.now(timezone.utc)

    optuna_config = load_yaml(project_path(moo_config["source"]["optuna_config"]))
    best_params = load_yaml(project_path(moo_config["source"]["best_params"]))
    data = load_data_bundle(optuna_config, artifact_dir / "data_probe")
    sampled = sampled_from_locked_params(best_params, data.target_stats)
    recbole_config = build_run_config(optuna_config, artifact_dir, sampled, epochs)
    assert_protocol_guards(moo_config, data, recbole_config)
    init_seed(int(moo_config["run"]["seed"]) + recbole_config["local_rank"], recbole_config["reproducibility"])
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required; run on cHARISMa type_e or pass --allow-cpu only for debugging.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_data, valid_data = create_loaders(recbole_config, data.train_dataset, data.valid_dataset)
    device = recbole_config["device"]
    pos_weights = pos_weight_tensors(sampled["effective_pos_weights"], device)
    preference_sampler = None

    if args.method == "epo":
        pref_records = preference_records(preferences, method_config["preference_set"])
        models = []
        optimizers = []
        solvers = []
        for idx, pref in enumerate(pref_records):
            init_seed(int(moo_config["run"]["seed"]) + idx, recbole_config["reproducibility"])
            model = new_model(args.method, recbole_config, train_data.dataset, method_config)
            models.append(model)
            optimizers.append(optimizer_for_trial(model, sampled))
            solvers.append(
                ExactParetoPreferenceSolver(
                    pref["weights"],
                    alpha_multiplier=len(TASK_ORDER),
                )
            )
        validation_preferences = pref_records
        method_state = {"solvers": [solver.state_dict() for solver in solvers]}
    elif args.method == "gradhv":
        solution_count = int(method_config["solution_count"])
        models = []
        optimizers = []
        for idx in range(solution_count):
            init_seed(int(moo_config["run"]["seed"]) + idx, recbole_config["reproducibility"])
            model = new_model(args.method, recbole_config, train_data.dataset, method_config)
            models.append(model)
            optimizers.append(optimizer_for_trial(model, sampled))
        validation_preferences = None
        method_state = None
    else:
        model = new_model(args.method, recbole_config, train_data.dataset, method_config)
        models = [model]
        optimizers = [optimizer_for_trial(model, sampled)]
        if args.method == "stch":
            method_state = SmoothTchebycheffScalarizer(
                mu=float(method_config["mu"]),
                warmup_steps=max(len(train_data), 1) * int(method_config["warmup_epochs"]),
                preference=preference_by_id(preferences, str(method_config["preference_id"])),
            )
            validation_preferences = None
        elif args.method == "famo":
            method_state = FAMO(
                n_tasks=len(TASK_ORDER),
                device=device,
                gamma=float(method_config["gamma"]),
                w_lr=float(method_config["w_lr"]),
                max_norm=float(method_config["max_norm"]),
            )
            validation_preferences = None
        else:
            method_state = None
            validation_preferences = preference_records(preferences, method_config["eval_preference_set"])
            sampling_config = method_config["preference_sampling"]
            sampler_seed = int(moo_config["continuous_preference_sampling"]["seed"]) + TRAIN_METHODS.index(args.method)
            preference_sampler = ContinuousPreferenceSampler(
                alpha=float(sampling_config["alpha"]),
                seed=sampler_seed,
                coverage_threshold=float(moo_config["continuous_preference_sampling"]["coverage_threshold"]),
            )

    diagnostic_model = models[0]
    normalization = compute_normalization_diagnostics(
        diagnostic_model,
        train_data,
        sampled=sampled,
        pos_weights=pos_weights,
        batches=int(moo_config["normalization"]["diagnostic_batches"]),
        selector=str(moo_config["normalization"]["gradient_selector"]),
    )
    loss_scales = [float(value) for value in normalization["loss_scales"]]
    if args.method == "gradhv":
        method_state = DominatedHypervolume(normalization["reference_point_normalized"])

    trainers = [Trainer(recbole_config, model) for model in models]
    for trainer, optimizer in zip(trainers, optimizers):
        trainer.optimizer = optimizer

    epochs_payload = []
    diagnostics = []
    best_validation = None
    best_epoch = None
    best_score = -float("inf")
    best_checkpoint = None
    last_checkpoint = None
    train_start = time.monotonic()

    for epoch in range(1, epochs + 1):
        epoch_start = time.monotonic()
        if args.method in {"stch", "famo"}:
            losses, first_diag = train_single_epoch(
                method=args.method,
                model=models[0],
                optimizer=optimizers[0],
                train_data=train_data,
                sampled=sampled,
                pos_weights=pos_weights,
                loss_scales=loss_scales,
                method_state=method_state,
                max_batches=max_batches,
                epoch=epoch,
                preferences=preferences,
                method_config=method_config,
            )
        elif args.method == "epo":
            losses, first_diag = train_epo_epoch(
                models=models,
                optimizers=optimizers,
                solvers=solvers,
                train_data=train_data,
                sampled=sampled,
                pos_weights=pos_weights,
                loss_scales=loss_scales,
                max_batches=max_batches,
            )
        elif args.method == "gradhv":
            losses, first_diag = train_gradhv_epoch(
                models=models,
                optimizers=optimizers,
                hv=method_state,
                train_data=train_data,
                sampled=sampled,
                pos_weights=pos_weights,
                loss_scales=loss_scales,
                max_batches=max_batches,
            )
        else:
            if preference_sampler is None:
                raise RuntimeError(f"Continuous preference sampler was not initialized for {args.method}")
            losses, first_diag = train_conditional_epoch(
                method=args.method,
                model=models[0],
                optimizer=optimizers[0],
                train_data=train_data,
                sampled=sampled,
                pos_weights=pos_weights,
                loss_scales=loss_scales,
                preference_sampler=preference_sampler,
                method_config=method_config,
                max_batches=max_batches,
                epoch=epoch,
            )
        train_time = time.monotonic() - epoch_start
        if first_diag is not None:
            diagnostics.append({"epoch": epoch, **first_diag})

        validation_payload = None
        if stage == "sanity":
            valid_start = time.monotonic()
            validation_payload = evaluate_models(
                method=args.method,
                models=models,
                trainers=trainers,
                valid_data=valid_data,
                train_data=train_data,
                topk=list(recbole_config["topk"]),
                pareto_reference_point=evaluation_reference["values"],
                preferences=validation_preferences,
            )
            validation_payload["validation_time_sec"] = float(time.monotonic() - valid_start)
            score = float(validation_payload["ranking_operating_point"]["metrics"]["NDCG@10"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_validation = validation_payload
                best_checkpoint = save_checkpoint(
                    artifact_dir / "checkpoints" / "best_validation.pth",
                    models,
                    optimizers,
                    epoch,
                    best_score,
                    {
                        "method": args.method,
                        "run_id": run_id,
                        "model_selection": validation_payload["ranking_operating_point_selection"],
                        "selection_is_validation_oracle": validation_payload["selection_is_validation_oracle"],
                    },
                )

        last_checkpoint = save_checkpoint(
            artifact_dir / "checkpoints" / "last.pth",
            models,
            optimizers,
            epoch,
            best_score if math.isfinite(best_score) else None,
            {"method": args.method, "run_id": run_id},
        )
        epoch_record = {
            "epoch": epoch,
            "losses": losses,
            "validation": validation_payload,
            "train_time_sec": float(train_time),
            "epoch_time_sec": float(time.monotonic() - epoch_start),
            "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        }
        epochs_payload.append(epoch_record)
        write_jsonl(training_log_path, epoch_record)
        save_json(
            partial_json,
            {
                "run_id": run_id,
                "status": "partial",
                "stage": stage,
                "method": args.method,
                "epoch": epoch,
                "epochs_completed": len(epochs_payload),
                "test_evaluation_count": 0,
            },
        )
        progress = {
            "run_id": run_id,
            "method": args.method,
            "stage": stage,
            "epoch": epoch,
            "train_loss_scalar": losses.get("moo_scalar"),
            "ranking_operating_point_ndcg10": (
                None if validation_payload is None else validation_payload["ranking_operating_point"]["metrics"]["NDCG@10"]
            ),
            "oracle_best_validation_ndcg10": (
                None if validation_payload is None else validation_payload["oracle_best_validation_point"]["metrics"]["NDCG@10"]
            ),
            "train_time_sec": train_time,
        }
        print(json.dumps(progress, ensure_ascii=False, allow_nan=False, default=json_default), flush=True)

    preference_sensitivity = None
    preference_sensitivity_failed = False
    sensitivity_config = dict(moo_config.get("preference_sensitivity", {}))
    if stage == "smoke" and args.method in tuple(sensitivity_config.get("enabled_methods", CONDITIONAL_METHODS)):
        preference_sensitivity = preference_sensitivity_diagnostic(
            model=models[0],
            train_data=train_data,
            preferences=preferences,
            p1_id=str(sensitivity_config.get("p1_id", RANKING_OPERATING_POINT_ID)),
            p2_id=str(sensitivity_config.get("p2_id", "like_heavy")),
            tolerance=float(sensitivity_config.get("tolerance", 1e-8)),
        )
        preference_sensitivity_failed = not bool(preference_sensitivity["output_metric_passed"])

    runtime_sec = time.monotonic() - start
    if stage == "smoke":
        status = "failed_preference_sensitivity" if preference_sensitivity_failed else "completed"
        validation_result = None
    else:
        if best_validation is None or best_epoch is None:
            raise RuntimeError("Sanity finished without validation.")
        status = "completed"
        validation_result = best_validation

    total_params = sum(count_parameters(model)["total"] for model in models)
    trainable_params = sum(count_parameters(model)["trainable"] for model in models)
    extra = {}
    if len(models) == 1 and hasattr(models[0], "extra_parameter_summary"):
        extra = models[0].extra_parameter_summary()

    method_warnings = []
    if not bool(method_config.get("exact_method_reproduction", True)):
        method_warnings.append(
            f"{method_config.get('implementation_name', method_config['representative'])} is marked as "
            f"{method_config.get('representative_fidelity', 'family-level adaptation')}, not exact method reproduction."
        )

    result: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "record_type": "sanity" if stage == "sanity" else "smoke",
        "stage": stage,
        "objective": "validation_full_ranking_NDCG@10" if stage == "sanity" else "train_only_smoke",
        "created_at_utc": run_started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": git_value(["config", "--get", "remote.origin.url"]),
            "expected_start_commit": moo_config["expected_start_commit"],
        },
        "source_files": source_hashes(),
        "source_checksum": sha256_json(source_hashes()),
        "config_checksum": sha256_json(moo_config),
        "preferences_checksum": sha256_json(preferences),
        "environment": environment_info(),
        "slurm": slurm_info(),
        "gpu": gpu_info(),
        "memory": {"process_ru_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
        "method": {
            "name": str(method_config["representative"]),
            "family": str(method_config["family"]),
            "method_key": args.method,
            "solution_type": str(method_config["solution_type"]),
            "implementation_name": method_config.get("implementation_name", method_config["representative"]),
            "representative_fidelity": method_config.get("representative_fidelity", "exact_or_close_reproduction"),
            "exact_method_reproduction": bool(method_config.get("exact_method_reproduction", True)),
            "config": dict(method_config),
            "state": method_state.state_dict() if hasattr(method_state, "state_dict") else method_state,
        },
        "dataset": {
            "name": "KuaiRand",
            "protocol": "B",
            "fingerprint_expected": EXPECTED_FINGERPRINT,
            "identity_hash_expected": EXPECTED_IDENTITY_HASH,
            "validation_only_summary": data.validation_only_summary,
            "loader": {
                "train_batches": len(train_data),
                "valid_batches": len(valid_data),
                "train_examples": len(data.train_dataset),
                "validation_examples": len(data.valid_dataset),
                "batch_size": int(recbole_config["train_batch_size"]),
            },
        },
        "test_safety": {
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
            "test_evaluated": False,
            "test_evaluation_count": 0,
        },
        "test_evaluation_count": 0,
        "normalization": normalization,
        "tuned_fixed_configuration": {
            "source_study": best_params["study_name"],
            "source_trial": int(best_params["trial_number"]),
            "lambda_aux": sampled["lambda_aux"],
            "learning_rate": sampled["learning_rate"],
            "weight_decay": sampled["weight_decay"],
            "dropout_prob": sampled["dropout_prob"],
            "head_lr_multiplier": sampled["head_lr_multiplier"],
            "head_learning_rate": sampled["head_learning_rate"],
            "normalized_task_weights": sampled["normalized_task_weights"],
            "effective_pos_weights": sampled["effective_pos_weights"],
            "effective_loss_multipliers": sampled["effective_loss_multipliers"],
            "effective_positive_multipliers": sampled["effective_positive_multipliers"],
            "locked_param_diff_checks": sampled.get("locked_param_diff_checks"),
        },
        "preferences": {
            "objective_order": list(TASK_ORDER),
            "used_for_training": (
                {
                    "distribution": "Dirichlet",
                    "sampler_diagnostics": preference_sampler.diagnostics(),
                    "fixed_grid_not_used_for_training": True,
                }
                if preference_sampler is not None
                else method_config.get("preference_set") or method_config.get("preference_id")
            ),
            "used_for_validation": method_config.get("eval_preference_set") if stage == "sanity" else None,
            "source": project_path(moo_config["source"]["preferences"]),
        },
        "evaluation": {
            "pareto_reference": evaluation_reference,
            "ranking_operating_point_id": RANKING_OPERATING_POINT_ID,
            "model_selection_metric": "ranking_operating_point.NDCG@10",
        },
        "preference_sensitivity": preference_sensitivity,
        "training": {
            "epochs": epochs_payload,
            "actual_epochs": len(epochs_payload),
            "max_batches": max_batches,
            "training_log_jsonl": str(training_log_path),
        },
        "validation": validation_result,
        "best_epoch": best_epoch,
        "best_valid_score": None if not math.isfinite(best_score) else best_score,
        "best_valid_metric": "NDCG@10",
        "gradient_diagnostics": diagnostics,
        "model_parameters": {
            "model_count": len(models),
            "total_parameters": int(total_params),
            "total_trainable": int(trainable_params),
            "per_model": [count_parameters(model) for model in models],
            "shared": [parameter_group_summary(shared_parameter_entries(model, "all_backbone")) for model in models],
            "extra": extra,
        },
        "checkpoints": {
            "best_validation": best_checkpoint,
            "last": last_checkpoint,
        },
        "artifact_dir": str(artifact_dir),
        "runtime": {
            "total_sec": float(runtime_sec),
            "train_total_sec": float(time.monotonic() - train_start),
            "mean_epoch_sec": float(sum(item["epoch_time_sec"] for item in epochs_payload) / len(epochs_payload)),
        },
        "cost": {
            "wall_time_sec": float(runtime_sec),
            "peak_vram_bytes": gpu_info().get("peak_allocated_bytes"),
            "model_count": len(models),
            "backward_passes_per_batch": {
                "stch": 1,
                "famo": 1,
                "epo": 2,
                "gradhv": 1,
                "phn": 1,
                "cosmos": 1,
                "palora": 1,
            }[args.method],
            "checkpoint_last_size_bytes": None if last_checkpoint is None else last_checkpoint["size_bytes"],
        },
        "warnings": method_warnings,
    }
    save_json(result_json, result)
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(build_notes(result) + "\n", encoding="utf-8")
    if partial_json.exists():
        partial_json.unlink()
    if preference_sensitivity_failed:
        raise RuntimeError(f"Preference sensitivity smoke check failed: {preference_sensitivity}")


if __name__ == "__main__":
    main()
