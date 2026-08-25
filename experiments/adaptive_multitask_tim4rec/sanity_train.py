#!/usr/bin/env python
"""5-epoch validation-only sanity runs for adaptive MultitaskTiM4Rec methods."""

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
from typing import Any

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
    AUX_TARGETS,
    TASK_ORDER,
    assign_flat_gradient,
    assign_gradient_tensors,
    conflict_summary,
    cosine_matrix,
    ensure_finite_gradients,
    gradient_norms,
    max_cuda_memory,
    parameter_group_summary,
    shared_parameter_entries,
    task_gradient_vectors,
    tensor_to_float,
)
from experiments.adaptive_multitask_tim4rec.methods.metabalance import MetaBalanceAuxiliaryBalancer  # noqa: E402
from experiments.adaptive_multitask_tim4rec.methods.pcgrad import PCGradProjector  # noqa: E402
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS  # noqa: E402
from experiments.multitask_tim4rec.train import (  # noqa: E402
    EXPECTED_FINGERPRINT,
    EXPECTED_IDENTITY_HASH,
    all_gradient_check,
    check_hit_recall_equal,
    count_parameters,
    evaluate_auxiliary,
    evaluate_full_sort_with_checks,
    load_json,
    metric_subset,
    sha256_file,
)
from experiments.multitask_tim4rec_optuna.optuna_search import (  # noqa: E402
    compact_validation,
    compute_tuned_losses,
    create_loaders,
    load_data_bundle,
    load_yaml,
    normalize_metrics,
    optimizer_for_trial,
    pos_weight_tensors,
    project_path,
)
from experiments.multitask_tim4rec_optuna.run_locked_tuned import sampled_from_locked_params  # noqa: E402


RUN_IDS = {
    "pcgrad": "pcgrad_sanity_001",
    "metabalance": "metabalance_sanity_001",
}
DEFAULT_REMOTE_ROOT = Path("/home/daryumin/iberdov/diplom/experiments/adaptive_multitask_tim4rec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(RUN_IDS), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--config", default=str(ROOT / "experiments/adaptive_multitask_tim4rec/config.yaml"))
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--diagnostic-epochs", default="1,3,5")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def git_value(args: list[str], default: str = "unknown") -> str:
    env_map = {
        ("rev-parse", "HEAD"): "ADAPTIVE_MTL_GIT_COMMIT",
        ("rev-parse", "--abbrev-ref", "HEAD"): "ADAPTIVE_MTL_GIT_BRANCH",
        ("config", "--get", "remote.origin.url"): "ADAPTIVE_MTL_GIT_REMOTE",
    }
    env_key = env_map.get(tuple(args))
    if env_key and os.environ.get(env_key):
        return str(os.environ[env_key])
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
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
    }


def source_hashes() -> dict[str, str]:
    relative_paths = [
        "experiments/adaptive_multitask_tim4rec/sanity_train.py",
        "experiments/adaptive_multitask_tim4rec/methods/common.py",
        "experiments/adaptive_multitask_tim4rec/methods/pcgrad.py",
        "experiments/adaptive_multitask_tim4rec/methods/metabalance.py",
        "experiments/adaptive_multitask_tim4rec/config.yaml",
        "experiments/multitask_tim4rec_optuna/optuna_search.py",
        "experiments/multitask_tim4rec_optuna/run_locked_tuned.py",
        "experiments/multitask_tim4rec_optuna/prepare_validation_only.py",
        "slurm/adaptive_multitask_sanity.sh",
    ]
    return {path: sha256_file(ROOT / path) for path in relative_paths}


def run_paths(run_id: str, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else DEFAULT_REMOTE_ROOT / run_id
    result_json = (
        Path(args.result_json)
        if args.result_json
        else ROOT / "experiments" / "adaptive_multitask_tim4rec" / "runs" / f"{run_id}.json"
    )
    notes = (
        Path(args.notes)
        if args.notes
        else ROOT / "experiments" / "adaptive_multitask_tim4rec" / "runs" / f"{run_id}_notes.md"
    )
    return artifact_dir, result_json, notes


def parse_diagnostic_epochs(value: str) -> set[int]:
    result = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not result:
        raise RuntimeError("At least one diagnostic epoch is required.")
    return result


def build_sanity_config(
    optuna_config: dict[str, Any],
    artifact_root: Path,
    sampled: dict[str, Any],
    epochs: int,
) -> Config:
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


def split_tuned_losses(losses: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    raw_aux = {target: losses[f"{target}_loss"] for target in AUX_TARGETS}
    task_contributions = {"rank": losses["rank"]}
    for target in AUX_TARGETS:
        task_contributions[target] = losses[f"{target}_scaled_contribution"]
    return raw_aux, task_contributions


def fixed_total_from_raw(
    rank_loss: torch.Tensor,
    aux_losses: dict[str, torch.Tensor],
    sampled: dict[str, Any],
) -> torch.Tensor:
    total = rank_loss
    for target in AUX_TARGETS:
        total = total + float(sampled["lambda_aux"]) * float(sampled["normalized_task_weights"][target]) * aux_losses[target]
    return total


def scalar_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    keys = [
        "total",
        "rank",
        "aux_sum",
        "weighted_aux_sum",
        "is_click_loss",
        "long_view_loss",
        "is_like_loss",
        "is_profile_enter_loss",
        "is_click_scaled_contribution",
        "long_view_scaled_contribution",
        "is_like_scaled_contribution",
        "is_profile_enter_scaled_contribution",
    ]
    return {key: tensor_to_float(losses[key]) for key in keys}


def rank_aux_summary(matrix: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    details = []
    observed = 0
    for target in AUX_TARGETS:
        value = matrix["rank"][target]
        if value is None:
            continue
        observed += 1
        if value < 0:
            details.append({"target": target, "cosine": value})
    return {
        "pairs": observed,
        "negative_pairs": len(details),
        "fraction_conflicting": len(details) / observed if observed else 0.0,
        "negative_pairs_detail": details,
    }


def auxiliary_summary(matrix: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    return conflict_summary(matrix, AUX_TARGETS)


def compact_conflicts(conflicts: dict[str, Any]) -> dict[str, Any]:
    return {
        "pairs": int(conflicts["pairs"]),
        "negative_pairs": int(conflicts["negative_pairs"]),
        "fraction_conflicting": float(conflicts["fraction_conflicting"]),
        "negative_pairs_detail": conflicts["negative_pairs_detail"],
    }


def diagnostic_record(
    *,
    method: str,
    epoch: int,
    batch_idx: int,
    losses: dict[str, torch.Tensor],
    before_matrix: dict[str, dict[str, float | None]],
    after_matrix: dict[str, dict[str, float | None]],
    before_norms: dict[str, float],
    after_norms: dict[str, float],
    before_conflicts: dict[str, Any],
    after_conflicts: dict[str, Any],
    method_effect: dict[str, Any],
) -> dict[str, Any]:
    task_table = []
    for task in TASK_ORDER:
        raw_key = "rank" if task == "rank" else f"{task}_loss"
        weighted_key = "rank" if task == "rank" else f"{task}_scaled_contribution"
        task_table.append(
            {
                "task": task,
                "raw_loss": tensor_to_float(losses[raw_key]),
                "weighted_or_effective_contribution": tensor_to_float(losses[weighted_key]),
                "shared_gradient_norm_before": before_norms[task],
                "shared_gradient_norm_after": after_norms[task],
                "cosine_with_ranking_before": 1.0 if task == "rank" else before_matrix["rank"][task],
                "cosine_with_ranking_after": 1.0 if task == "rank" else after_matrix["rank"][task],
            }
        )
    like = next(item for item in task_table if item["task"] == "is_like")
    return {
        "method": method,
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "task_table": task_table,
        "rank_aux_cosines_before": {target: before_matrix["rank"][target] for target in AUX_TARGETS},
        "rank_aux_cosines_after": {target: after_matrix["rank"][target] for target in AUX_TARGETS},
        "rank_aux_conflicts_before": rank_aux_summary(before_matrix),
        "rank_aux_conflicts_after": rank_aux_summary(after_matrix),
        "auxiliary_conflicts_before": auxiliary_summary(before_matrix),
        "auxiliary_conflicts_after": auxiliary_summary(after_matrix),
        "all_task_conflicts_before": compact_conflicts(before_conflicts),
        "all_task_conflicts_after": compact_conflicts(after_conflicts),
        "is_like": like,
        "method_effect": method_effect,
    }


def empty_conflict_stats() -> dict[str, Any]:
    return {
        "batches": 0,
        "all_pairs_before": 0,
        "all_negative_before": 0,
        "all_pairs_after": 0,
        "all_negative_after": 0,
        "rank_aux_pairs_before": 0,
        "rank_aux_negative_before": 0,
        "rank_aux_pairs_after": 0,
        "rank_aux_negative_after": 0,
        "aux_pairs_before": 0,
        "aux_negative_before": 0,
        "aux_pairs_after": 0,
        "aux_negative_after": 0,
        "rank_aux_negative_by_target_before": {target: 0 for target in AUX_TARGETS},
        "rank_aux_negative_by_target_after": {target: 0 for target in AUX_TARGETS},
        "projection_event_count": 0,
        "scale_summary": {},
    }


def update_scale_stats(stats: dict[str, Any], scale_summary: dict[str, Any] | None) -> None:
    if not scale_summary:
        return
    target_stats = stats["scale_summary"]
    for target, values in scale_summary.items():
        if values["mean"] is None:
            continue
        current = target_stats.setdefault(
            target,
            {"batches": 0, "min": float("inf"), "max": 0.0, "mean_sum": 0.0},
        )
        current["batches"] += 1
        current["min"] = min(float(current["min"]), float(values["min"]))
        current["max"] = max(float(current["max"]), float(values["max"]))
        current["mean_sum"] += float(values["mean"])


def update_conflict_stats(
    stats: dict[str, Any],
    before_matrix: dict[str, dict[str, float | None]],
    after_matrix: dict[str, dict[str, float | None]],
    before_conflicts: dict[str, Any],
    after_conflicts: dict[str, Any],
    method_effect: dict[str, Any],
) -> None:
    before_rank = rank_aux_summary(before_matrix)
    after_rank = rank_aux_summary(after_matrix)
    before_aux = auxiliary_summary(before_matrix)
    after_aux = auxiliary_summary(after_matrix)
    stats["batches"] += 1
    stats["all_pairs_before"] += int(before_conflicts["pairs"])
    stats["all_negative_before"] += int(before_conflicts["negative_pairs"])
    stats["all_pairs_after"] += int(after_conflicts["pairs"])
    stats["all_negative_after"] += int(after_conflicts["negative_pairs"])
    stats["rank_aux_pairs_before"] += int(before_rank["pairs"])
    stats["rank_aux_negative_before"] += int(before_rank["negative_pairs"])
    stats["rank_aux_pairs_after"] += int(after_rank["pairs"])
    stats["rank_aux_negative_after"] += int(after_rank["negative_pairs"])
    stats["aux_pairs_before"] += int(before_aux["pairs"])
    stats["aux_negative_before"] += int(before_aux["negative_pairs"])
    stats["aux_pairs_after"] += int(after_aux["pairs"])
    stats["aux_negative_after"] += int(after_aux["negative_pairs"])
    for item in before_rank["negative_pairs_detail"]:
        stats["rank_aux_negative_by_target_before"][item["target"]] += 1
    for item in after_rank["negative_pairs_detail"]:
        stats["rank_aux_negative_by_target_after"][item["target"]] += 1
    stats["projection_event_count"] += int(method_effect.get("projection_event_count") or 0)
    update_scale_stats(stats, method_effect.get("scale_summary"))


def finalize_conflict_stats(stats: dict[str, Any]) -> dict[str, Any]:
    def fraction(negative: int, pairs: int) -> float:
        return negative / pairs if pairs else 0.0

    scale = {}
    for target, values in stats["scale_summary"].items():
        scale[target] = {
            "batches": int(values["batches"]),
            "min": float(values["min"]),
            "max": float(values["max"]),
            "mean": float(values["mean_sum"] / max(values["batches"], 1)),
        }
    return {
        **{key: value for key, value in stats.items() if key != "scale_summary"},
        "all_fraction_before": fraction(stats["all_negative_before"], stats["all_pairs_before"]),
        "all_fraction_after": fraction(stats["all_negative_after"], stats["all_pairs_after"]),
        "rank_aux_fraction_before": fraction(stats["rank_aux_negative_before"], stats["rank_aux_pairs_before"]),
        "rank_aux_fraction_after": fraction(stats["rank_aux_negative_after"], stats["rank_aux_pairs_after"]),
        "aux_fraction_before": fraction(stats["aux_negative_before"], stats["aux_pairs_before"]),
        "aux_fraction_after": fraction(stats["aux_negative_after"], stats["aux_pairs_after"]),
        "scale_summary": scale,
    }


def merge_finalized_conflict_stats(total: dict[str, Any], epoch_stats: dict[str, Any]) -> None:
    for key in (
        "batches",
        "all_pairs_before",
        "all_negative_before",
        "all_pairs_after",
        "all_negative_after",
        "rank_aux_pairs_before",
        "rank_aux_negative_before",
        "rank_aux_pairs_after",
        "rank_aux_negative_after",
        "aux_pairs_before",
        "aux_negative_before",
        "aux_pairs_after",
        "aux_negative_after",
        "projection_event_count",
    ):
        total[key] += int(epoch_stats[key])
    for target in AUX_TARGETS:
        total["rank_aux_negative_by_target_before"][target] += int(
            epoch_stats["rank_aux_negative_by_target_before"][target]
        )
        total["rank_aux_negative_by_target_after"][target] += int(
            epoch_stats["rank_aux_negative_by_target_after"][target]
        )
    for target, values in epoch_stats.get("scale_summary", {}).items():
        if values["mean"] is None:
            continue
        current = total["scale_summary"].setdefault(
            target,
            {"batches": 0, "min": float("inf"), "max": 0.0, "mean_sum": 0.0},
        )
        batches = int(values["batches"])
        current["batches"] += batches
        current["min"] = min(float(current["min"]), float(values["min"]))
        current["max"] = max(float(current["max"]), float(values["max"]))
        current["mean_sum"] += float(values["mean"]) * batches


def pcgrad_step(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    interaction: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    shared_entries: list[Any],
    projector: PCGradProjector,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    losses = compute_tuned_losses(model, interaction, sampled, pos_weights)
    raw_aux, task_contributions = split_tuned_losses(losses)
    vectors = task_gradient_vectors(task_contributions, shared_entries, TASK_ORDER)
    projection = projector.project(vectors, TASK_ORDER)

    optimizer.zero_grad(set_to_none=True)
    refreshed = compute_tuned_losses(model, interaction, sampled, pos_weights)
    refreshed_raw_aux, _ = split_tuned_losses(refreshed)
    total = fixed_total_from_raw(refreshed["rank"], refreshed_raw_aux, sampled)
    total.backward()
    assign_flat_gradient(shared_entries, projection["combined_gradient"])
    finite = all_gradient_check(model)
    shared_finite = ensure_finite_gradients(shared_entries)
    if not bool(finite["all_finite"]) or not bool(shared_finite["all_finite"]):
        raise RuntimeError(f"Non-finite PCGrad gradients: model={finite}, shared={shared_finite}")
    optimizer.step()

    method_effect = {
        "mode": projector.mode,
        "projection_events": projection["projection_events"],
        "projection_event_count": projection["projection_event_count"],
        "combined_gradient_norm_before": projection["combined_gradient_norm_before"],
        "combined_gradient_norm_after": projection["combined_gradient_norm_after"],
        "shared_gradient_finite": shared_finite,
    }
    compact = {
        "cosine_matrix_before": projection["cosine_matrix_before"],
        "cosine_matrix_after": projection["cosine_matrix_after"],
        "gradient_norms_before": projection["gradient_norms_before"],
        "gradient_norms_after": projection["gradient_norms_after"],
        "conflicts_before": projection["conflicts_before"],
        "conflicts_after": projection["conflicts_after"],
        "method_effect": method_effect,
        "diagnostic_losses": losses,
    }
    return scalar_losses(refreshed), compact, {"raw_aux": raw_aux}


def metabalance_step(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    interaction: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    shared_entries: list[Any],
    balancer: MetaBalanceAuxiliaryBalancer,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    losses = compute_tuned_losses(model, interaction, sampled, pos_weights)
    raw_aux, task_contributions = split_tuned_losses(losses)
    balanced = balancer.balanced_shared_gradients(task_contributions, shared_entries, TASK_ORDER)

    optimizer.zero_grad(set_to_none=True)
    refreshed = compute_tuned_losses(model, interaction, sampled, pos_weights)
    refreshed_raw_aux, _ = split_tuned_losses(refreshed)
    total = fixed_total_from_raw(refreshed["rank"], refreshed_raw_aux, sampled)
    total.backward()
    assign_gradient_tensors(shared_entries, balanced["combined_gradients"])
    finite = all_gradient_check(model)
    shared_finite = ensure_finite_gradients(shared_entries)
    if not bool(finite["all_finite"]) or not bool(shared_finite["all_finite"]):
        raise RuntimeError(f"Non-finite MetaBalance gradients: model={finite}, shared={shared_finite}")
    optimizer.step()

    method_effect = {
        "variant": "MetaBalance-Fix",
        "relax_factor": balancer.relax_factor,
        "beta": balancer.beta,
        "scale_summary": balanced["scale_summary"],
        "shared_gradient_finite": shared_finite,
    }
    compact = {
        "cosine_matrix_before": balanced["cosine_matrix_before"],
        "cosine_matrix_after": balanced["cosine_matrix_after"],
        "gradient_norms_before": balanced["gradient_norms_before"],
        "gradient_norms_after": balanced["gradient_norms_after"],
        "conflicts_before": balanced["conflicts_before"],
        "conflicts_after": balanced["conflicts_after"],
        "method_effect": method_effect,
        "diagnostic_losses": losses,
    }
    return scalar_losses(refreshed), compact, {"raw_aux": raw_aux}


def train_one_epoch_adaptive(
    *,
    method: str,
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    device: torch.device,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    shared_entries: list[Any],
    method_state: Any,
    epoch: int,
    diagnostic_epochs: set[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    model.train()
    sums: dict[str, float] = {}
    examples = 0
    batches = 0
    stats = empty_conflict_stats()
    first_batch_diagnostic = None

    for batch_idx, interaction in enumerate(train_data):
        interaction = interaction.to(device)
        batch_size = len(interaction)
        if method == "pcgrad":
            loss_scalars, step_diag, _extra = pcgrad_step(
                model,
                optimizer,
                interaction,
                sampled,
                pos_weights,
                shared_entries,
                method_state,
            )
        elif method == "metabalance":
            loss_scalars, step_diag, _extra = metabalance_step(
                model,
                optimizer,
                interaction,
                sampled,
                pos_weights,
                shared_entries,
                method_state,
            )
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported method: {method}")

        for key, value in loss_scalars.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
        examples += batch_size
        batches += 1
        update_conflict_stats(
            stats,
            step_diag["cosine_matrix_before"],
            step_diag["cosine_matrix_after"],
            step_diag["conflicts_before"],
            step_diag["conflicts_after"],
            step_diag["method_effect"],
        )
        if epoch in diagnostic_epochs and batch_idx == 0:
            first_batch_diagnostic = diagnostic_record(
                method=method,
                epoch=epoch,
                batch_idx=batch_idx,
                losses=step_diag["diagnostic_losses"],
                before_matrix=step_diag["cosine_matrix_before"],
                after_matrix=step_diag["cosine_matrix_after"],
                before_norms=step_diag["gradient_norms_before"],
                after_norms=step_diag["gradient_norms_after"],
                before_conflicts=step_diag["conflicts_before"],
                after_conflicts=step_diag["conflicts_after"],
                method_effect=step_diag["method_effect"],
            )
    if examples == 0:
        raise RuntimeError("No training examples.")
    losses = {key: value / examples for key, value in sums.items()}
    rank = losses["rank"]
    losses["auxiliary_scaled_contribution"] = float(sampled["lambda_aux"]) * losses["weighted_aux_sum"]
    losses["auxiliary_rank_ratio"] = losses["auxiliary_scaled_contribution"] / rank if rank else None
    losses["per_task_rank_ratio"] = {
        target: losses[f"{target}_scaled_contribution"] / rank if rank else None
        for target in AUX_TARGETS
    }
    losses["batches"] = batches
    losses["examples"] = examples
    return losses, finalize_conflict_stats(stats), first_batch_diagnostic


def save_checkpoint(
    path: Path,
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_valid_score: float,
    valid_metrics: dict[str, float],
    sampled: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "method": method,
            "best_valid_score": float(best_valid_score),
            "valid_metrics": valid_metrics,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "sampled": sampled,
        },
        path,
        pickle_protocol=4,
    )
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_reference_metrics() -> dict[str, Any]:
    tim4rec = load_json(ROOT / "experiments/tim4rec_baseline/runs/tim4rec_001.json")
    tim4rec_sanity = load_json(ROOT / "experiments/tim4rec_baseline/runs/tim4rec_sanity_001.json")
    fixed = load_json(ROOT / "experiments/multitask_tim4rec/runs/multitask_tim4rec_001.json")
    fixed_sanity = load_json(ROOT / "experiments/multitask_tim4rec/runs/multitask_tim4rec_sanity_001.json")
    tuned = load_json(ROOT / "experiments/multitask_tim4rec_optuna/runs/multitask_tim4rec_tuned_001.json")
    search = load_json(ROOT / "experiments/multitask_tim4rec_optuna/runs/multitask_optuna_search_001.json")
    tuned_reproduction = tuned["validation_reproduction"]["reproduced_validation"]
    return {
        "tim4rec_001_full_reference": {
            "run_id": "tim4rec_001",
            "run_type": "full_reference",
            "validation_metrics": normalize_metrics(tim4rec["best_validation_metrics"]),
            "best_epoch": tim4rec.get("best_epoch"),
        },
        "tim4rec_sanity_001": {
            "run_id": "tim4rec_sanity_001",
            "run_type": "5_epoch_sanity_reference",
            "validation_metrics": normalize_metrics(tim4rec_sanity["epochs"][-1]["validation"]),
            "best_epoch": tim4rec_sanity.get("best_epoch"),
            "limitation": "Использует TiM4Rec-only config, не tuned fixed params.",
        },
        "multitask_tim4rec_001_full_reference": {
            "run_id": "multitask_tim4rec_001",
            "run_type": "full_reference",
            "validation_metrics": normalize_metrics(fixed["best_validation_metrics"]),
            "best_epoch": fixed.get("best_epoch"),
        },
        "multitask_tim4rec_sanity_001": {
            "run_id": "multitask_tim4rec_sanity_001",
            "run_type": "5_epoch_sanity_reference",
            "validation_metrics": normalize_metrics(fixed_sanity["epochs"][-1]["validation"]),
            "best_epoch": fixed_sanity.get("best_epoch"),
            "limitation": "Использует старый fixed lambda_aux=0.2, не tuned trial 110.",
        },
        "multitask_tim4rec_tuned_001_validation_reproduction": {
            "run_id": "multitask_tim4rec_tuned_001",
            "run_type": "full_tuned_reference",
            "validation_metrics": normalize_metrics(tuned_reproduction),
            "best_epoch": tuned["validation_reproduction"].get("best_epoch"),
            "actual_epochs": tuned["validation_reproduction"].get("actual_epochs"),
            "limitation": "Это full tuned reference, а не отдельный 5-epoch control.",
        },
        "multitask_optuna_search_001_best_trial": {
            "run_id": "multitask_optuna_search_001",
            "run_type": "search_reference",
            "validation_metrics": normalize_metrics(search["best_trial"]["validation_metrics"]),
            "best_epoch": search["best_trial"].get("best_epoch"),
            "limitation": "Optuna objective value; использован только как источник exact tuned params.",
        },
    }


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def metric_table(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['HR@10']:.4f} | {metrics['HR@20']:.4f} | {metrics['HR@50']:.4f} | "
        f"{metrics['NDCG@10']:.4f} | {metrics['NDCG@20']:.4f} | {metrics['NDCG@50']:.4f}"
    )


def build_notes(result: dict[str, Any]) -> str:
    best = result["best_validation"]
    method = result["method"]["name"]
    epochs = result["epochs"]
    lines = [
        f"# {result['run_id']}",
        "",
        "## Test safety",
        "",
        "- `test_evaluation_count = 0`.",
        "- Test dataset не загружался, test dataloader не создавался.",
        "- Training идёт на validation-only RecBole benchmark: только `train` и `valid`.",
        "",
        "## Метод",
        "",
        f"- Method: `{method}`.",
        f"- Exact variant: `{result['method']['variant']}`.",
        f"- Shared selector: `{result['method']['shared_selector']}`.",
        f"- Epochs: `{result['actual_epochs']}`.",
        "",
        "## Tuned fixed стартовая конфигурация",
        "",
        f"- Study: `{result['tuned_fixed_configuration']['source_study']}`.",
        f"- Trial: `{result['tuned_fixed_configuration']['source_trial']}`.",
        f"- `lambda_aux`: `{result['tuned_fixed_configuration']['lambda_aux']:.12g}`.",
        f"- `learning_rate`: `{result['tuned_fixed_configuration']['learning_rate']:.12g}`.",
        f"- `weight_decay`: `{result['tuned_fixed_configuration']['weight_decay']:.12g}`.",
        f"- `dropout_prob`: `{result['tuned_fixed_configuration']['dropout_prob']:.12g}`.",
        f"- `head_lr_multiplier`: `{result['tuned_fixed_configuration']['head_lr_multiplier']:.12g}`.",
        "",
        "## Validation best",
        "",
        "| HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {metric_table(best['metrics'])} |",
        "",
        "## Epoch trajectory",
        "",
        "| epoch | L_total | L_rank | L_click | L_long | L_like | L_profile | HR@10 | NDCG@10 | train sec | valid sec |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in epochs:
        losses = epoch["losses"]
        valid = epoch["validation_metrics"]
        lines.append(
            f"| {epoch['epoch']} | {losses['total']:.4f} | {losses['rank']:.4f} | "
            f"{losses['is_click_loss']:.4f} | {losses['long_view_loss']:.4f} | "
            f"{losses['is_like_loss']:.4f} | {losses['is_profile_enter_loss']:.4f} | "
            f"{valid['HR@10']:.4f} | {valid['NDCG@10']:.4f} | "
            f"{epoch['train_time_sec']:.1f} | {epoch['validation_time_sec']:.1f} |"
        )
    lines += [
        "",
        "## Auxiliary validation at best epoch",
        "",
        "| target | ROC-AUC | PR-AUC | BCE | positive rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, metrics in best["auxiliary_validation"].items():
        lines.append(
            f"| `{target}` | {format_float(metrics['roc_auc'])} | {format_float(metrics['pr_auc'])} | "
            f"{format_float(metrics['bce_loss'])} | {format_float(metrics['positive_rate'])} |"
        )
    lines += [
        "",
        "## Rank-aux conflicts",
        "",
        f"- До adaptive update: `{result['gradient_conflict_summary']['rank_aux_fraction_before']:.6f}`.",
        f"- После adaptive update: `{result['gradient_conflict_summary']['rank_aux_fraction_after']:.6f}`.",
        f"- Negative rank-vs-aux counts before: `{result['gradient_conflict_summary']['rank_aux_negative_by_target_before']}`.",
        "",
        "## Diagnostic points",
        "",
        "| epoch | rank-click | rank-long | rank-like | rank-profile | rank-aux conflict fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for diag in result["gradient_diagnostics"]:
        cos = diag["rank_aux_cosines_before"]
        lines.append(
            f"| {diag['epoch']} | {cos['is_click']:.6f} | {cos['long_view']:.6f} | "
            f"{cos['is_like']:.6f} | {cos['is_profile_enter']:.6f} | "
            f"{diag['rank_aux_conflicts_before']['fraction_conflicting']:.4f} |"
        )
    lines += [
        "",
        "## Cost",
        "",
        f"- Mean train epoch: `{result['cost']['mean_train_epoch_time_sec']:.3f}` sec.",
        f"- Mean validation: `{result['cost']['mean_validation_time_sec']:.3f}` sec.",
        f"- Peak VRAM: `{result['gpu']['peak_allocated_bytes']}` bytes.",
        f"- Process MaxRSS: `{result['memory']['process_ru_maxrss_kb']}` KB.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or RUN_IDS[args.method]
    expected_run_id = RUN_IDS[args.method]
    if run_id != expected_run_id:
        raise RuntimeError(f"Expected run_id={expected_run_id} for method={args.method}, got {run_id}")
    if int(args.epochs) != 5:
        raise RuntimeError(f"This prompt requires exactly 5 epochs, got {args.epochs}")

    artifact_dir, result_json, notes_path = run_paths(run_id, args)
    partial_json = result_json.with_suffix(".partial.json")
    if not args.allow_overwrite:
        if result_json.exists() or notes_path.exists() or partial_json.exists():
            raise RuntimeError(f"Refusing to overwrite existing run artifact: {result_json}")
        if artifact_dir.exists() and any(artifact_dir.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty artifact dir: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = artifact_dir / "training_log.jsonl"

    adaptive_config = load_yaml(Path(args.config))
    optuna_config = load_yaml(project_path(adaptive_config["base"]["optuna_config"]))
    best_params = load_yaml(project_path(adaptive_config["base"]["best_params"]))
    data = load_data_bundle(optuna_config, artifact_dir / "data_probe")
    sampled = sampled_from_locked_params(best_params, data.target_stats)
    config = build_sanity_config(optuna_config, artifact_dir, sampled, int(args.epochs))
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    if tuple(config["multitask_targets"]) != TARGETS:
        raise RuntimeError(f"Task set changed: {config['multitask_targets']}")
    if not bool(config["is_time"]):
        raise RuntimeError("TiM4Rec is_time must stay True.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for adaptive sanity training.")
    if data.validation_only_summary["forbidden_test_paths_loaded"] != []:
        raise RuntimeError(f"Validation-only prep touched test paths: {data.validation_only_summary}")
    if int(data.validation_only_summary["rows"]["test"]) != 0:
        raise RuntimeError(f"Validation-only benchmark unexpectedly has test rows: {data.validation_only_summary}")
    if data.validation_only_summary.get("identity_hash") != EXPECTED_IDENTITY_HASH:
        raise RuntimeError(f"Identity hash mismatch: {data.validation_only_summary.get('identity_hash')}")

    train_data, valid_data = create_loaders(config, data.train_dataset, data.valid_dataset)
    device = config["device"]
    pos_weights = pos_weight_tensors(sampled["effective_pos_weights"], device)
    diagnostic_epochs = parse_diagnostic_epochs(args.diagnostic_epochs)
    torch.cuda.reset_peak_memory_stats()

    model = MultitaskTiM4Rec(config, train_data.dataset).to(device)
    optimizer = optimizer_for_trial(model, sampled)
    trainer = Trainer(config, model)
    trainer.optimizer = optimizer
    shared_entries = shared_parameter_entries(model, "all_backbone")
    if args.method == "pcgrad":
        method_state = PCGradProjector(
            mode=str(adaptive_config["pcgrad"]["mode"]),
            seed=int(adaptive_config["pcgrad"]["seed"]),
        )
        method_info = {
            "name": "PCGrad",
            "variant": "ranking_anchored",
            "shared_selector": str(adaptive_config["pcgrad"]["shared_selector"]),
            "mode": method_state.mode,
            "seed": method_state.seed,
            "algorithm": "g_rank is unchanged; each auxiliary gradient is projected only if dot(g_aux, g_rank) < 0; auxiliary-auxiliary conflicts are not processed.",
        }
    else:
        method_state = MetaBalanceAuxiliaryBalancer(
            relax_factor=float(adaptive_config["metabalance"]["relax_factor"]),
            beta=float(adaptive_config["metabalance"]["beta"]),
        )
        method_info = {
            "name": "MetaBalance",
            "variant": "MetaBalance-Fix",
            "shared_selector": str(adaptive_config["metabalance"]["shared_selector"]),
            "relax_factor": method_state.relax_factor,
            "beta": method_state.beta,
            "algorithm": "L_rank is the target task; auxiliary gradient magnitudes are moved toward target gradient magnitude with moving averages; directions are not projected.",
        }

    start = time.monotonic()
    run_started = datetime.now(timezone.utc)
    best_epoch = None
    best_score = -float("inf")
    best_snapshot: dict[str, Any] | None = None
    best_checkpoint = None
    last_checkpoint = None
    epochs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    aggregate_stats = empty_conflict_stats()
    topk = list(config["topk"])

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.monotonic()
        train_start = time.monotonic()
        losses, epoch_conflicts, first_diag = train_one_epoch_adaptive(
            method=args.method,
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            device=device,
            sampled=sampled,
            pos_weights=pos_weights,
            shared_entries=shared_entries,
            method_state=method_state,
            epoch=epoch,
            diagnostic_epochs=diagnostic_epochs,
        )
        train_time = time.monotonic() - train_start
        if first_diag is not None:
            diagnostics.append(first_diag)

        valid_start = time.monotonic()
        valid_result, full_checks = evaluate_full_sort_with_checks(trainer, valid_data, train_data)
        auxiliary_validation = evaluate_auxiliary(model, valid_data, device)
        validation_time = time.monotonic() - valid_start
        check_hit_recall_equal(valid_result, topk)
        if not full_checks["raw_scores_all_finite"] or not full_checks["positive_scores_all_finite"]:
            raise RuntimeError(f"Non-finite validation scores: {full_checks}")
        metrics = normalize_metrics(metric_subset(valid_result))
        valid_score = float(metrics["NDCG@10"])
        update_flag = valid_score > best_score
        if update_flag:
            best_epoch = epoch
            best_score = valid_score
            best_snapshot = {
                "epoch": epoch,
                "metrics": metrics,
                "compact_metrics": compact_validation(metrics),
                "auxiliary_validation": auxiliary_validation,
                "losses": losses,
                "full_ranking_checks": full_checks,
                "validation_time_sec": validation_time,
            }
            best_checkpoint = save_checkpoint(
                checkpoint_dir / "best_validation.pth",
                model,
                optimizer,
                epoch,
                best_score,
                metrics,
                sampled,
                args.method,
            )
        last_checkpoint = save_checkpoint(
            checkpoint_dir / "last.pth",
            model,
            optimizer,
            epoch,
            best_score,
            metrics,
            sampled,
            args.method,
        )
        epoch_result = {
            "epoch": epoch,
            "losses": losses,
            "validation_metrics": metrics,
            "auxiliary_validation": auxiliary_validation,
            "valid_score": valid_score,
            "valid_metric": "NDCG@10",
            "best_so_far": {"epoch": best_epoch, "NDCG@10": best_score},
            "hit_recall_equal_check": check_hit_recall_equal(valid_result, topk),
            "full_ranking_checks": full_checks,
            "gradient_conflicts": epoch_conflicts,
            "train_time_sec": train_time,
            "validation_time_sec": validation_time,
            "epoch_time_sec": time.monotonic() - epoch_start,
            "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()),
        }
        epochs.append(epoch_result)
        merge_finalized_conflict_stats(aggregate_stats, epoch_conflicts)

        with training_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_result, ensure_ascii=False, default=json_default) + "\n")
        save_json(
            partial_json,
            {
                "run_id": run_id,
                "status": "partial",
                "method": args.method,
                "epochs_completed": len(epochs),
                "latest_epoch": epoch_result,
                "best_epoch_so_far": best_epoch,
                "best_valid_score_so_far": best_score,
                "test_evaluation_count": 0,
            },
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "validation_ndcg10": metrics["NDCG@10"],
                    "validation_hr10": metrics["HR@10"],
                    "train_time_sec": train_time,
                    "validation_time_sec": validation_time,
                    "rank_aux_conflict_fraction": epoch_conflicts["rank_aux_fraction_before"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if best_snapshot is None or best_epoch is None:
        raise RuntimeError("No validation snapshot recorded.")

    runtime_sec = time.monotonic() - start
    aggregate = finalize_conflict_stats(aggregate_stats)
    references = load_reference_metrics()
    result: dict[str, Any] = {
        "run_id": run_id,
        "status": "completed",
        "sanity": True,
        "objective": "validation_full_ranking_NDCG@10",
        "created_at_utc": run_started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": git_value(["config", "--get", "remote.origin.url"]),
            "expected_start_commit": "914d5c583e2dafefe9bb30fca92a17a8bc6b6852",
        },
        "source_files": source_hashes(),
        "environment": environment_info(),
        "slurm": slurm_info(),
        "gpu": gpu_info()
        | {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "memory": {
            "process_ru_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "method": method_info,
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
                "batch_size": int(config["train_batch_size"]),
            },
        },
        "test_safety": {
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
            "test_evaluated": False,
            "test_evaluation_count": 0,
        },
        "test_evaluation_count": 0,
        "model_parameters": {
            "multitask": count_parameters(model),
            "shared": parameter_group_summary(shared_entries),
        },
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
            "locked_param_diff_checks": sampled["locked_param_diff_checks"],
        },
        "config": {
            "epochs_requested": int(args.epochs),
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "is_time": bool(config["is_time"]),
            "metrics": list(config["metrics"]),
            "topk": topk,
            "eval_args": config["eval_args"],
        },
        "epochs": epochs,
        "actual_epochs": len(epochs),
        "best_epoch": best_epoch,
        "best_valid_score": best_score,
        "best_valid_metric": "NDCG@10",
        "best_validation": best_snapshot,
        "best_validation_metrics": best_snapshot["metrics"],
        "best_validation_compact": best_snapshot["compact_metrics"],
        "best_auxiliary_metrics": best_snapshot["auxiliary_validation"],
        "best_epoch_losses": best_snapshot["losses"],
        "gradient_diagnostics": diagnostics,
        "gradient_conflict_summary": aggregate,
        "rank_aux_conflict_appeared": bool(aggregate["rank_aux_negative_before"] > 0),
        "is_like_summary": {
            "tuned_fixed_weight": sampled["normalized_task_weights"]["is_like"],
            "effective_pos_weight": sampled["effective_pos_weights"]["is_like"],
            "diagnostic_points": [{"epoch": diag["epoch"], **diag["is_like"]} for diag in diagnostics],
        },
        "baseline_comparison": references,
        "control_run_policy": {
            "new_tuned_fixed_control_run_started": False,
            "reason": "Prompt explicitly forbids starting a third tuned fixed control if no exact 5-epoch tuned run already exists.",
        },
        "checkpoints": {
            "best_validation": best_checkpoint,
            "last": last_checkpoint,
        },
        "artifact_dir": str(artifact_dir),
        "runtime": {
            "total_sec": runtime_sec,
            "mean_epoch_sec": sum(item["epoch_time_sec"] for item in epochs) / len(epochs),
        },
        "cost": {
            "mean_train_epoch_time_sec": sum(item["train_time_sec"] for item in epochs) / len(epochs),
            "mean_validation_time_sec": sum(item["validation_time_sec"] for item in epochs) / len(epochs),
            "mean_epoch_time_sec": sum(item["epoch_time_sec"] for item in epochs) / len(epochs),
            "fixed_tuned_smoke_mean_step_sec": 0.10299260293443997,
            "method_smoke_mean_step_sec": 0.1972333835437894 if args.method == "pcgrad" else 0.24934941778580347,
            "overhead_vs_fixed_tuned_smoke": (
                (0.1972333835437894 if args.method == "pcgrad" else 0.24934941778580347)
                / 0.10299260293443997
            ),
        },
        "warnings": [],
    }
    save_json(result_json, result)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(build_notes(result) + "\n", encoding="utf-8")
    if partial_json.exists():
        partial_json.unlink()


if __name__ == "__main__":
    main()
