#!/usr/bin/env python
"""Full validation-only ranking-anchored PCGrad run for MultitaskTiM4Rec."""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
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
    conflict_summary,
    cosine_matrix,
    ensure_finite_gradients,
    gradient_norms,
    shared_parameter_entries,
    task_gradient_vectors,
)
from experiments.adaptive_multitask_tim4rec.methods.pcgrad import PCGradProjector  # noqa: E402
from experiments.adaptive_multitask_tim4rec.sanity_train import (  # noqa: E402
    DEFAULT_REMOTE_ROOT,
    auxiliary_summary,
    compact_conflicts,
    compact_validation,
    diagnostic_record,
    empty_conflict_stats,
    environment_info,
    fixed_total_from_raw,
    format_float,
    finalize_conflict_stats,
    gpu_info,
    json_default,
    load_reference_metrics,
    metric_table,
    rank_aux_summary,
    save_checkpoint,
    save_json,
    scalar_losses,
    split_tuned_losses,
    update_conflict_stats,
)
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


RUN_ID = "pcgrad_001"
EXPECTED_STUDY = "multitask_tim4rec_optuna_v1"
EXPECTED_TRIAL = 110
METRIC_TOPK = (5, 10, 20, 50)
DEFAULT_DIAGNOSTIC_DETAIL_EPOCHS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--config", default=str(ROOT / "experiments/adaptive_multitask_tim4rec/config.yaml"))
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--diagnostic-detail-epochs", default="1,3,5,10")
    parser.add_argument("--diagnostic-batches", type=int, default=10)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--soft-time-limit-sec",
        type=float,
        default=0.0,
        help="Stop after an epoch before the Slurm hard limit; writes resume state and leaves final JSON absent.",
    )
    return parser.parse_args()


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


def slurm_info() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "constraint": os.environ.get("ADAPTIVE_MTL_SLURM_CONSTRAINT") or os.environ.get("SLURM_JOB_CONSTRAINT"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "hostname": socket.gethostname(),
    }


def source_hashes() -> dict[str, str]:
    relative_paths = [
        "experiments/adaptive_multitask_tim4rec/pcgrad_train.py",
        "experiments/adaptive_multitask_tim4rec/sanity_train.py",
        "experiments/adaptive_multitask_tim4rec/methods/common.py",
        "experiments/adaptive_multitask_tim4rec/methods/pcgrad.py",
        "experiments/adaptive_multitask_tim4rec/config.yaml",
        "experiments/multitask_tim4rec_optuna/optuna_search.py",
        "experiments/multitask_tim4rec_optuna/run_locked_tuned.py",
        "experiments/multitask_tim4rec_optuna/prepare_validation_only.py",
        "slurm/adaptive_multitask_pcgrad_full.sh",
    ]
    return {path: sha256_file(ROOT / path) for path in relative_paths}


def parse_epoch_set(value: str) -> set[int]:
    epochs = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not epochs:
        return set(DEFAULT_DIAGNOSTIC_DETAIL_EPOCHS)
    return epochs


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


def build_full_config(
    optuna_config: dict[str, Any],
    artifact_root: Path,
    sampled: dict[str, Any],
    max_epochs: int,
    patience: int,
) -> Config:
    overrides = copy.deepcopy(optuna_config["recbole_overrides"])
    overrides.update(
        {
            "checkpoint_dir": str(artifact_root / "recbole_checkpoints"),
            "epochs": int(max_epochs),
            "stopping_step": int(patience),
            "final_test_evaluation_count": 0,
            "test_evaluation_count": 0,
            "learning_rate": float(sampled["learning_rate"]),
            "weight_decay": float(sampled["weight_decay"]),
            "dropout_prob": float(sampled["dropout_prob"]),
            "metrics": ["Hit", "Recall", "NDCG"],
            "topk": list(METRIC_TOPK),
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


def ensure_locked_params(best_params: dict[str, Any]) -> None:
    if str(best_params["study_name"]) != EXPECTED_STUDY:
        raise RuntimeError(f"Unexpected study: {best_params['study_name']} != {EXPECTED_STUDY}")
    if int(best_params["trial_number"]) != EXPECTED_TRIAL:
        raise RuntimeError(f"Unexpected trial: {best_params['trial_number']} != {EXPECTED_TRIAL}")


def project_ranking_anchored(
    vectors: dict[str, torch.Tensor],
    projector: PCGradProjector,
    *,
    collect_diagnostics: bool,
) -> dict[str, Any]:
    if projector.mode != "ranking_anchored":
        raise RuntimeError(f"Full pcgrad_001 must use ranking_anchored mode, got {projector.mode}")

    before_matrix = cosine_matrix(vectors, TASK_ORDER) if collect_diagnostics else None
    before_conflicts = conflict_summary(before_matrix, TASK_ORDER) if before_matrix is not None else None
    adjusted: dict[str, torch.Tensor] = {"rank": vectors["rank"].clone()}
    projection_events: list[dict[str, Any]] = []
    rank = vectors["rank"]
    for task in TASK_ORDER:
        if task == "rank":
            continue
        projected, changed = projector._project_one(vectors[task].clone(), rank)
        adjusted[task] = projected
        if changed:
            projection_events.append({"source": task, "reference": "rank"})

    after_matrix = cosine_matrix(adjusted, TASK_ORDER) if collect_diagnostics else None
    after_conflicts = conflict_summary(after_matrix, TASK_ORDER) if after_matrix is not None else None
    combined = torch.stack([adjusted[task] for task in TASK_ORDER]).sum(dim=0)
    original_combined = torch.stack([vectors[task] for task in TASK_ORDER]).sum(dim=0)
    payload: dict[str, Any] = {
        "mode": projector.mode,
        "combined_gradient": combined,
        "projection_events": projection_events,
        "projection_event_count": len(projection_events),
        "projection_event_count_by_target": {
            target: sum(1 for item in projection_events if item["source"] == target) for target in AUX_TARGETS
        },
        "combined_gradient_norm_before": float(torch.linalg.vector_norm(original_combined.float()).cpu().item()),
        "combined_gradient_norm_after": float(torch.linalg.vector_norm(combined.float()).cpu().item()),
    }
    if collect_diagnostics:
        payload.update(
            {
                "cosine_matrix_before": before_matrix,
                "cosine_matrix_after": after_matrix,
                "conflicts_before": before_conflicts,
                "conflicts_after": after_conflicts,
                "gradient_norms_before": gradient_norms(vectors, TASK_ORDER),
                "gradient_norms_after": gradient_norms(adjusted, TASK_ORDER),
            }
        )
    return payload


def update_batch_conflict_frequency(
    stats: dict[str, Any],
    before_matrix: dict[str, dict[str, float | None]],
    after_matrix: dict[str, dict[str, float | None]],
    projection_events: list[dict[str, Any]],
) -> None:
    stats["sample_batches"] += 1
    projected = {str(item["source"]) for item in projection_events}
    any_before = False
    any_after = False
    for target in AUX_TARGETS:
        before = before_matrix["rank"][target]
        after = after_matrix["rank"][target]
        if before is not None and before < 0:
            stats["rank_aux_conflict_batches_before_by_target"][target] += 1
            any_before = True
        if after is not None and after < 0:
            stats["rank_aux_conflict_batches_after_by_target"][target] += 1
            any_after = True
        if target in projected:
            stats["projection_batches_by_target"][target] += 1
    if any_before:
        stats["batches_with_any_rank_aux_conflict_before"] += 1
    if any_after:
        stats["batches_with_any_rank_aux_conflict_after"] += 1
    if projected:
        stats["batches_with_projection"] += 1


def empty_frequency_stats() -> dict[str, Any]:
    return {
        "sample_batches": 0,
        "rank_aux_conflict_batches_before_by_target": {target: 0 for target in AUX_TARGETS},
        "rank_aux_conflict_batches_after_by_target": {target: 0 for target in AUX_TARGETS},
        "projection_batches_by_target": {target: 0 for target in AUX_TARGETS},
        "batches_with_any_rank_aux_conflict_before": 0,
        "batches_with_any_rank_aux_conflict_after": 0,
        "batches_with_projection": 0,
    }


def finalize_frequency_stats(stats: dict[str, Any]) -> dict[str, Any]:
    sample_batches = int(stats["sample_batches"])

    def fractions(source: dict[str, int]) -> dict[str, float]:
        return {target: (int(value) / sample_batches if sample_batches else 0.0) for target, value in source.items()}

    return {
        **stats,
        "sample_batches": sample_batches,
        "rank_aux_conflict_fraction_before_by_target": fractions(stats["rank_aux_conflict_batches_before_by_target"]),
        "rank_aux_conflict_fraction_after_by_target": fractions(stats["rank_aux_conflict_batches_after_by_target"]),
        "projection_fraction_by_target": fractions(stats["projection_batches_by_target"]),
        "any_rank_aux_conflict_fraction_before": (
            int(stats["batches_with_any_rank_aux_conflict_before"]) / sample_batches if sample_batches else 0.0
        ),
        "any_rank_aux_conflict_fraction_after": (
            int(stats["batches_with_any_rank_aux_conflict_after"]) / sample_batches if sample_batches else 0.0
        ),
        "projection_batch_fraction": int(stats["batches_with_projection"]) / sample_batches if sample_batches else 0.0,
    }


def pcgrad_step(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    interaction: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    shared_entries: list[Any],
    projector: PCGradProjector,
    *,
    collect_diagnostics: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    losses = compute_tuned_losses(model, interaction, sampled, pos_weights)
    raw_aux, task_contributions = split_tuned_losses(losses)
    vectors = task_gradient_vectors(task_contributions, shared_entries, TASK_ORDER)
    projection = project_ranking_anchored(vectors, projector, collect_diagnostics=collect_diagnostics)

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
        "projection_event_count_by_target": projection["projection_event_count_by_target"],
        "combined_gradient_norm_before": projection["combined_gradient_norm_before"],
        "combined_gradient_norm_after": projection["combined_gradient_norm_after"],
        "shared_gradient_finite": shared_finite,
    }
    compact = {
        "method_effect": method_effect,
        "diagnostic_losses": losses,
    }
    if collect_diagnostics:
        compact.update(
            {
                "cosine_matrix_before": projection["cosine_matrix_before"],
                "cosine_matrix_after": projection["cosine_matrix_after"],
                "gradient_norms_before": projection["gradient_norms_before"],
                "gradient_norms_after": projection["gradient_norms_after"],
                "conflicts_before": projection["conflicts_before"],
                "conflicts_after": projection["conflicts_after"],
            }
        )
    return scalar_losses(refreshed), compact


def train_one_epoch_pcgrad(
    *,
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    device: torch.device,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    shared_entries: list[Any],
    projector: PCGradProjector,
    epoch: int,
    diagnostic_batches: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model.train()
    sums: dict[str, float] = {}
    examples = 0
    batches = 0
    sample_stats = empty_conflict_stats()
    frequency = empty_frequency_stats()
    all_projection_events = 0
    all_projection_events_by_target = {target: 0 for target in AUX_TARGETS}
    first_batch_diagnostic = None

    for batch_idx, interaction in enumerate(train_data):
        interaction = interaction.to(device)
        batch_size = len(interaction)
        collect = batch_idx < diagnostic_batches
        loss_scalars, step_diag = pcgrad_step(
            model,
            optimizer,
            interaction,
            sampled,
            pos_weights,
            shared_entries,
            projector,
            collect_diagnostics=collect,
        )

        for key, value in loss_scalars.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
        examples += batch_size
        batches += 1

        effect = step_diag["method_effect"]
        all_projection_events += int(effect["projection_event_count"])
        for target, value in effect["projection_event_count_by_target"].items():
            all_projection_events_by_target[target] += int(value)

        if collect:
            update_conflict_stats(
                sample_stats,
                step_diag["cosine_matrix_before"],
                step_diag["cosine_matrix_after"],
                step_diag["conflicts_before"],
                step_diag["conflicts_after"],
                effect,
            )
            update_batch_conflict_frequency(
                frequency,
                step_diag["cosine_matrix_before"],
                step_diag["cosine_matrix_after"],
                effect["projection_events"],
            )
            if batch_idx == 0:
                first_batch_diagnostic = diagnostic_record(
                    method="pcgrad",
                    epoch=epoch,
                    batch_idx=batch_idx,
                    losses=step_diag["diagnostic_losses"],
                    before_matrix=step_diag["cosine_matrix_before"],
                    after_matrix=step_diag["cosine_matrix_after"],
                    before_norms=step_diag["gradient_norms_before"],
                    after_norms=step_diag["gradient_norms_after"],
                    before_conflicts=step_diag["conflicts_before"],
                    after_conflicts=step_diag["conflicts_after"],
                    method_effect=effect,
                )

    if examples == 0 or first_batch_diagnostic is None:
        raise RuntimeError("No training examples or diagnostic batch.")
    losses = {key: value / examples for key, value in sums.items()}
    rank = losses["rank"]
    losses["auxiliary_scaled_contribution"] = float(sampled["lambda_aux"]) * losses["weighted_aux_sum"]
    losses["auxiliary_rank_ratio"] = losses["auxiliary_scaled_contribution"] / rank if rank else None
    losses["per_task_rank_ratio"] = {
        target: losses[f"{target}_scaled_contribution"] / rank if rank else None for target in AUX_TARGETS
    }
    losses["batches"] = batches
    losses["examples"] = examples

    epoch_conflicts = finalize_conflict_stats(sample_stats)
    frequency_summary = finalize_frequency_stats(frequency)
    epoch_conflicts["diagnostic_sample_policy"] = {
        "sampled_batches_per_epoch": min(int(diagnostic_batches), int(batches)),
        "sampled_from": "first_n_train_batches_each_epoch",
        "full_cosine_matrix_computed_for_every_train_batch": False,
    }
    epoch_conflicts["rank_aux_batch_conflict_frequency"] = frequency_summary
    epoch_conflicts["projection_events_all_train_batches"] = int(all_projection_events)
    epoch_conflicts["projection_events_all_train_batches_by_target"] = all_projection_events_by_target
    epoch_conflicts["train_batches"] = int(batches)
    epoch_conflicts["projection_events_per_train_batch"] = all_projection_events / batches if batches else 0.0
    return losses, epoch_conflicts, first_batch_diagnostic


def selected_diagnostics(
    diagnostics_by_epoch: dict[int, dict[str, Any]],
    fixed_epochs: set[int],
    best_epoch: int,
    last_epoch: int,
) -> list[dict[str, Any]]:
    selected = sorted({epoch for epoch in fixed_epochs if epoch in diagnostics_by_epoch} | {best_epoch, last_epoch})
    return [diagnostics_by_epoch[epoch] for epoch in selected if epoch in diagnostics_by_epoch]


def selected_frequency(
    frequency_by_epoch: dict[int, dict[str, Any]],
    fixed_epochs: set[int],
    best_epoch: int,
    last_epoch: int,
) -> list[dict[str, Any]]:
    selected = sorted({epoch for epoch in fixed_epochs if epoch in frequency_by_epoch} | {best_epoch, last_epoch})
    return [frequency_by_epoch[epoch] for epoch in selected if epoch in frequency_by_epoch]


def summarize_is_like(diagnostics: list[dict[str, Any]], frequency: list[dict[str, Any]], sampled: dict[str, Any]) -> dict[str, Any]:
    by_epoch = {int(item["epoch"]): item for item in frequency}
    points = []
    for diag in diagnostics:
        epoch = int(diag["epoch"])
        freq = by_epoch.get(epoch, {}).get("rank_aux_batch_conflict_frequency", {})
        points.append(
            {
                "epoch": epoch,
                **diag["is_like"],
                "sample_conflict_fraction_before": freq.get("rank_aux_conflict_fraction_before_by_target", {}).get(
                    "is_like"
                ),
                "sample_conflict_fraction_after": freq.get("rank_aux_conflict_fraction_after_by_target", {}).get(
                    "is_like"
                ),
                "sample_projection_fraction": freq.get("projection_fraction_by_target", {}).get("is_like"),
            }
        )
    return {
        "tuned_fixed_weight": sampled["normalized_task_weights"]["is_like"],
        "effective_pos_weight": sampled["effective_pos_weights"]["is_like"],
        "diagnostic_points": points,
    }


def compare_to_references(best_metrics: dict[str, float], references: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    mapping = {
        "tim4rec_001": "tim4rec_001_full_reference",
        "multitask_tim4rec_001": "multitask_tim4rec_001_full_reference",
        "multitask_tim4rec_tuned_001": "multitask_tim4rec_tuned_001_validation_reproduction",
    }
    for label, key in mapping.items():
        metrics = references[key]["validation_metrics"]
        ref_ndcg = float(metrics["NDCG@10"])
        delta = float(best_metrics["NDCG@10"]) - ref_ndcg
        rows[label] = {
            "run_id": references[key]["run_id"],
            "run_type": references[key]["run_type"],
            "best_epoch": references[key].get("best_epoch"),
            "validation_metrics": metrics,
            "pcgrad_minus_reference_ndcg10": delta,
            "pcgrad_minus_reference_ndcg10_relative_pct": 100.0 * delta / ref_ndcg if ref_ndcg else None,
        }
    tuned = rows["multitask_tim4rec_tuned_001"]
    delta = float(tuned["pcgrad_minus_reference_ndcg10"])
    tuned["decision_threshold_note"] = "practically_tied_or_marginal" if abs(delta) < 0.0005 else "visible_delta"
    return rows


def metric_row(label: str, metrics: dict[str, float], best_epoch: Any = "") -> str:
    return (
        f"| {label} | {best_epoch} | {metrics['HR@5']:.4f} | {metrics['HR@10']:.4f} | "
        f"{metrics['HR@20']:.4f} | {metrics['HR@50']:.4f} | {metrics['Recall@5']:.4f} | "
        f"{metrics['Recall@10']:.4f} | {metrics['Recall@20']:.4f} | {metrics['Recall@50']:.4f} | "
        f"{metrics['NDCG@5']:.4f} | {metrics['NDCG@10']:.4f} | {metrics['NDCG@20']:.4f} | "
        f"{metrics['NDCG@50']:.4f} |"
    )


def build_notes(result: dict[str, Any]) -> str:
    best = result["best_validation"]
    references = result["reference_comparison"]
    tuned_delta = references["multitask_tim4rec_tuned_001"]["pcgrad_minus_reference_ndcg10"]
    tuned_rel = references["multitask_tim4rec_tuned_001"]["pcgrad_minus_reference_ndcg10_relative_pct"]
    diagnostics = result["gradient_diagnostics"]
    frequency = {int(item["epoch"]): item for item in result["conflict_frequency"]["selected_epochs"]}
    lines = [
        "# PCGrad full validation run 001",
        "",
        "## Цель",
        "",
        "Проверить, способен ли ranking-anchored PCGrad при полном validation-only обучении превзойти tuned fixed MultitaskTiM4Rec по validation `NDCG@10`.",
        "",
        "## Base configuration",
        "",
        f"- Base run: `{result['base_run']}`.",
        f"- Study: `{result['tuned_fixed_configuration']['source_study']}`.",
        f"- Trial: `{result['tuned_fixed_configuration']['source_trial']}`.",
        f"- `lambda_aux`: `{result['tuned_fixed_configuration']['lambda_aux']:.12g}`.",
        f"- `learning_rate`: `{result['tuned_fixed_configuration']['learning_rate']:.12g}`.",
        f"- `weight_decay`: `{result['tuned_fixed_configuration']['weight_decay']:.12g}`.",
        f"- `dropout_prob`: `{result['tuned_fixed_configuration']['dropout_prob']:.12g}`.",
        f"- `head_lr_multiplier`: `{result['tuned_fixed_configuration']['head_lr_multiplier']:.12g}`.",
        "",
        "## PCGrad algorithm",
        "",
        f"- Anchor gradient: `{result['method']['anchor_gradient']}`.",
        f"- Projected gradients: `{', '.join(result['method']['projected_gradients'])}`.",
        f"- Auxiliary-auxiliary conflicts processed: `{str(result['method']['auxiliary_auxiliary_conflicts_processed']).lower()}`.",
        f"- Projection order: `{result['method']['projection_order']}`.",
        f"- Seed: `{result['method']['seed']}`.",
        "",
        "## Dataset",
        "",
        f"- Protocol identity hash: `{result['dataset']['validation_only_summary']['identity_hash']}`.",
        f"- Train rows: `{result['dataset']['validation_only_summary']['rows']['train']}`.",
        f"- Validation rows: `{result['dataset']['validation_only_summary']['rows']['validation']}`.",
        f"- Test rows in validation-only benchmark: `{result['dataset']['validation_only_summary']['rows']['test']}`.",
        "",
        "## Training",
        "",
        f"- Requested epochs: `{result['requested_epochs']}`.",
        f"- Actual epochs: `{result['actual_epochs']}`.",
        f"- Best epoch: `{result['best_epoch']}`.",
        f"- Stop reason: `{result['stop_reason']}`.",
        "",
        "## Validation trajectory",
        "",
        "| epoch | HR@10 | NDCG@10 | best so far | improved | no improve | train sec | valid sec |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for epoch in result["epochs"]:
        valid = epoch["validation_metrics"]
        lines.append(
            f"| {epoch['epoch']} | {valid['HR@10']:.4f} | {valid['NDCG@10']:.4f} | "
            f"{epoch['best_so_far']['NDCG@10']:.4f} | {str(epoch['improved']).lower()} | "
            f"{epoch['no_improve_count']:.0f} | {epoch['train_time_sec']:.1f} | {epoch['validation_time_sec']:.1f} |"
        )
    lines += [
        "",
        "## Best validation",
        "",
        "| Method | Best epoch | HR@5 | HR@10 | HR@20 | HR@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("PCGrad", best["metrics"], result["best_epoch"]),
        "",
        "## Auxiliary tasks",
        "",
        "| target | ROC-AUC | PR-AUC | BCE | positive rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, metrics in result["best_auxiliary_metrics"].items():
        lines.append(
            f"| `{target}` | {format_float(metrics['roc_auc'])} | {format_float(metrics['pr_auc'])} | "
            f"{format_float(metrics['bce_loss'])} | {format_float(metrics['positive_rate'])} |"
        )
    lines += [
        "",
        "## Gradient conflicts",
        "",
        f"- Diagnostic sample: `{result['conflict_frequency']['sample_policy']['sampled_batches_per_epoch']}` train batches per epoch.",
        f"- Projection events across all train batches: `{result['conflict_frequency']['projection_events_all_train_batches']}`.",
        f"- Mean projection events per train batch: `{result['conflict_frequency']['projection_events_per_train_batch']:.6f}`.",
        "",
        "| epoch | rank-aux before | rank-aux after | any conflict before | any conflict after | projection batch fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["conflict_frequency"]["selected_epochs"]:
        freq = item["rank_aux_batch_conflict_frequency"]
        lines.append(
            f"| {item['epoch']} | {item['rank_aux_fraction_before']:.4f} | {item['rank_aux_fraction_after']:.4f} | "
            f"{freq['any_rank_aux_conflict_fraction_before']:.4f} | {freq['any_rank_aux_conflict_fraction_after']:.4f} | "
            f"{freq['projection_batch_fraction']:.4f} |"
        )
    lines += [
        "",
        "## is_like analysis",
        "",
        f"- Tuned fixed task weight: `{result['is_like_summary']['tuned_fixed_weight']:.6f}`.",
        f"- Effective pos weight: `{result['is_like_summary']['effective_pos_weight']:.6f}`.",
        "",
        "| epoch | raw loss | contribution | grad norm before | cosine before | cosine after | conflict fraction | projection fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in result["is_like_summary"]["diagnostic_points"]:
        lines.append(
            f"| {point['epoch']} | {point['raw_loss']:.4f} | {point['weighted_or_effective_contribution']:.4f} | "
            f"{point['shared_gradient_norm_before']:.6f} | {point['cosine_with_ranking_before']:.6f} | "
            f"{point['cosine_with_ranking_after']:.6f} | "
            f"{point['sample_conflict_fraction_before']:.4f} | {point['sample_projection_fraction']:.4f} |"
        )
    lines += [
        "",
        "## Comparison with tuned fixed MultitaskTiM4Rec",
        "",
        "| Reference | Best epoch | HR@10 | NDCG@10 | PCGrad minus reference NDCG@10 | Relative delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, payload in references.items():
        metrics = payload["validation_metrics"]
        rel = payload["pcgrad_minus_reference_ndcg10_relative_pct"]
        rel_text = "" if rel is None else f"{rel:.2f}%"
        lines.append(
            f"| `{label}` | {payload.get('best_epoch', '')} | {metrics['HR@10']:.4f} | {metrics['NDCG@10']:.4f} | "
            f"{payload['pcgrad_minus_reference_ndcg10']:.4f} | {rel_text} |"
        )
    lines += [
        "",
        f"PCGrad vs tuned fixed: absolute delta `{tuned_delta:.6f}`, relative delta `{tuned_rel:.2f}%`.",
        "",
        "## Computational cost",
        "",
        f"- Slurm job: `{result['slurm']['job_id']}`.",
        f"- Partition: `{result['slurm']['partition']}`.",
        f"- Constraint: `{result['slurm']['constraint']}`.",
        f"- Node: `{result['slurm']['node_list']}`.",
        f"- GPU: `{result['gpu']['name']}`.",
        f"- Mean train epoch: `{result['cost']['mean_train_epoch_time_sec']:.3f}` sec.",
        f"- Mean validation: `{result['cost']['mean_validation_time_sec']:.3f}` sec.",
        f"- Peak VRAM: `{result['gpu']['peak_allocated_bytes']}` bytes.",
        f"- Process MaxRSS: `{result['memory']['process_ru_maxrss_kb']}` KB.",
        "",
        "## Test safety",
        "",
        "- `test_dataset_loaded=false`.",
        "- `test_dataloader_created=false`.",
        "- `test_evaluated=false`.",
        "- `test_evaluation_count=0`.",
        "",
        "## Вывод",
        "",
    ]
    if tuned_delta > 0:
        if abs(tuned_delta) < 0.0005:
            lines.append("PCGrad формально выше tuned fixed на validation, но разница marginal; без multi-seed это нельзя трактовать как значимое улучшение.")
        else:
            lines.append("PCGrad выше tuned fixed на validation; перед locked test всё равно нужен отдельный decision gate и желательно multi-seed проверка.")
    else:
        lines.append("PCGrad не превзошёл tuned fixed MultitaskTiM4Rec на validation; locked test для PCGrad на этом основании открывать не следует.")
    return "\n".join(lines)


def compact_epoch_for_partial(epoch_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": epoch_result["epoch"],
        "validation_metrics": {
            "HR@10": epoch_result["validation_metrics"]["HR@10"],
            "NDCG@10": epoch_result["validation_metrics"]["NDCG@10"],
            "NDCG@20": epoch_result["validation_metrics"]["NDCG@20"],
            "NDCG@50": epoch_result["validation_metrics"]["NDCG@50"],
        },
        "best_so_far": epoch_result["best_so_far"],
        "improved": epoch_result["improved"],
        "no_improve_count": epoch_result["no_improve_count"],
        "train_time_sec": epoch_result["train_time_sec"],
        "validation_time_sec": epoch_result["validation_time_sec"],
        "rank_aux_fraction_before_sample": epoch_result["gradient_conflicts"]["rank_aux_fraction_before"],
        "projection_events_all_train_batches": epoch_result["gradient_conflicts"][
            "projection_events_all_train_batches"
        ],
    }


def optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_resume_state(
    path: Path,
    *,
    run_started_utc: str,
    epochs: list[dict[str, Any]],
    diagnostics_by_epoch: dict[int, dict[str, Any]],
    frequency_by_epoch: dict[int, dict[str, Any]],
    best_epoch: int | None,
    best_score: float,
    best_snapshot: dict[str, Any] | None,
    best_checkpoint: dict[str, Any] | None,
    last_checkpoint: dict[str, Any] | None,
    no_improve: int,
    current_job_runtime_sec: float,
    previous_runtime_sec: float,
    slurm_history: list[dict[str, Any]],
    current_slurm: dict[str, Any],
) -> None:
    history = [item for item in slurm_history if item.get("job_id") != current_slurm.get("job_id")]
    if current_slurm.get("job_id"):
        history.append(current_slurm)
    save_json(
        path,
        {
            "run_id": RUN_ID,
            "status": "partial",
            "record_type": "experiment_validation_only",
            "run_started_utc": run_started_utc,
            "epochs": epochs,
            "diagnostics_by_epoch": {str(key): value for key, value in diagnostics_by_epoch.items()},
            "frequency_by_epoch": {str(key): value for key, value in frequency_by_epoch.items()},
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_snapshot": best_snapshot,
            "best_checkpoint": best_checkpoint,
            "last_checkpoint": last_checkpoint,
            "no_improve": int(no_improve),
            "runtime_sec_so_far": float(previous_runtime_sec + current_job_runtime_sec),
            "slurm_history": history,
            "test_evaluation_count": 0,
        },
    )


def load_resume_state(path: Path) -> dict[str, Any]:
    state = load_json(path)
    if state.get("run_id") != RUN_ID:
        raise RuntimeError(f"Resume state belongs to another run: {state.get('run_id')}")
    if int(state.get("test_evaluation_count", 0)) != 0:
        raise RuntimeError(f"Resume state has non-zero test_evaluation_count: {state.get('test_evaluation_count')}")
    return state


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise RuntimeError(f"This runner is locked to run_id={RUN_ID}, got {args.run_id}")
    if int(args.max_epochs) != 300:
        raise RuntimeError(f"pcgrad_001 requires max_epochs=300, got {args.max_epochs}")
    if int(args.patience) != 10:
        raise RuntimeError(f"pcgrad_001 requires patience=10, got {args.patience}")
    if int(args.diagnostic_batches) < 5 or int(args.diagnostic_batches) > 10:
        raise RuntimeError("diagnostic_batches must stay in the requested 5-10 range.")

    artifact_dir, result_json, notes_path = run_paths(args.run_id, args)
    partial_json = result_json.with_suffix(".partial.json")
    resume_state_path = artifact_dir / "resume_state.json"
    if result_json.exists():
        raise RuntimeError(f"Refusing to overwrite completed run artifact: {result_json}")
    if args.resume and not resume_state_path.exists():
        raise RuntimeError(f"--resume was requested but resume state is missing: {resume_state_path}")
    if not args.allow_overwrite and not args.resume:
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
    ensure_locked_params(best_params)
    data = load_data_bundle(optuna_config, artifact_dir / "data_probe")
    sampled = sampled_from_locked_params(best_params, data.target_stats)
    config = build_full_config(optuna_config, artifact_dir, sampled, int(args.max_epochs), int(args.patience))
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    if tuple(config["multitask_targets"]) != TARGETS:
        raise RuntimeError(f"Task set changed: {config['multitask_targets']}")
    if not bool(config["is_time"]):
        raise RuntimeError("TiM4Rec is_time must stay True.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for pcgrad_001.")
    if data.validation_only_summary["forbidden_test_paths_loaded"] != []:
        raise RuntimeError(f"Validation-only prep touched test paths: {data.validation_only_summary}")
    if int(data.validation_only_summary["rows"]["test"]) != 0:
        raise RuntimeError(f"Validation-only benchmark unexpectedly has test rows: {data.validation_only_summary}")
    if data.validation_only_summary.get("identity_hash") != EXPECTED_IDENTITY_HASH:
        raise RuntimeError(f"Identity hash mismatch: {data.validation_only_summary.get('identity_hash')}")

    train_data, valid_data = create_loaders(config, data.train_dataset, data.valid_dataset)
    device = config["device"]
    pos_weights = pos_weight_tensors(sampled["effective_pos_weights"], device)
    torch.cuda.reset_peak_memory_stats()

    model = MultitaskTiM4Rec(config, train_data.dataset).to(device)
    optimizer = optimizer_for_trial(model, sampled)
    trainer = Trainer(config, model)
    trainer.optimizer = optimizer
    shared_entries = shared_parameter_entries(model, str(adaptive_config["pcgrad"]["shared_selector"]))
    projector = PCGradProjector(
        mode=str(adaptive_config["pcgrad"]["mode"]),
        seed=int(adaptive_config["pcgrad"]["seed"]),
    )
    if projector.mode != "ranking_anchored":
        raise RuntimeError(f"pcgrad_001 must use ranking_anchored PCGrad, got {projector.mode}")

    fixed_detail_epochs = parse_epoch_set(args.diagnostic_detail_epochs)
    current_slurm = slurm_info()
    start = time.monotonic()
    run_started = datetime.now(timezone.utc).isoformat()
    previous_runtime_sec = 0.0
    slurm_history: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_score = -float("inf")
    best_snapshot: dict[str, Any] | None = None
    best_checkpoint = None
    last_checkpoint = None
    no_improve = 0
    stop_reason = "max_epochs_reached"
    epochs: list[dict[str, Any]] = []
    diagnostics_by_epoch: dict[int, dict[str, Any]] = {}
    frequency_by_epoch: dict[int, dict[str, Any]] = {}
    topk = list(config["topk"])
    start_epoch = 1

    if args.resume:
        state = load_resume_state(resume_state_path)
        checkpoint_ref = state.get("last_checkpoint") or {}
        checkpoint_path = Path(checkpoint_ref.get("path") or checkpoint_dir / "last.pth")
        if not checkpoint_path.exists():
            raise RuntimeError(f"Last checkpoint is missing for resume: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_state_to_device(optimizer, device)
        run_started = str(state["run_started_utc"])
        previous_runtime_sec = float(state.get("runtime_sec_so_far", 0.0))
        slurm_history = list(state.get("slurm_history", []))
        epochs = list(state["epochs"])
        diagnostics_by_epoch = {int(key): value for key, value in state.get("diagnostics_by_epoch", {}).items()}
        frequency_by_epoch = {int(key): value for key, value in state.get("frequency_by_epoch", {}).items()}
        best_epoch = None if state.get("best_epoch") is None else int(state["best_epoch"])
        best_score = float(state["best_score"])
        best_snapshot = state.get("best_snapshot")
        best_checkpoint = state.get("best_checkpoint")
        last_checkpoint = state.get("last_checkpoint")
        no_improve = int(state.get("no_improve", 0))
        start_epoch = len(epochs) + 1

    for epoch in range(start_epoch, int(args.max_epochs) + 1):
        epoch_start = time.monotonic()
        train_start = time.monotonic()
        losses, epoch_conflicts, first_diag = train_one_epoch_pcgrad(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            device=device,
            sampled=sampled,
            pos_weights=pos_weights,
            shared_entries=shared_entries,
            projector=projector,
            epoch=epoch,
            diagnostic_batches=int(args.diagnostic_batches),
        )
        train_time = time.monotonic() - train_start
        diagnostics_by_epoch[int(epoch)] = first_diag
        frequency_by_epoch[int(epoch)] = {"epoch": int(epoch), **epoch_conflicts}

        valid_start = time.monotonic()
        valid_result, full_checks = evaluate_full_sort_with_checks(trainer, valid_data, train_data)
        auxiliary_validation = evaluate_auxiliary(model, valid_data, device)
        validation_time = time.monotonic() - valid_start
        check_hit_recall_equal(valid_result, topk)
        if not full_checks["raw_scores_all_finite"] or not full_checks["positive_scores_all_finite"]:
            raise RuntimeError(f"Non-finite validation scores: {full_checks}")
        metrics = normalize_metrics(metric_subset(valid_result))
        valid_score = float(metrics["NDCG@10"])
        improved = valid_score > best_score + float(args.min_delta)
        if improved:
            best_epoch = int(epoch)
            best_score = valid_score
            no_improve = 0
            best_snapshot = {
                "epoch": int(epoch),
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
                "pcgrad",
            )
        else:
            no_improve += 1
        last_checkpoint = save_checkpoint(
            checkpoint_dir / "last.pth",
            model,
            optimizer,
            epoch,
            best_score,
            metrics,
            sampled,
            "pcgrad",
        )
        epoch_result = {
            "epoch": int(epoch),
            "losses": losses,
            "validation_metrics": metrics,
            "auxiliary_validation": auxiliary_validation,
            "valid_score": valid_score,
            "valid_metric": "NDCG@10",
            "improved": bool(improved),
            "no_improve_count": int(no_improve),
            "best_so_far": {"epoch": best_epoch, "NDCG@10": best_score},
            "hit_recall_equal_check": check_hit_recall_equal(valid_result, topk),
            "full_ranking_checks": full_checks,
            "gradient_conflicts": epoch_conflicts,
            "train_time_sec": float(train_time),
            "validation_time_sec": float(validation_time),
            "epoch_time_sec": float(time.monotonic() - epoch_start),
            "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()),
        }
        epochs.append(epoch_result)
        with training_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_result, ensure_ascii=False, default=json_default) + "\n")
        save_json(
            partial_json,
            {
                "run_id": RUN_ID,
                "status": "partial",
                "record_type": "experiment_validation_only",
                "method": "ranking_anchored_pcgrad",
                "epochs_completed": len(epochs),
                "latest_epoch": compact_epoch_for_partial(epoch_result),
                "best_epoch_so_far": best_epoch,
                "best_valid_score_so_far": best_score,
                "no_improve_count": no_improve,
                "patience": int(args.patience),
                "test_evaluation_count": 0,
            },
        )
        save_resume_state(
            resume_state_path,
            run_started_utc=run_started,
            epochs=epochs,
            diagnostics_by_epoch=diagnostics_by_epoch,
            frequency_by_epoch=frequency_by_epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            best_snapshot=best_snapshot,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
            no_improve=no_improve,
            current_job_runtime_sec=time.monotonic() - start,
            previous_runtime_sec=previous_runtime_sec,
            slurm_history=slurm_history,
            current_slurm=current_slurm,
        )
        print(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "epoch": int(epoch),
                    "validation_ndcg10": metrics["NDCG@10"],
                    "validation_hr10": metrics["HR@10"],
                    "best_epoch": best_epoch,
                    "best_valid_score": best_score,
                    "no_improve_count": no_improve,
                    "train_time_sec": train_time,
                    "validation_time_sec": validation_time,
                    "projection_events_all_train_batches": epoch_conflicts["projection_events_all_train_batches"],
                    "rank_aux_conflict_fraction_sample": epoch_conflicts["rank_aux_fraction_before"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if no_improve >= int(args.patience):
            stop_reason = f"early_stopping_no_validation_ndcg10_improvement_{int(args.patience)}"
            break
        if args.soft_time_limit_sec > 0:
            elapsed = time.monotonic() - start
            current_job_epochs = max(len(epochs) - start_epoch + 1, 1)
            mean_epoch = sum(item["epoch_time_sec"] for item in epochs[start_epoch - 1 :]) / current_job_epochs
            if elapsed + max(mean_epoch, 90.0) >= float(args.soft_time_limit_sec):
                print(
                    json.dumps(
                        {
                            "run_id": RUN_ID,
                            "status": "partial_soft_time_limit",
                            "epochs_completed": len(epochs),
                            "best_epoch_so_far": best_epoch,
                            "best_valid_score_so_far": best_score,
                            "elapsed_sec": elapsed,
                            "mean_epoch_sec_current_job": mean_epoch,
                            "test_evaluation_count": 0,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return

    if best_snapshot is None or best_epoch is None:
        raise RuntimeError("No validation snapshot recorded.")

    last_epoch = int(epochs[-1]["epoch"])
    diagnostics = selected_diagnostics(diagnostics_by_epoch, fixed_detail_epochs, best_epoch, last_epoch)
    selected_freq = selected_frequency(frequency_by_epoch, fixed_detail_epochs, best_epoch, last_epoch)
    total_projection_events = sum(int(epoch["gradient_conflicts"]["projection_events_all_train_batches"]) for epoch in epochs)
    total_batches = sum(int(epoch["gradient_conflicts"]["train_batches"]) for epoch in epochs)
    references = load_reference_metrics()
    comparison = compare_to_references(best_snapshot["metrics"], references)
    runtime_sec = previous_runtime_sec + (time.monotonic() - start)
    result: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": "completed",
        "record_type": "experiment_validation_only",
        "sanity": False,
        "objective": "validation_full_ranking_NDCG@10",
        "base_model": "MultitaskTiM4Rec",
        "base_run": "multitask_tim4rec_tuned_001",
        "created_at_utc": run_started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": git_value(["config", "--get", "remote.origin.url"]),
            "expected_start_commit": "5e5ae42",
        },
        "source_files": source_hashes(),
        "environment": environment_info(),
        "slurm": current_slurm,
        "slurm_history": [item for item in slurm_history if item.get("job_id") != current_slurm.get("job_id")]
        + ([current_slurm] if current_slurm.get("job_id") else []),
        "gpu": gpu_info()
        | {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "memory": {
            "process_ru_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "method": {
            "name": "PCGrad",
            "method": "ranking_anchored_pcgrad",
            "variant": "ranking_anchored",
            "shared_selector": str(adaptive_config["pcgrad"]["shared_selector"]),
            "anchor_gradient": "rank",
            "projected_gradients": list(AUX_TARGETS),
            "auxiliary_auxiliary_conflicts_processed": False,
            "projection_order": "fixed_TASK_ORDER_auxiliaries_after_rank",
            "random_order_used": False,
            "seed": projector.seed,
            "algorithm": "g_rank is unchanged; each auxiliary gradient is projected only if dot(g_aux, g_rank) < 0; auxiliary-auxiliary conflicts are not processed.",
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
            "shared": {
                "parameter_tensors": len(shared_entries),
                "parameter_count": int(sum(entry.parameter.numel() for entry in shared_entries)),
                "first_names": [entry.name for entry in shared_entries[:8]],
                "last_names": [entry.name for entry in shared_entries[-8:]],
            },
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
        "training_config": {
            "requested_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "valid_metric": "NDCG@10",
            "mode": "maximize",
            "checkpoint_selection": "validation_NDCG@10_only",
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "is_time": bool(config["is_time"]),
            "metrics": list(config["metrics"]),
            "topk": topk,
            "eval_args": config["eval_args"],
        },
        "requested_epochs": int(args.max_epochs),
        "actual_epochs": len(epochs),
        "best_epoch": best_epoch,
        "best_valid_score": best_score,
        "best_valid_metric": "NDCG@10",
        "stop_reason": stop_reason,
        "epochs": epochs,
        "validation_history": [
            {
                "epoch": item["epoch"],
                "validation_metrics": item["validation_metrics"],
                "valid_score": item["valid_score"],
                "improved": item["improved"],
                "best_so_far": item["best_so_far"],
            }
            for item in epochs
        ],
        "loss_history": [{"epoch": item["epoch"], **item["losses"]} for item in epochs],
        "best_validation": best_snapshot,
        "best_validation_metrics": best_snapshot["metrics"],
        "best_validation_compact": best_snapshot["compact_metrics"],
        "best_auxiliary_metrics": best_snapshot["auxiliary_validation"],
        "gradient_diagnostics": diagnostics,
        "conflict_frequency": {
            "sample_policy": {
                "sampled_batches_per_epoch": int(args.diagnostic_batches),
                "sampled_from": "first_n_train_batches_each_epoch",
                "detail_epochs": sorted(fixed_detail_epochs),
                "final_selected_epochs": [item["epoch"] for item in selected_freq],
                "full_cosine_matrix_computed_for_every_train_batch": False,
            },
            "selected_epochs": selected_freq,
            "projection_events_all_train_batches": int(total_projection_events),
            "train_batches_total": int(total_batches),
            "projection_events_per_train_batch": total_projection_events / total_batches if total_batches else 0.0,
        },
        "is_like_summary": summarize_is_like(diagnostics, selected_freq, sampled),
        "reference_comparison": comparison,
        "baseline_comparison": references,
        "checkpoints": {
            "best_validation": best_checkpoint,
            "last": last_checkpoint,
        },
        "artifact_dir": str(artifact_dir),
        "remote_artifact_path": str(artifact_dir),
        "runtime": {
            "total_sec": runtime_sec,
            "mean_epoch_sec": sum(item["epoch_time_sec"] for item in epochs) / len(epochs),
        },
        "cost": {
            "mean_train_epoch_time_sec": sum(item["train_time_sec"] for item in epochs) / len(epochs),
            "mean_validation_time_sec": sum(item["validation_time_sec"] for item in epochs) / len(epochs),
            "mean_epoch_time_sec": sum(item["epoch_time_sec"] for item in epochs) / len(epochs),
            "fixed_tuned_smoke_mean_step_sec": 0.10299260293443997,
            "pcgrad_smoke_mean_step_sec": 0.1972333835437894,
            "overhead_vs_fixed_tuned_smoke": 0.1972333835437894 / 0.10299260293443997,
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
