#!/usr/bin/env python
"""Smoke and 5-epoch sanity training for MultitaskTiM4Rec.

This is intentionally not a full training script. It keeps the validated
TiM4Rec backbone and adds only fixed-weight auxiliary behavior heads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from typing import Any, Iterable

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Collector, Evaluator
from recbole.trainer import Trainer
from recbole.utils import early_stopping, init_seed


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tim4rec import TiM4Rec  # noqa: E402
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS  # noqa: E402


RUN_ID = "multitask_tim4rec_sanity_001"
EXPECTED_FINGERPRINT = {
    "users": 23951,
    "items": 7111,
    "interactions": 1134420,
    "train": 1086518,
    "validation": 23951,
    "test": 23951,
}
EXPECTED_IDENTITY_HASH = "954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7"
DEFAULT_MULTITASK_DIR = Path("/home/daryumin/iberdov/diplom/data/processed/protocol_b_multitask")
DEFAULT_ARTIFACT_DIR = Path(
    "/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec/multitask_tim4rec_sanity_001"
)
DEFAULT_MANIFEST = ROOT / "outputs" / "data" / "protocol_b_multitask_manifest.json"
DEFAULT_BASE_RUN = Path("/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/runs/tim4rec_001.json")
DEFAULT_BASE_SANITY = Path(
    "/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/runs/tim4rec_sanity_001.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments" / "multitask_tim4rec" / "config.yaml"))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--result-json", default=str(ROOT / "experiments" / "multitask_tim4rec" / "runs" / f"{RUN_ID}.json"))
    parser.add_argument("--notes", default=str(ROOT / "experiments" / "multitask_tim4rec" / "runs" / f"{RUN_ID}_notes.md"))
    parser.add_argument("--multitask-dir", default=str(DEFAULT_MULTITASK_DIR))
    parser.add_argument("--multitask-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--target-statistics", default=str(ROOT / "experiments" / "multitask_tim4rec" / "target_statistics.csv"))
    parser.add_argument("--base-run-json", default=str(DEFAULT_BASE_RUN))
    parser.add_argument("--base-sanity-json", default=str(DEFAULT_BASE_SANITY))
    parser.add_argument("--lambda-aux", type=float, default=None)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_size(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_multitask_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    fingerprint = manifest["dataset_fingerprint"]
    observed = {
        "users": int(fingerprint["users"]),
        "items": int(fingerprint["items"]),
        "interactions": int(fingerprint["interactions"]),
        "train": int(fingerprint["split_counts"]["train"]),
        "validation": int(fingerprint["split_counts"]["validation"]),
        "test": int(fingerprint["split_counts"]["test"]),
    }
    if observed != EXPECTED_FINGERPRINT:
        raise RuntimeError(f"Multitask Protocol B fingerprint mismatch: {observed}")
    identity_hash = fingerprint["identity_hash_user_item_timestamp_split"]
    if identity_hash != EXPECTED_IDENTITY_HASH:
        raise RuntimeError(f"Identity hash mismatch: {identity_hash} != {EXPECTED_IDENTITY_HASH}")
    if not manifest["join_diagnostics"]["join_is_exact"]:
        raise RuntimeError(f"Join was not exact: {manifest['join_diagnostics']}")
    return manifest


def ensure_recbole_inter(multitask_dir: Path) -> dict[str, Any]:
    dataset_dir = multitask_dir / "recbole" / "kuairand_multitask"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    inter_path = dataset_dir / "kuairand_multitask.inter"
    validation_ids_path = dataset_dir / "validation_source_row_ids.txt"
    if inter_path.exists():
        if not validation_ids_path.exists():
            try:
                import polars as pl

                validation_ids = (
                    pl.read_parquet(multitask_dir / "validation.parquet", columns=["source_row_id"])
                    .select("source_row_id")
                    .sort("source_row_id")
                )
                validation_ids_path.write_text(
                    "\n".join(str(int(value)) for value in validation_ids["source_row_id"].to_list()) + "\n",
                    encoding="utf-8",
                )
            except ModuleNotFoundError:
                pass
        with inter_path.open("r", encoding="utf-8") as handle:
            rows = max(sum(1 for _line in handle) - 1, 0)
        return {
            "path": str(inter_path),
            "rows": rows,
            "size_bytes": inter_path.stat().st_size,
            "size": human_size(inter_path.stat().st_size),
            "sha256": sha256_file(inter_path),
            "validation_source_row_ids_path": str(validation_ids_path),
            "validation_source_row_ids_available": validation_ids_path.exists(),
        }

    import polars as pl

    full_path = multitask_dir / "full_filtered.parquet"
    if not full_path.exists():
        raise FileNotFoundError(f"Missing multitask parquet: {full_path}")
    df = (
        pl.read_parquet(full_path)
        .select([
            "user_id",
            "item_id",
            "timestamp",
            "source_row_id",
            "is_click",
            "long_view",
            "is_like",
            "is_profile_enter",
        ])
        .sort(["user_id", "timestamp", "source_row_id", "item_id"])
    )
    typed = df.rename(
        {
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
            "source_row_id": "source_row_id:float",
            "is_click": "is_click:float",
            "long_view": "long_view:float",
            "is_like": "is_like:float",
            "is_profile_enter": "is_profile_enter:float",
        }
    )
    typed.write_csv(inter_path, separator="\t")
    validation_ids = (
        pl.read_parquet(multitask_dir / "validation.parquet", columns=["source_row_id"])
        .select("source_row_id")
        .sort("source_row_id")
    )
    validation_ids_path.write_text(
        "\n".join(str(int(value)) for value in validation_ids["source_row_id"].to_list()) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(inter_path),
        "rows": int(df.height),
        "size_bytes": inter_path.stat().st_size,
        "size": human_size(inter_path.stat().st_size),
        "sha256": sha256_file(inter_path),
        "validation_source_row_ids_path": str(validation_ids_path),
        "validation_source_row_ids_available": True,
    }


def load_target_stats(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["scope"] != "train" or row["field"] not in TARGETS:
            continue
        positives = float(row["positives"])
        negatives = float(row["negatives"])
        stats[row["field"]] = {
            "rows": float(row["rows"]),
            "positives": positives,
            "negatives": negatives,
            "positive_rate": float(row["positive_rate"]),
            "negative_positive_ratio": negatives / positives,
        }
    missing = set(TARGETS).difference(stats)
    if missing:
        raise RuntimeError(f"Missing target stats for {sorted(missing)} in {path}")
    return stats


def pos_weight_tensors(target_stats: dict[str, dict[str, float]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        target: torch.tensor([stats["negative_positive_ratio"]], dtype=torch.float32, device=device)
        for target, stats in target_stats.items()
    }


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    head = sum(
        param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad and any(token in name for token in ("click_head", "long_view_head", "like_head", "profile_enter_head"))
    )
    return {"total": int(total), "trainable": int(trainable), "auxiliary_heads": int(head)}


def backbone_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in ("click_head", "long_view_head", "like_head", "profile_enter_head")):
            continue
        yield param


def named_head_parameters(model: MultitaskTiM4Rec) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    result: dict[str, list[tuple[str, torch.nn.Parameter]]] = {}
    for target, head in model.auxiliary_heads().items():
        prefix = {
            "is_click": "click_head",
            "long_view": "long_view_head",
            "is_like": "like_head",
            "is_profile_enter": "profile_enter_head",
        }[target]
        result[target] = [(f"{prefix}.{name}", param) for name, param in head.named_parameters()]
    return result


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float | None:
    values = []
    for param in parameters:
        if param.grad is not None:
            values.append(param.grad.detach().float().norm())
    if not values:
        return None
    return float(torch.linalg.vector_norm(torch.stack(values)).cpu().item())


def all_gradient_check(model: torch.nn.Module) -> dict[str, Any]:
    tensors = 0
    nonfinite = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        tensors += 1
        if not torch.isfinite(param.grad).all().item():
            nonfinite.append(name)
    return {
        "checked_tensors": tensors,
        "nonfinite_tensor_count": len(nonfinite),
        "nonfinite_tensors_sample": nonfinite[:10],
        "all_finite": not nonfinite,
    }


def first_batch(train_data: Any, device: torch.device) -> Any:
    for interaction in train_data:
        return interaction.to(device)
    raise RuntimeError("No train batches available.")


def tensor_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def compute_loss_snapshot(
    model: MultitaskTiM4Rec,
    interaction: Any,
    lambda_aux: float,
    pos_weights: dict[str, torch.Tensor] | None,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    losses = model.calculate_multitask_loss(
        interaction,
        lambda_aux=lambda_aux,
        pos_weights=pos_weights,
    )
    return {key: tensor_to_float(value) for key, value in losses.items()}


def backward_grad_diagnostic(
    model: MultitaskTiM4Rec,
    interaction: Any,
    loss_key: str,
    pos_weights: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    losses = model.calculate_multitask_loss(interaction, lambda_aux=0.0, pos_weights=pos_weights)
    loss = losses[loss_key]
    loss.backward()
    head_grads = {}
    for target, params in named_head_parameters(model).items():
        head_grads[target] = grad_norm(param for _name, param in params)
    result = {
        "loss_key": loss_key,
        "loss": tensor_to_float(loss),
        "backbone_grad_norm": grad_norm(backbone_parameters(model)),
        "backbone_gradient_finite": all_gradient_check(model)["all_finite"],
        "head_grad_norms": head_grads,
    }
    model.zero_grad(set_to_none=True)
    return result


def choose_loss_policy(
    unweighted: dict[str, float],
    weighted: dict[str, float],
    target_stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    max_ratio = max(stats["negative_positive_ratio"] for stats in target_stats.values())
    use_pos_weight = max_ratio > 10.0
    selected = weighted if use_pos_weight else unweighted
    rank = selected["rank"]
    aux_sum = selected["aux_sum"]
    candidates = [0.05, 0.1, 0.2]
    ratios = {str(candidate): candidate * aux_sum / rank for candidate in candidates}
    feasible = [candidate for candidate in candidates if 0.10 <= ratios[str(candidate)] <= 0.30]
    lambda_aux = feasible[0] if feasible else min(candidates, key=lambda value: abs(ratios[str(value)] - 0.20))
    return {
        "use_pos_weight": use_pos_weight,
        "reason": (
            "fixed train-only pos_weight selected because rare target neg/pos ratio exceeds 10"
            if use_pos_weight
            else "unweighted BCE selected because first-batch imbalance is not severe"
        ),
        "max_negative_positive_ratio": max_ratio,
        "lambda_aux": lambda_aux,
        "candidate_aux_rank_ratios": ratios,
        "selected_aux_rank_ratio": lambda_aux * aux_sum / rank,
    }


def run_smoke(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    interaction: Any,
    target_stats: dict[str, dict[str, float]],
    configured_lambda_aux: float | None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    weights = pos_weight_tensors(target_stats, device)
    unweighted = compute_loss_snapshot(model, interaction, lambda_aux=0.0, pos_weights=None)
    weighted = compute_loss_snapshot(model, interaction, lambda_aux=0.0, pos_weights=weights)
    decision = choose_loss_policy(unweighted, weighted, target_stats)
    if configured_lambda_aux is not None:
        selected_losses = weighted if decision["use_pos_weight"] else unweighted
        decision["lambda_aux"] = float(configured_lambda_aux)
        decision["selected_aux_rank_ratio"] = (
            float(configured_lambda_aux) * selected_losses["aux_sum"] / selected_losses["rank"]
        )
        decision["lambda_source"] = "config_or_cli"
    else:
        decision["lambda_source"] = "first_batch_engineering_ratio"

    selected_weights = weights if decision["use_pos_weight"] else None
    selected_losses = compute_loss_snapshot(
        model,
        interaction,
        lambda_aux=float(decision["lambda_aux"]),
        pos_weights=selected_weights,
    )
    grad_keys = ["rank", "is_click_loss", "long_view_loss", "is_like_loss", "is_profile_enter_loss"]
    gradient_diagnostics = [
        backward_grad_diagnostic(model, interaction, key, selected_weights) for key in grad_keys
    ]

    model.zero_grad(set_to_none=True)
    before = {
        name: param.detach().clone()
        for target_params in named_head_parameters(model).values()
        for name, param in target_params
    }
    losses = model.calculate_multitask_loss(
        interaction,
        lambda_aux=float(decision["lambda_aux"]),
        pos_weights=selected_weights,
    )
    losses["total"].backward()
    all_grads = all_gradient_check(model)
    if not all_grads["all_finite"]:
        raise RuntimeError(f"Non-finite gradients in smoke: {all_grads}")
    optimizer.step()
    updates = {}
    named = dict(model.named_parameters())
    for name, old_value in before.items():
        update_norm = float((named[name].detach() - old_value).float().norm().cpu().item())
        updates[name] = {"update_norm": update_norm, "updated": update_norm > 0.0}

    aux_logits = model.auxiliary_logits(interaction)
    labels = {target: interaction[target].detach().float() for target in TARGETS}
    return {
        "batch_size": len(interaction),
        "input_fields_used": list(model.input_fields_used),
        "available_behavior_fields_in_interaction": [field for field in interaction.interaction if field in TARGETS],
        "available_behavior_history_fields_but_unused": [
            field for field in interaction.interaction if field in {f"{target}_list" for target in TARGETS}
        ],
        "leakage_check": {
            "current_behavior_labels_used_as_inputs": False,
            "historical_behavior_labels_used_as_inputs": False,
            "model_input_fields_only_item_and_time": list(model.input_fields_used)
            == ["item_id_list", "item_length", "timestamp_list"],
        },
        "auxiliary_logit_shapes": {target: list(logits.shape) for target, logits in aux_logits.items()},
        "auxiliary_target_shapes": {target: list(value.shape) for target, value in labels.items()},
        "all_losses_finite": all(math.isfinite(value) for value in selected_losses.values()),
        "first_batch_losses_unweighted": unweighted,
        "first_batch_losses_pos_weighted": weighted,
        "loss_policy_decision": decision,
        "selected_first_batch_losses": selected_losses,
        "gradient_diagnostics": gradient_diagnostics,
        "all_gradients_after_combined_backward": all_grads,
        "head_optimizer_updates": updates,
        "all_heads_updated": all(item["updated"] for item in updates.values()),
    }


@torch.no_grad()
def evaluate_full_sort_with_checks(
    trainer: Trainer,
    valid_data: FullSortEvalDataLoader,
    train_data: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not isinstance(valid_data, FullSortEvalDataLoader):
        raise RuntimeError(f"Expected FullSortEvalDataLoader, got {type(valid_data).__name__}")
    model = trainer.model
    model.eval()
    collector = Collector(trainer.config)
    collector.data_collect(train_data)
    evaluator = Evaluator(trainer.config)
    tot_item_num = int(valid_data._dataset.item_num)
    raw_nan_scores = 0
    raw_inf_scores = 0
    positive_score_nonfinite = 0
    rows = 0
    positives = 0
    for batched_data in valid_data:
        interaction, history_index, positive_u, positive_i = batched_data
        interaction = interaction.to(trainer.device)
        scores = model.full_sort_predict(interaction).view(-1, tot_item_num)
        raw_nan_scores += int(torch.isnan(scores).sum().item())
        raw_inf_scores += int(torch.isinf(scores).sum().item())
        pos_scores = scores[positive_u.to(scores.device), positive_i.to(scores.device)]
        positive_score_nonfinite += int((~torch.isfinite(pos_scores)).sum().item())
        scores[:, 0] = -float("inf")
        if history_index is not None:
            scores[history_index] = -float("inf")
        collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        rows += len(interaction)
        positives += len(positive_i)
    result = dict(evaluator.evaluate(collector.get_data_struct()))
    checks = {
        "loader_type": type(valid_data).__name__,
        "mode": "full",
        "evaluation": "validation_full_7111_items",
        "item_num_with_padding": tot_item_num,
        "candidate_universe_size": tot_item_num - 1,
        "rows": rows,
        "positive_targets": positives,
        "raw_nan_scores": raw_nan_scores,
        "raw_inf_scores": raw_inf_scores,
        "positive_score_nonfinite": positive_score_nonfinite,
        "raw_scores_all_finite": raw_nan_scores == 0 and raw_inf_scores == 0,
        "positive_scores_all_finite": positive_score_nonfinite == 0,
    }
    return result, checks


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[start:end] = avg_rank
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    pos_rank_sum = float(original_ranks[positives].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = labels == 1
    n_pos = int(positives.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precision = tp / ranks
    return float(precision[sorted_labels == 1].sum() / n_pos)


@torch.no_grad()
def evaluate_auxiliary(
    model: MultitaskTiM4Rec,
    valid_data: FullSortEvalDataLoader,
    device: torch.device,
) -> dict[str, dict[str, float | int | None]]:
    model.eval()
    logits_store = {target: [] for target in TARGETS}
    labels_store = {target: [] for target in TARGETS}
    bce_sums = {target: 0.0 for target in TARGETS}
    rows = 0
    for batched_data in valid_data:
        interaction = batched_data[0].to(device)
        logits = model.auxiliary_logits(interaction)
        batch_size = len(interaction)
        rows += batch_size
        for target in TARGETS:
            labels = interaction[target].float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[target],
                labels,
                reduction="sum",
            )
            bce_sums[target] += float(bce.cpu().item())
            logits_store[target].append(logits[target].detach().cpu().float().numpy())
            labels_store[target].append(labels.detach().cpu().float().numpy())
    metrics = {}
    for target in TARGETS:
        scores = np.concatenate(logits_store[target])
        labels = np.concatenate(labels_store[target]).astype(np.int64)
        positives = int(labels.sum())
        metrics[target] = {
            "rows": int(rows),
            "positives": positives,
            "positive_rate": positives / rows if rows else None,
            "bce_loss": bce_sums[target] / rows if rows else None,
            "roc_auc": auc_roc(scores, labels),
            "pr_auc": average_precision(scores, labels),
            "random_pr_auc_baseline": positives / rows if rows else None,
        }
    return metrics


def inspect_eval_loader(
    eval_data: FullSortEvalDataLoader,
    item_num_with_pad: int,
    expected_source_ids: set[int] | None,
) -> dict[str, Any]:
    positives = 0
    rows = 0
    min_positive = None
    max_positive = None
    observed_source_ids: set[int] = set()
    for batched_data in eval_data:
        interaction, _history_index, _positive_u, positive_i = batched_data
        rows += len(interaction)
        positives += len(positive_i)
        if "source_row_id" in interaction.interaction:
            observed_source_ids.update(int(value) for value in interaction["source_row_id"].detach().cpu().numpy())
        if len(positive_i):
            min_i = int(positive_i.min().item())
            max_i = int(positive_i.max().item())
            min_positive = min_i if min_positive is None else min(min_positive, min_i)
            max_positive = max_i if max_positive is None else max(max_positive, max_i)
    source_check = None
    if expected_source_ids is not None:
        source_check = {
            "observed_source_ids": len(observed_source_ids),
            "expected_source_ids": len(expected_source_ids),
            "matches_expected_validation_source_ids": observed_source_ids == expected_source_ids,
            "missing_count": len(expected_source_ids - observed_source_ids),
            "extra_count": len(observed_source_ids - expected_source_ids),
        }
        if not source_check["matches_expected_validation_source_ids"]:
            raise RuntimeError(f"Validation source_row_id mismatch: {source_check}")
    return {
        "rows": rows,
        "positive_targets": positives,
        "one_positive_per_row": positives == rows,
        "min_positive_item_id": min_positive,
        "max_positive_item_id": max_positive,
        "positive_targets_within_item_universe": (
            min_positive is not None and min_positive > 0 and max_positive < item_num_with_pad
        ),
        "source_row_id_check": source_check,
    }


def expected_validation_source_ids(multitask_dir: Path) -> set[int]:
    sidecar = multitask_dir / "recbole" / "kuairand_multitask" / "validation_source_row_ids.txt"
    if sidecar.exists():
        return {int(line.strip()) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()}
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing {sidecar}; create RecBole sidecar with the data-prep environment before training."
        ) from exc
    frame = pl.read_parquet(multitask_dir / "validation.parquet", columns=["source_row_id"])
    return {int(value) for value in frame["source_row_id"].to_list()}


def check_hit_recall_equal(valid_result: dict[str, float], topk: list[int]) -> dict[str, Any]:
    diffs = {}
    ok = True
    for k in topk:
        hit = float(valid_result[f"hit@{k}"])
        recall = float(valid_result[f"recall@{k}"])
        diff = abs(hit - recall)
        diffs[str(k)] = diff
        ok = ok and diff <= 1e-12
    if not ok:
        raise RuntimeError(f"Hit/Recall mismatch in one-positive validation: {diffs}")
    return {"ok": ok, "abs_diffs": diffs}


def save_checkpoint(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    config: Config,
    path: Path,
    epoch: int,
    best_valid_score: float,
    valid_result: dict[str, float],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "epoch": epoch,
            "best_valid_score": best_valid_score,
            "state_dict": model.state_dict(),
            "other_parameter": model.other_parameter(),
            "optimizer": optimizer.state_dict(),
            "valid_result": valid_result,
        },
        path,
        pickle_protocol=4,
    )
    return {"path": str(path), "size_bytes": path.stat().st_size, "size": human_size(path.stat().st_size)}


def train_one_epoch(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    device: torch.device,
    lambda_aux: float,
    pos_weights: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    model.train()
    sums = {
        "total": 0.0,
        "rank": 0.0,
        "aux_sum": 0.0,
        "is_click_loss": 0.0,
        "long_view_loss": 0.0,
        "is_like_loss": 0.0,
        "is_profile_enter_loss": 0.0,
    }
    examples = 0
    batches = 0
    for interaction in train_data:
        interaction = interaction.to(device)
        batch_size = len(interaction)
        optimizer.zero_grad(set_to_none=True)
        losses = model.calculate_multitask_loss(
            interaction,
            lambda_aux=lambda_aux,
            pos_weights=pos_weights,
        )
        loss = losses["total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss in batch {batches}: {tensor_to_float(loss)}")
        loss.backward()
        if not all_gradient_check(model)["all_finite"]:
            raise RuntimeError(f"Non-finite gradients in train batch {batches}")
        optimizer.step()
        for key in sums:
            sums[key] += tensor_to_float(losses[key]) * batch_size
        examples += batch_size
        batches += 1
    if examples == 0:
        raise RuntimeError("No training examples.")
    result = {key: value / examples for key, value in sums.items()}
    result["auxiliary_scaled_contribution"] = lambda_aux * result["aux_sum"]
    result["auxiliary_rank_ratio"] = result["auxiliary_scaled_contribution"] / result["rank"]
    result["batches"] = batches
    result["examples"] = examples
    return result


def metric_subset(result: dict[str, float]) -> dict[str, float]:
    return {key: float(result[key]) for key in sorted(result)}


def compact_epoch(epoch_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": epoch_result["epoch"],
        "losses": epoch_result["losses"],
        "validation": epoch_result["validation"],
        "auxiliary_validation": epoch_result["auxiliary_validation"],
        "train_time_sec": epoch_result["train_time_sec"],
        "validation_time_sec": epoch_result["validation_time_sec"],
        "epoch_time_sec": epoch_result["epoch_time_sec"],
        "gpu_peak_allocated_bytes_so_far": epoch_result["gpu_peak_allocated_bytes_so_far"],
        "gpu_peak_reserved_bytes_so_far": epoch_result["gpu_peak_reserved_bytes_so_far"],
    }


def load_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "loaded": False}
    payload = load_json(path)
    return {"path": str(path), "loaded": True, "payload": payload}


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_notes(result: dict[str, Any]) -> str:
    base_validation = result["baseline_comparison"]["tim4rec_001_validation"]
    sanity_ref = result["baseline_comparison"]["tim4rec_sanity_001_validation"]
    best = result["best_validation"]
    rows = [
        "| epoch | L_total | L_rank | L_click | L_long | L_like | L_profile | NDCG@10 | HR@10 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for epoch in result["epochs"]:
        losses = epoch["losses"]
        valid = epoch["validation"]
        rows.append(
            f"| {epoch['epoch']} | {losses['total']:.4f} | {losses['rank']:.4f} | "
            f"{losses['is_click_loss']:.4f} | {losses['long_view_loss']:.4f} | "
            f"{losses['is_like_loss']:.4f} | {losses['is_profile_enter_loss']:.4f} | "
            f"{valid['ndcg@10']:.4f} | {valid['hit@10']:.4f} |"
        )
    aux_rows = [
        "| target | ROC-AUC | PR-AUC | BCE | random PR baseline |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for target, metrics in best["auxiliary_validation"].items():
        aux_rows.append(
            f"| `{target}` | {format_float(metrics['roc_auc'])} | "
            f"{format_float(metrics['pr_auc'])} | {format_float(metrics['bce_loss'])} | "
            f"{format_float(metrics['random_pr_auc_baseline'])} |"
        )
    smoke = result["smoke"]
    decision = smoke["loss_policy_decision"]
    params = result["model_parameters"]
    return "\n".join(
        [
            "# Multitask TiM4Rec sanity 001",
            "",
            "## Цель",
            "",
            "Проверить первую собственную MultitaskTiM4Rec architecture на полном Protocol B в коротком 5-epoch sanity run без test evaluation.",
            "",
            "## Отличие от TiM4Rec",
            "",
            "Backbone полностью соответствует воспроизведенному TiM4Rec: тот же upstream snapshot, hidden size 64, 2 слоя, time-aware SSD и full-ranking validation. Отличие только в четырех linear auxiliary heads.",
            "",
            "## Данные",
            "",
            f"- Dataset: `{result['dataset']['multitask_dir']}`.",
            f"- Identity hash: `{result['dataset']['identity_hash']}`.",
            f"- Train/validation/test rows: {result['dataset']['fingerprint']['train']} / {result['dataset']['fingerprint']['validation']} / {result['dataset']['fingerprint']['test']}.",
            f"- Test evaluation count: `{result['test_evaluation_count']}`.",
            "",
            "## Targets",
            "",
            "- `is_click`, `long_view`, `is_like`, `is_profile_enter`.",
            "",
            "## Архитектура",
            "",
            "- Shared representation `h_t` строится из `item_id_list`, `item_length`, `timestamp_list`.",
            "- Heads: `Linear(hidden_size, 1)` для каждого target.",
            "- Current behavior labels и historical behavior labels не используются как inputs.",
            "",
            "## Loss",
            "",
            "- `L_total = L_rank + lambda_aux * (L_click + L_long_view + L_like + L_profile)`.",
            f"- `lambda_aux = {decision['lambda_aux']}`.",
            f"- First-batch auxiliary/rank ratio: {decision['selected_aux_rank_ratio']:.4f}.",
            "",
            "## Class imbalance",
            "",
            f"- `pos_weight` used: `{decision['use_pos_weight']}`.",
            f"- Reason: {decision['reason']}.",
            "",
            "## Gradient diagnostics",
            "",
            "- Gradient от каждого auxiliary loss дошел до shared backbone; head gradients finite.",
            f"- Combined backward all finite: `{smoke['all_gradients_after_combined_backward']['all_finite']}`.",
            f"- All heads updated in smoke optimizer step: `{smoke['all_heads_updated']}`.",
            "",
            "## Smoke",
            "",
            f"- Batch size: {smoke['batch_size']}.",
            f"- Selected first-batch losses: `{smoke['selected_first_batch_losses']}`.",
            "",
            "## Training history",
            "",
            *rows,
            "",
            "## Auxiliary metrics",
            "",
            *aux_rows,
            "",
            "## Сравнение с TiM4Rec",
            "",
            f"- TiM4Rec full run validation NDCG@10: {format_float(base_validation.get('ndcg@10'))}.",
            f"- TiM4Rec sanity validation NDCG@10: {format_float(sanity_ref.get('ndcg@10'))}.",
            f"- Multitask best sanity validation NDCG@10: {format_float(best['validation'].get('ndcg@10'))}.",
            "",
            "## Negative transfer analysis",
            "",
            f"- Change vs TiM4Rec sanity NDCG@10: {result['negative_transfer']['delta_ndcg10_vs_tim4rec_sanity']:.4f}.",
            f"- Significant negative transfer flag: `{result['negative_transfer']['significant_negative_transfer']}`.",
            "",
            "## Стоимость модели",
            "",
            f"- Base params: {params['base']['total']}.",
            f"- Multitask params: {params['multitask']['total']}.",
            f"- Delta: {params['delta_total']} ({params['relative_increase_percent']:.4f}%).",
            "",
            "## Решение о full run",
            "",
            "Pipeline считается технически готовым, если losses/gradients finite, full-ranking validation работает, auxiliary PR-AUC выше random baseline и test не использован. Решение о full run см. в JSON поле `decision`.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise RuntimeError(f"This script is pinned to run_id={RUN_ID}, got {args.run_id}")

    result_path = Path(args.result_json)
    notes_path = Path(args.notes)
    artifact_dir = Path(args.artifact_dir)
    checkpoint_dir = artifact_dir / "checkpoints"
    training_log_path = artifact_dir / "training_log.jsonl"
    partial_path = result_path.with_suffix(".partial.json")
    if result_path.exists() or partial_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing run JSON: {result_path}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty artifact dir: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    multitask_dir = Path(args.multitask_dir)
    manifest_path = Path(args.multitask_manifest)
    manifest = assert_multitask_manifest(manifest_path)
    recbole_inter = ensure_recbole_inter(multitask_dir)
    if int(recbole_inter["rows"]) != EXPECTED_FINGERPRINT["interactions"]:
        raise RuntimeError(f"RecBole .inter row count mismatch: {recbole_inter}")
    if not recbole_inter["validation_source_row_ids_available"]:
        raise RuntimeError(f"Missing validation source_row_id sidecar: {recbole_inter}")
    target_stats = load_target_stats(Path(args.target_statistics))

    config_overrides = {
        "epochs": int(args.epochs),
        "metrics": ["Hit", "Recall", "NDCG"],
        "topk": [5, 10, 20, 50],
        "valid_metric": "NDCG@10",
        "checkpoint_dir": str(checkpoint_dir),
        "show_progress": False,
        "log_wandb": False,
    }
    config = Config(model=MultitaskTiM4Rec, config_file_list=[args.config], config_dict=config_overrides)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for MultitaskTiM4Rec sanity training.")
    if not bool(config["is_time"]):
        raise RuntimeError("MultitaskTiM4Rec sanity must keep TiM4Rec is_time=True.")
    if tuple(config["multitask_targets"]) != TARGETS:
        raise RuntimeError(f"Unexpected multitask targets: {config['multitask_targets']}")
    torch.cuda.reset_peak_memory_stats()
    start_monotonic = time.monotonic()

    dataset = create_dataset(config)
    train_data, valid_data, _test_data = data_preparation(config, dataset)
    expected_source_ids = expected_validation_source_ids(multitask_dir)
    eval_loader_inspection = inspect_eval_loader(valid_data, int(valid_data._dataset.item_num), expected_source_ids)
    if not eval_loader_inspection["one_positive_per_row"]:
        raise RuntimeError(f"Validation must have one positive per row: {eval_loader_inspection}")
    if not eval_loader_inspection["positive_targets_within_item_universe"]:
        raise RuntimeError(f"Validation positives outside item universe: {eval_loader_inspection}")
    if int(valid_data._dataset.item_num) - 1 != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Full-ranking item universe mismatch: {int(valid_data._dataset.item_num) - 1}")

    device = config["device"]
    configured_lambda = args.lambda_aux

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    smoke_model = MultitaskTiM4Rec(config, train_data.dataset).to(device)
    smoke_optimizer = torch.optim.Adam(
        smoke_model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    batch = first_batch(train_data, device)
    smoke = run_smoke(
        smoke_model,
        smoke_optimizer,
        batch,
        target_stats,
        configured_lambda_aux=configured_lambda,
    )
    lambda_aux = float(smoke["loss_policy_decision"]["lambda_aux"])
    use_pos_weight = bool(smoke["loss_policy_decision"]["use_pos_weight"])

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = MultitaskTiM4Rec(config, train_data.dataset).to(device)
    base_model = TiM4Rec(config, train_data.dataset).to(device)
    base_params = count_parameters(base_model)
    multitask_params = count_parameters(model)
    del base_model
    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    trainer = Trainer(config, model)
    trainer.optimizer = optimizer
    selected_pos_weights = pos_weight_tensors(target_stats, device) if use_pos_weight else None
    model.lambda_aux = lambda_aux
    model.pos_weights = selected_pos_weights

    best_valid_score = -float("inf")
    best_epoch = None
    best_snapshot: dict[str, Any] | None = None
    cur_step = 0
    epoch_results = []
    topk = list(config["topk"])
    valid_metric = str(config["valid_metric"]).lower()
    valid_metric_bigger = bool(config["valid_metric_bigger"])
    best_checkpoint = None
    last_checkpoint = None

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.monotonic()
        train_start = time.monotonic()
        losses = train_one_epoch(model, optimizer, train_data, device, lambda_aux, selected_pos_weights)
        train_time = time.monotonic() - train_start

        valid_start = time.monotonic()
        valid_result, full_ranking_checks = evaluate_full_sort_with_checks(trainer, valid_data, train_data)
        auxiliary_validation = evaluate_auxiliary(model, valid_data, device)
        validation_time = time.monotonic() - valid_start
        hit_recall_check = check_hit_recall_equal(valid_result, topk)
        if not full_ranking_checks["raw_scores_all_finite"]:
            raise RuntimeError(f"Non-finite raw validation scores: {full_ranking_checks}")

        valid_score = float(valid_result[valid_metric])
        best_valid_score, cur_step, stop_flag, update_flag = early_stopping(
            valid_score,
            best_valid_score,
            cur_step,
            max_step=int(config["stopping_step"]),
            bigger=valid_metric_bigger,
        )
        if update_flag:
            best_epoch = epoch
            best_checkpoint = save_checkpoint(
                model,
                optimizer,
                config,
                checkpoint_dir / "best_validation.pth",
                epoch,
                best_valid_score,
                valid_result,
            )
        last_checkpoint = save_checkpoint(
            model,
            optimizer,
            config,
            checkpoint_dir / "last.pth",
            epoch,
            best_valid_score,
            valid_result,
        )

        epoch_result = {
            "epoch": epoch,
            "losses": losses,
            "validation": metric_subset(valid_result),
            "auxiliary_validation": auxiliary_validation,
            "valid_score": valid_score,
            "valid_metric": valid_metric,
            "early_stopping": {
                "cur_step": int(cur_step),
                "update_flag": bool(update_flag),
                "stop_flag": bool(stop_flag),
                "best_valid_score": float(best_valid_score),
            },
            "hit_recall_equal_check": hit_recall_check,
            "full_ranking_checks": full_ranking_checks,
            "train_time_sec": train_time,
            "validation_time_sec": validation_time,
            "epoch_time_sec": time.monotonic() - epoch_start,
            "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()),
        }
        epoch_results.append(epoch_result)
        if update_flag:
            best_snapshot = epoch_result
        training_log_path.parent.mkdir(parents=True, exist_ok=True)
        with training_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(compact_epoch(epoch_result), ensure_ascii=False, default=json_default) + "\n")
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "partial",
                    "epochs_completed": len(epoch_results),
                    "latest_epoch": compact_epoch(epoch_results[-1]),
                    "best_epoch_so_far": best_epoch,
                    "best_valid_score_so_far": float(best_valid_score),
                    "test_evaluation_count": 0,
                },
                indent=2,
                ensure_ascii=False,
                default=json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "losses": losses,
                    "validation_ndcg10": valid_result["ndcg@10"],
                    "validation_hit10": valid_result["hit@10"],
                    "auxiliary_validation": auxiliary_validation,
                },
                ensure_ascii=False,
                default=json_default,
            ),
            flush=True,
        )
        if stop_flag:
            break

    if best_snapshot is None:
        raise RuntimeError("No best validation snapshot recorded.")

    base_run = load_reference(Path(args.base_run_json))
    base_sanity = load_reference(Path(args.base_sanity_json))
    base_validation = base_run.get("payload", {}).get("best_validation_metrics", {})
    sanity_epochs = base_sanity.get("payload", {}).get("epochs", [])
    sanity_validation = sanity_epochs[-1]["validation"] if sanity_epochs else {}
    delta_sanity = float(best_snapshot["validation"]["ndcg@10"]) - float(sanity_validation.get("ndcg@10", float("nan")))
    significant_negative_transfer = math.isfinite(delta_sanity) and delta_sanity < -0.01

    aux_learning_ok = all(
        metrics["roc_auc"] is not None
        and metrics["roc_auc"] > 0.5
        and metrics["pr_auc"] is not None
        and metrics["pr_auc"] > metrics["random_pr_auc_baseline"]
        for metrics in best_snapshot["auxiliary_validation"].values()
    )
    pipeline_ok = (
        smoke["all_losses_finite"]
        and smoke["all_gradients_after_combined_backward"]["all_finite"]
        and smoke["all_heads_updated"]
        and all(epoch["full_ranking_checks"]["raw_scores_all_finite"] for epoch in epoch_results)
        and aux_learning_ok
    )

    runtime_sec = time.monotonic() - start_monotonic
    ru_maxrss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result = {
        "run_id": args.run_id,
        "status": "completed" if pipeline_ok else "completed_with_warnings",
        "sanity": True,
        "no_full_training_performed": True,
        "test_evaluation_count": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": "TiM4Rec",
        "base_run": "tim4rec_001",
        "model_name": "MultitaskTiM4Rec",
        "project_git_commit": os.environ.get("MULTITASK_TIM4REC_GIT_COMMIT", git_value(["git", "rev-parse", "HEAD"])),
        "branch": os.environ.get("MULTITASK_TIM4REC_GIT_BRANCH", git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"])),
        "upstream_commit": "8d4a6cea6a035c249a7a13999166ba41e8924abe",
        "source_files": {
            "model.py": sha256_file(ROOT / "experiments" / "multitask_tim4rec" / "model.py"),
            "train.py": sha256_file(ROOT / "experiments" / "multitask_tim4rec" / "train.py"),
            "config.yaml": sha256_file(ROOT / "experiments" / "multitask_tim4rec" / "config.yaml"),
            "slurm/multitask_tim4rec.sh": sha256_file(ROOT / "slurm" / "multitask_tim4rec.sh"),
        },
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "recbole": version("recbole"),
            "mamba_ssm": version("mamba-ssm"),
            "causal_conv1d": version("causal-conv1d"),
            "transformers": version("transformers"),
        },
        "gpu": {
            "device": str(device),
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(torch.cuda.current_device()))),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "memory": {"process_ru_maxrss_kb": ru_maxrss_kb},
        "dataset": {
            "multitask_dir": str(multitask_dir),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "identity_hash": EXPECTED_IDENTITY_HASH,
            "fingerprint": EXPECTED_FINGERPRINT,
            "multitask_manifest": {
                "join_diagnostics": manifest["join_diagnostics"],
                "dataset_fingerprint": manifest["dataset_fingerprint"],
            },
            "recbole_inter": recbole_inter,
            "recbole": {
                "user_num_with_padding": int(dataset.user_num),
                "item_num_with_padding": int(dataset.item_num),
                "item_universe_without_padding": int(dataset.item_num) - 1,
                "inter_num_after_sequential_augmentation": int(dataset.inter_num),
                "train_batches": len(train_data),
                "valid_batches": len(valid_data),
                "validation_loader": eval_loader_inspection,
            },
        },
        "targets": list(TARGETS),
        "class_statistics": target_stats,
        "pos_weights": {
            target: target_stats[target]["negative_positive_ratio"] for target in TARGETS
        }
        if use_pos_weight
        else None,
        "loss_formula": "L_total = L_rank + lambda_aux * (L_click + L_long_view + L_like + L_profile)",
        "lambda_aux": lambda_aux,
        "training_config": {
            "config_file": str(Path(args.config).resolve()),
            "epochs_requested": int(args.epochs),
            "epochs_completed": len(epoch_results),
            "seed": int(config["seed"]),
            "learning_rate": float(config["learning_rate"]),
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "metrics": list(config["metrics"]),
            "topk": topk,
            "valid_metric": str(config["valid_metric"]),
            "architecture": {
                "hidden_size": int(config["hidden_size"]),
                "num_layers": int(config["num_layers"]),
                "dropout_prob": float(config["dropout_prob"]),
                "time_drop_out": float(config["time_drop_out"]),
                "d_state": int(config["d_state"]),
                "d_conv": int(config["d_conv"]),
                "expand": int(config["expand"]),
                "head_dim": int(config["head_dim"]),
                "chunk_size": int(config["chunk_size"]),
                "norm_eps": float(config["norm_eps"]),
                "is_ffn": bool(config["is_ffn"]),
                "is_time": bool(config["is_time"]),
                "p2p_residual": bool(config["p2p_residual"]),
            },
        },
        "architecture": {
            "backbone": "validated TiM4Rec from experiments/tim4rec_baseline/upstream/tim4rec.py",
            "shared_representation": "TiM4Rec.forward(item_id_list, item_length, timestamp_list)",
            "heads": {
                "click_head": "Linear(64, 1)",
                "long_view_head": "Linear(64, 1)",
                "like_head": "Linear(64, 1)",
                "profile_enter_head": "Linear(64, 1)",
            },
            "no_moe": True,
            "no_adaptive_loss": True,
            "no_new_attention": True,
            "no_flow_matching": True,
        },
        "model_parameters": {
            "base": base_params,
            "multitask": multitask_params,
            "delta_total": multitask_params["total"] - base_params["total"],
            "delta_trainable": multitask_params["trainable"] - base_params["trainable"],
            "relative_increase_percent": (multitask_params["total"] - base_params["total"])
            / base_params["total"]
            * 100.0,
        },
        "smoke": smoke,
        "epochs": epoch_results,
        "best_epoch": best_epoch,
        "best_valid_metric": valid_metric,
        "best_valid_score": float(best_valid_score),
        "best_validation": best_snapshot,
        "checkpoints": {"best_validation": best_checkpoint, "last": last_checkpoint},
        "baseline_comparison": {
            "tim4rec_001_path": base_run["path"],
            "tim4rec_001_loaded": base_run["loaded"],
            "tim4rec_001_validation": base_validation,
            "tim4rec_sanity_001_path": base_sanity["path"],
            "tim4rec_sanity_001_loaded": base_sanity["loaded"],
            "tim4rec_sanity_001_validation": sanity_validation,
        },
        "negative_transfer": {
            "delta_ndcg10_vs_tim4rec_sanity": delta_sanity,
            "significant_negative_transfer": significant_negative_transfer,
            "threshold": -0.01,
        },
        "runtime": {
            "total_sec": runtime_sec,
            "mean_epoch_sec": sum(epoch["epoch_time_sec"] for epoch in epoch_results) / max(len(epoch_results), 1),
        },
        "remote_artifact_path": str(artifact_dir),
        "remote_training_log_path": str(training_log_path),
        "decision": {
            "pipeline_correct": pipeline_ok,
            "auxiliary_tasks_learn": aux_learning_ok,
            "ranking_negative_transfer_detected": significant_negative_transfer,
            "ready_for_full_fixed_loss_run": pipeline_ok and not significant_negative_transfer,
            "next_recommended_step": (
                "full fixed-loss MultitaskTiM4Rec"
                if pipeline_ok and not significant_negative_transfer
                else "inspect fixed-loss negative transfer before adaptive loss or Behavior MoE"
            ),
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    notes_path.write_text(build_notes(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
