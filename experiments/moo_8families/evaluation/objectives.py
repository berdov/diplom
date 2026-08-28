"""Objective and diagnostic helpers for MOO runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adaptive_multitask_tim4rec.methods.common import (  # noqa: E402
    conflict_summary,
    cosine_matrix,
    gradient_norms,
    shared_parameter_entries,
    task_gradient_vectors,
    tensor_to_float,
)
from experiments.moo_8families.strategies.base import TASK_ORDER, losses_to_vector, normalize_loss_vector, require_torch  # noqa: E402
from experiments.multitask_tim4rec_optuna.optuna_search import compute_tuned_losses  # noqa: E402


def task_losses(
    model: Any,
    interaction: Any,
    sampled: Mapping[str, Any],
    pos_weights: Mapping[str, Any],
    *,
    loss_scales: Sequence[float] | Any | None = None,
) -> dict[str, Any]:
    losses = compute_tuned_losses(model, interaction, dict(sampled), dict(pos_weights))
    vector = losses_to_vector(losses)
    normalized = normalize_loss_vector(vector, loss_scales)
    losses["task_vector"] = vector
    losses["normalized_task_vector"] = normalized
    return losses


def scalar_loss_record(losses: Mapping[str, Any]) -> dict[str, float]:
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
    return {key: tensor_to_float(losses[key]) for key in keys if key in losses}


def gradient_diagnostics(
    model: Any,
    losses: Mapping[str, Any],
    *,
    selector: str = "all_backbone",
) -> dict[str, Any]:
    th = require_torch()
    entries = shared_parameter_entries(model, selector)
    task_map = {
        "rank": losses["rank"],
        "is_click": losses["is_click_loss"],
        "long_view": losses["long_view_loss"],
        "is_like": losses["is_like_loss"],
        "is_profile_enter": losses["is_profile_enter_loss"],
    }
    vectors = task_gradient_vectors(task_map, entries, TASK_ORDER)
    matrix = cosine_matrix(vectors, TASK_ORDER)
    norms = gradient_norms(vectors, TASK_ORDER)
    conflicts = conflict_summary(matrix, TASK_ORDER)
    model.zero_grad(set_to_none=True)
    return {
        "selector": selector,
        "cosine_matrix": matrix,
        "gradient_norms": norms,
        "conflicts": conflicts,
        "task_losses": {task: tensor_to_float(task_map[task]) for task in TASK_ORDER},
        "all_finite_vectors": bool(th.isfinite(th.stack(list(vectors.values()))).all()) if vectors else True,
    }


def batch_positive_rates(interaction: Any) -> dict[str, float]:
    return {
        target: tensor_to_float(interaction[target].float().mean())
        for target in TASK_ORDER
        if target != "rank" and target in interaction
    }

