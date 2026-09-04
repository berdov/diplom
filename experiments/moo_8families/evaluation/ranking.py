"""Ranking evaluation wrappers for validation-only full-sort metrics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multitask_tim4rec.train import (  # noqa: E402
    check_hit_recall_equal,
    evaluate_full_sort_with_checks,
    metric_subset,
)
from experiments.multitask_tim4rec_optuna.optuna_search import normalize_metrics  # noqa: E402


def set_model_preference(model: Any, preference: Sequence[float] | None) -> None:
    if preference is not None and hasattr(model, "set_preference"):
        model.set_preference(preference)


def evaluate_validation_ranking(
    *,
    trainer: Any,
    model: Any,
    valid_data: Any,
    train_data: Any,
    topk: Sequence[int],
    preference: Sequence[float] | None = None,
) -> dict[str, Any]:
    set_model_preference(model, preference)
    valid_result, checks = evaluate_full_sort_with_checks(trainer, valid_data, train_data)
    check_hit_recall_equal(valid_result, list(topk))
    if not checks["raw_scores_all_finite"] or not checks["positive_scores_all_finite"]:
        raise RuntimeError(f"Non-finite validation scores: {checks}")
    return {
        "metrics": normalize_metrics(metric_subset(valid_result)),
        "checks": checks,
    }

