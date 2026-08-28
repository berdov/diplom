"""Pareto-set metrics for MOO benchmark reports."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def non_dominated_mask(points: Sequence[Sequence[float]]) -> list[bool]:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected [solutions, objectives], got {array.shape}")
    mask = []
    for i, point in enumerate(array):
        dominated = False
        for j, other in enumerate(array):
            if i == j:
                continue
            if np.all(other <= point) and np.any(other < point):
                dominated = True
                break
        mask.append(not dominated)
    return mask


def spread(points: Sequence[Sequence[float]]) -> float | None:
    array = np.asarray(points, dtype=float)
    if len(array) <= 1:
        return None
    distances = []
    for i in range(len(array)):
        for j in range(i + 1, len(array)):
            distances.append(float(np.linalg.norm(array[i] - array[j], ord=2)))
    return float(np.mean(distances)) if distances else None


def exact_hypervolume_numpy(points: Sequence[Sequence[float]], reference_point: Sequence[float]) -> float:
    import itertools

    array = np.asarray(points, dtype=float)
    reference = np.asarray(reference_point, dtype=float)
    hv = 0.0
    for size in range(1, len(array) + 1):
        sign = 1.0 if size % 2 == 1 else -1.0
        for subset in itertools.combinations(range(len(array)), size):
            lower = np.max(array[list(subset)], axis=0)
            side = np.maximum(reference - lower, 0.0)
            hv += sign * float(np.prod(side))
    return float(hv)


def pareto_summary(points: Sequence[Sequence[float]], reference_point: Sequence[float]) -> dict[str, Any]:
    mask = non_dominated_mask(points)
    nondominated = [point for point, keep in zip(points, mask) if keep]
    return {
        "solution_count": len(points),
        "non_dominated_count": int(sum(mask)),
        "non_dominated_mask": mask,
        "spread_l2_mean": spread(nondominated),
        "hypervolume": exact_hypervolume_numpy(nondominated, reference_point) if nondominated else 0.0,
        "reference_point": [float(value) for value in reference_point],
    }


def objective_point_from_record(record: dict[str, Any]) -> list[float]:
    aux = record.get("auxiliary_validation") or record.get("best_auxiliary_metrics") or {}
    metrics = record.get("metrics") or record.get("best_validation_metrics") or {}
    ndcg = metrics.get("NDCG@10")
    rank_objective = 1.0 - float(ndcg) if ndcg is not None else float("nan")
    result = [rank_objective]
    for target in ("is_click", "long_view", "is_like", "is_profile_enter"):
        target_metrics = aux.get(target) or {}
        if "bce_loss" in target_metrics:
            result.append(float(target_metrics["bce_loss"]))
        elif "BCE" in target_metrics:
            result.append(float(target_metrics["BCE"]))
        else:
            result.append(float("nan"))
    return result
