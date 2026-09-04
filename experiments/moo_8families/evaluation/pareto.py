"""Pareto-set metrics for MOO benchmark reports."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


EVAL_OBJECTIVE_ORDER: tuple[str, ...] = (
    "1_minus_NDCG@10",
    "click_BCE",
    "long_view_BCE",
    "like_BCE",
    "profile_BCE",
)
RANKING_OPERATING_POINT_ID = "rank_heavy"
PREDEFINED_RANKING_METHODS = {"epo", "phn", "cosmos", "palora"}
PREFERENCE_FREE_FINITE_METHODS = {"gradhv"}


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


def objective_points_from_records(records: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    return [objective_point_from_record(dict(record)) for record in records]


def assert_reference_is_worse_than_points(
    points: Sequence[Sequence[float]],
    reference_point: Sequence[float],
    *,
    objective_order: Sequence[str] = EVAL_OBJECTIVE_ORDER,
    tol: float = 1e-12,
) -> dict[str, Any]:
    array = np.asarray(points, dtype=float)
    reference = np.asarray(reference_point, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected validation objective points [solutions, objectives], got {array.shape}")
    if reference.shape != (array.shape[1],):
        raise ValueError(f"Reference point shape {reference.shape} does not match objective count {array.shape[1]}")
    if len(tuple(objective_order)) != array.shape[1]:
        raise ValueError(f"Objective order length {len(tuple(objective_order))} does not match {array.shape[1]} objectives")
    if not np.isfinite(array).all():
        raise ValueError(f"Validation objective point contains non-finite values: {array}")
    if not np.isfinite(reference).all():
        raise ValueError(f"Validation reference point contains non-finite values: {reference}")
    violations = array > reference + float(tol)
    if bool(violations.any()):
        solution_idx, objective_idx = np.argwhere(violations)[0]
        objective = tuple(objective_order)[int(objective_idx)]
        value = float(array[int(solution_idx), int(objective_idx)])
        ref = float(reference[int(objective_idx)])
        raise ValueError(
            f"Validation point {solution_idx} is worse than frozen evaluation reference on {objective}: "
            f"point={value}, reference={ref}"
        )
    margins = reference.reshape(1, -1) - array
    return {
        "status": "valid",
        "objective_order": list(objective_order),
        "reference_point": reference.astype(float).tolist(),
        "min_margin_to_reference": float(margins.min()) if margins.size else None,
        "max_margin_to_reference": float(margins.max()) if margins.size else None,
        "solution_count": int(array.shape[0]),
    }


def _record_ndcg10(record: Mapping[str, Any]) -> float:
    return float((record.get("metrics") or {}).get("NDCG@10"))


def _same_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("preference_id") == right.get("preference_id")
        and left.get("solution_index") == right.get("solution_index")
        and _record_ndcg10(left) == _record_ndcg10(right)
    )


def select_validation_points(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    ranking_preference_id: str = RANKING_OPERATING_POINT_ID,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot select validation operating point from empty records.")
    copied = [dict(record) for record in records]
    oracle = max(copied, key=_record_ndcg10)
    method_key = str(method)

    if method_key in PREFERENCE_FREE_FINITE_METHODS:
        ranking = oracle
        selection_rule = "best_validation_NDCG@10_among_preference_free_finite_solutions"
        selection_is_validation_oracle = True
    elif method_key in PREDEFINED_RANKING_METHODS:
        matches = [record for record in copied if record.get("preference_id") == ranking_preference_id]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one predefined ranking operating point "
                f"{ranking_preference_id!r} for {method_key}, got {len(matches)}"
            )
        ranking = matches[0]
        selection_rule = f"predefined_preference_id:{ranking_preference_id}"
        selection_is_validation_oracle = False
    elif len(copied) == 1:
        ranking = copied[0]
        selection_rule = "single_solution"
        selection_is_validation_oracle = False
    else:
        ranking = copied[0]
        selection_rule = "single_or_first_solution"
        selection_is_validation_oracle = False

    ranking = dict(ranking)
    oracle = dict(oracle)
    ranking["selection_rule"] = selection_rule
    ranking["selection_is_validation_oracle"] = selection_is_validation_oracle
    oracle["selection_rule"] = "max_validation_NDCG@10"
    oracle["selection_is_validation_oracle"] = True
    return {
        "ranking_operating_point": ranking,
        "oracle_best_validation_point": oracle,
        "ranking_operating_point_selection": selection_rule,
        "selection_is_validation_oracle": selection_is_validation_oracle,
        "oracle_best_differs_from_ranking_operating_point": not _same_record(ranking, oracle),
    }


def validation_summary_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    reference_point: Sequence[float],
    ranking_preference_id: str = RANKING_OPERATING_POINT_ID,
    objective_order: Sequence[str] = EVAL_OBJECTIVE_ORDER,
) -> dict[str, Any]:
    copied = [dict(record) for record in records]
    objective_points = objective_points_from_records(copied)
    for record, point in zip(copied, objective_points):
        record["objective_point"] = point
        record["objective_order"] = list(objective_order)
    reference_check = assert_reference_is_worse_than_points(
        objective_points,
        reference_point,
        objective_order=objective_order,
    )
    selection = select_validation_points(
        copied,
        method=method,
        ranking_preference_id=ranking_preference_id,
    )
    return {
        "records": copied,
        "ranking_operating_point": selection["ranking_operating_point"],
        "oracle_best_validation_point": selection["oracle_best_validation_point"],
        "best": selection["ranking_operating_point"],
        "ranking_operating_point_selection": selection["ranking_operating_point_selection"],
        "selection_is_validation_oracle": selection["selection_is_validation_oracle"],
        "oracle_best_differs_from_ranking_operating_point": selection[
            "oracle_best_differs_from_ranking_operating_point"
        ],
        "pareto_validation": pareto_summary(objective_points, reference_point),
        "reference_check": reference_check,
    }
