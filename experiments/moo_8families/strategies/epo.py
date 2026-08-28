"""Exact Pareto Optimization LP helper.

The official EPO implementation solves a small LP. To keep this benchmark
portable on the cluster environment, this module solves the same low-dimensional
LP by enumerating vertices of the feasible polytope.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .base import TASK_ORDER, preference_tensor, require_torch, tensor_to_float


@dataclass
class LPResult:
    alpha: list[float]
    objective: float
    feasible: bool
    active_constraints: list[int]
    backend: str


def _solve_lp_vertices(
    objective: np.ndarray,
    inequality_matrix: np.ndarray,
    inequality_rhs: np.ndarray,
    *,
    tol: float = 1e-8,
) -> LPResult:
    m = int(objective.shape[0])
    nonneg_a = -np.eye(m)
    nonneg_b = np.zeros(m)
    a_ub = np.vstack([nonneg_a, -inequality_matrix])
    b_ub = np.concatenate([nonneg_b, -inequality_rhs])
    candidates: list[tuple[float, np.ndarray, list[int]]] = []
    active_indices = list(range(a_ub.shape[0]))

    for selected in itertools.combinations(active_indices, max(m - 1, 0)):
        a_eq = [np.ones(m)]
        b_eq = [1.0]
        for idx in selected:
            a_eq.append(a_ub[idx])
            b_eq.append(b_ub[idx])
        a_eq_np = np.asarray(a_eq, dtype=float)
        b_eq_np = np.asarray(b_eq, dtype=float)
        try:
            alpha = np.linalg.solve(a_eq_np, b_eq_np)
        except np.linalg.LinAlgError:
            alpha, *_ = np.linalg.lstsq(a_eq_np, b_eq_np, rcond=None)
            if np.linalg.norm(a_eq_np @ alpha - b_eq_np, ord=np.inf) > 1e-7:
                continue
        if abs(alpha.sum() - 1.0) > 1e-6:
            continue
        if np.any(alpha < -tol):
            continue
        if np.any(inequality_matrix @ alpha < inequality_rhs - 1e-6):
            continue
        alpha = np.maximum(alpha, 0.0)
        alpha = alpha / alpha.sum()
        candidates.append((float(objective @ alpha), alpha, list(selected)))

    for idx in range(m):
        alpha = np.zeros(m)
        alpha[idx] = 1.0
        if np.all(inequality_matrix @ alpha >= inequality_rhs - 1e-6):
            candidates.append((float(objective @ alpha), alpha, [idx]))

    if not candidates:
        alpha = np.ones(m) / m
        return LPResult(alpha=alpha.tolist(), objective=float(objective @ alpha), feasible=False, active_constraints=[], backend="vertex_enum")

    objective_value, alpha, active = max(candidates, key=lambda item: item[0])
    return LPResult(
        alpha=alpha.tolist(),
        objective=float(objective_value),
        feasible=True,
        active_constraints=active,
        backend="vertex_enum",
    )


class ExactParetoPreferenceSolver:
    """EPO LP solver for a single preference vector."""

    def __init__(
        self,
        preference: Sequence[float],
        *,
        task_order: Sequence[str] = TASK_ORDER,
        eps: float = 1e-4,
        alpha_multiplier: float | None = None,
    ):
        self.task_order = tuple(task_order)
        self.preference_source = [float(value) for value in preference]
        self.eps = float(eps)
        self.alpha_multiplier = len(self.task_order) if alpha_multiplier is None else float(alpha_multiplier)
        self.last_result: dict[str, Any] | None = None

    def _adjustments(self, losses: np.ndarray, preference: np.ndarray) -> tuple[np.ndarray, float, float]:
        m = losses.shape[0]
        weighted = preference * losses
        weighted_sum = max(float(weighted.sum()), 1e-12)
        l_hat = weighted / weighted_sum
        mu = float(np.sum(l_hat * np.log(np.clip(l_hat * m, 1e-12, None))))
        adjustment = preference * (np.log(np.clip(l_hat * m, 1e-12, None)) - mu)
        mu_rl = float(weighted_sum * mu)
        return adjustment, mu_rl, mu

    def alpha(self, losses: Any, gradients: Any) -> Any:
        th = require_torch()
        loss_np = losses.detach().float().cpu().numpy()
        if gradients.ndim != 2:
            raise ValueError(f"Expected gradients [task, dim], got {tuple(gradients.shape)}")
        grad_np = gradients.detach().float().cpu().numpy()
        pref = preference_tensor(self.preference_source, device=losses.device, dtype=losses.dtype)
        pref_np = pref.detach().float().cpu().numpy()
        c_matrix = grad_np @ grad_np.T
        adjustment, mu_rl, mu = self._adjustments(loss_np, pref_np)
        ca = c_matrix @ adjustment
        m = len(loss_np)

        if mu_rl > self.eps:
            constraints = c_matrix
            rhs = ca
            objective = ca
            lp_type = "balance"
        else:
            max_ca = float(np.max(ca))
            rhs_dom = min(max_ca, 0.0)
            constraints = np.vstack([c_matrix, ca.reshape(1, -1)])
            rhs = np.concatenate([np.zeros(m), np.asarray([rhs_dom])])
            objective = c_matrix.sum(axis=1)
            lp_type = "dominance"

        result = _solve_lp_vertices(objective, constraints, rhs)
        alpha = th.tensor(result.alpha, dtype=losses.dtype, device=losses.device)
        alpha = alpha * self.alpha_multiplier
        self.last_result = {
            "type": lp_type,
            "alpha": [tensor_to_float(value) for value in alpha],
            "alpha_simplex": result.alpha,
            "lp_objective": result.objective,
            "lp_feasible": result.feasible,
            "lp_backend": result.backend,
            "active_constraints": result.active_constraints,
            "mu_rl": mu_rl,
            "mu": mu,
        }
        return alpha

    def scalarize(self, losses: Any, gradients: Any) -> Any:
        alpha = self.alpha(losses, gradients)
        return (alpha.detach() * losses).sum()

    def state_dict(self) -> dict[str, Any]:
        return {
            "preference": self.preference_source,
            "eps": self.eps,
            "alpha_multiplier": self.alpha_multiplier,
            "last_result": self.last_result,
        }

