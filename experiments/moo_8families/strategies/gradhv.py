"""Hypervolume objective for GradHV-style finite Pareto sets."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Sequence

from .base import TASK_ORDER, require_torch, tensor_to_float


@dataclass
class HypervolumeRecord:
    hypervolume: float
    reference_point: list[float]
    solution_count: int
    objective_count: int


class DominatedHypervolume:
    """Exact dominated hypervolume for a small set of minimization points."""

    def __init__(
        self,
        reference_point: Sequence[float],
        *,
        task_order: Sequence[str] = TASK_ORDER,
        eps: float = 1e-12,
    ):
        self.task_order = tuple(task_order)
        if self.task_order != TASK_ORDER:
            raise ValueError(f"Expected task order {TASK_ORDER}, got {self.task_order}")
        self.reference_point_source = [float(value) for value in reference_point]
        self.eps = float(eps)
        self.last_record: HypervolumeRecord | None = None

    def reference(self, points: Any) -> Any:
        th = require_torch()
        return th.tensor(self.reference_point_source, dtype=points.dtype, device=points.device)

    def value(self, points: Any) -> Any:
        th = require_torch()
        if points.ndim != 2:
            raise ValueError(f"Expected points [solutions, objectives], got {tuple(points.shape)}")
        solutions, objectives = points.shape
        reference = self.reference(points)
        hv = th.zeros((), dtype=points.dtype, device=points.device)
        for size in range(1, solutions + 1):
            sign = 1.0 if size % 2 == 1 else -1.0
            for subset in itertools.combinations(range(solutions), size):
                lower = th.stack([points[idx] for idx in subset], dim=0).max(dim=0).values
                side = (reference - lower).clamp_min(0.0)
                hv = hv + sign * side.prod()
        self.last_record = HypervolumeRecord(
            hypervolume=tensor_to_float(hv),
            reference_point=list(self.reference_point_source),
            solution_count=int(solutions),
            objective_count=int(objectives),
        )
        return hv

    def loss(self, points: Any) -> Any:
        return -self.value(points)

    def state_dict(self) -> dict[str, Any]:
        return {
            "reference_point": self.reference_point_source,
            "task_order": self.task_order,
            "last_record": None if self.last_record is None else self.last_record.__dict__,
        }

