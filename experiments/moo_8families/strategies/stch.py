"""Smooth Tchebycheff scalarization.

Formula reference:
    g_mu(x | lambda) = mu * log(sum_i exp(lambda_i * (f_i(x) - z_i*) / mu))

The stable implementation subtracts a detached max term. This changes the
reported scalar value by a detached constant, but preserves gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .base import TASK_ORDER, normalize_loss_vector, preference_tensor, require_torch, tensor_to_float


@dataclass
class STCHState:
    steps: int
    warmup_steps: int
    nadir_vector: list[float] | None


class SmoothTchebycheffScalarizer:
    """Faithful STCH scalarizer with official-style log loss normalization."""

    def __init__(
        self,
        *,
        task_order: Sequence[str] = TASK_ORDER,
        mu: float = 1.0,
        warmup_steps: int = 0,
        preference: Sequence[float] | None = None,
        nadir_vector: Sequence[float] | None = None,
        reference_point: Sequence[float] | None = None,
        eps: float = 1e-20,
        multiply_by_task_count: bool = True,
    ):
        if mu <= 0:
            raise ValueError(f"STCH mu must be positive, got {mu}")
        self.task_order = tuple(task_order)
        if self.task_order != TASK_ORDER:
            raise ValueError(f"Expected task order {TASK_ORDER}, got {self.task_order}")
        self.mu = float(mu)
        self.warmup_steps = int(max(warmup_steps, 0))
        self.eps = float(eps)
        self.multiply_by_task_count = bool(multiply_by_task_count)
        self._preference_source = list(preference or [1.0] * len(self.task_order))
        self.reference_point_source = list(reference_point or [0.0] * len(self.task_order))
        self._nadir_source = None if nadir_vector is None else [float(value) for value in nadir_vector]
        self._nadir_tensor = None
        self._running_loss_sum = None
        self._running_loss_steps = 0
        self.steps = 0

    def _preference(self, losses: Any) -> Any:
        return preference_tensor(self._preference_source, device=losses.device, dtype=losses.dtype)

    def _reference_point(self, losses: Any) -> Any:
        th = require_torch()
        return th.tensor(self.reference_point_source, dtype=losses.dtype, device=losses.device)

    def _nadir(self, losses: Any) -> Any:
        th = require_torch()
        if self._nadir_tensor is None:
            if self._nadir_source is None:
                return None
            self._nadir_tensor = th.tensor(self._nadir_source, dtype=losses.dtype, device=losses.device)
        return self._nadir_tensor.to(device=losses.device, dtype=losses.dtype)

    def observe_train_losses(self, losses: Any) -> None:
        detached = losses.detach().float().cpu()
        if self._running_loss_sum is None:
            self._running_loss_sum = detached.clone()
        else:
            self._running_loss_sum += detached
        self._running_loss_steps += 1
        if self._nadir_source is None and self.warmup_steps > 0 and self._running_loss_steps >= self.warmup_steps:
            mean = self._running_loss_sum / max(self._running_loss_steps, 1)
            self._nadir_source = [float(value) for value in mean.tolist()]

    def scalarize(self, losses: Any, *, loss_scales: Sequence[float] | Any | None = None) -> Any:
        th = require_torch()
        vector = normalize_loss_vector(losses, loss_scales)
        pref = self._preference(vector)
        nadir = self._nadir(vector)
        if nadir is not None:
            terms = th.log(vector.clamp_min(self.eps) / nadir.clamp_min(self.eps))
        elif self.warmup_steps > 0 and self.steps < self.warmup_steps:
            terms = th.log(vector.clamp_min(self.eps))
        else:
            terms = vector
        terms = pref * (terms - self._reference_point(vector))
        shift = terms.detach().max()
        scalar = self.mu * th.log(th.exp((terms - shift) / self.mu).sum().clamp_min(self.eps))
        if self.multiply_by_task_count:
            scalar = scalar * len(self.task_order)
        self.steps += 1
        return scalar

    def state_dict(self) -> dict[str, Any]:
        return {
            "steps": int(self.steps),
            "warmup_steps": int(self.warmup_steps),
            "mu": self.mu,
            "preference": list(self._preference_source),
            "reference_point": list(self.reference_point_source),
            "nadir_vector": self._nadir_source,
            "running_loss_steps": int(self._running_loss_steps),
        }

    def diagnostic(self, losses: Any, loss_scales: Sequence[float] | Any | None = None) -> dict[str, Any]:
        vector = normalize_loss_vector(losses, loss_scales)
        pref = self._preference(vector)
        linear = (pref * vector).sum()
        stch = self.scalarize(vector)
        return {
            "stch": tensor_to_float(stch),
            "linear_weighted_sum": tensor_to_float(linear),
            "not_equal_to_linear": abs(tensor_to_float(stch) - tensor_to_float(linear)) > 1e-8,
            "state": self.state_dict(),
        }

