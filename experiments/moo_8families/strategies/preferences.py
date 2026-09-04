"""Continuous preference sampling for conditional Pareto-front methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .base import TASK_ORDER, preference_tensor, require_torch


@dataclass
class PreferenceSampleSummary:
    count: int
    mean: list[float]
    min: list[float]
    max: list[float]
    max_simplex_sum_error: float
    coverage_threshold: float
    coverage_fraction: list[float]
    coordinate_nonzero_count: list[int]
    deterministic_reproduction_max_abs_error: float


class ContinuousPreferenceSampler:
    """Deterministic Dirichlet sampler over the task simplex."""

    def __init__(
        self,
        *,
        task_order: Sequence[str] = TASK_ORDER,
        alpha: float | Sequence[float] = 1.0,
        seed: int = 2026,
        coverage_threshold: float = 0.01,
    ):
        self.task_order = tuple(task_order)
        if self.task_order != TASK_ORDER:
            raise ValueError(f"Expected task order {TASK_ORDER}, got {self.task_order}")
        if isinstance(alpha, (int, float)):
            if float(alpha) <= 0:
                raise ValueError(f"Dirichlet alpha must be positive, got {alpha}")
            self.alpha = np.full(len(self.task_order), float(alpha), dtype=np.float64)
        else:
            self.alpha = np.asarray([float(value) for value in alpha], dtype=np.float64)
            if self.alpha.shape != (len(self.task_order),):
                raise ValueError(f"Dirichlet alpha shape mismatch: {self.alpha.shape}")
            if np.any(self.alpha <= 0):
                raise ValueError(f"Dirichlet alpha must be positive: {self.alpha}")
        self.seed = int(seed)
        self.coverage_threshold = float(coverage_threshold)
        self.rng = np.random.default_rng(self.seed)
        self.samples: list[np.ndarray] = []

    def sample_numpy(self) -> np.ndarray:
        sample = self.rng.dirichlet(self.alpha).astype(np.float32)
        sample = sample / np.maximum(sample.sum(), 1e-12)
        self.samples.append(sample.astype(np.float64))
        return sample

    def sample_tensor(self, *, device: Any | None = None, dtype: Any | None = None) -> Any:
        th = require_torch()
        sample = self.sample_numpy()
        return preference_tensor(th.tensor(sample, device=device, dtype=dtype or th.float32))

    def deterministic_reproduction_error(self, *, sample_count: int) -> float:
        left = ContinuousPreferenceSampler(
            task_order=self.task_order,
            alpha=self.alpha.tolist(),
            seed=self.seed,
            coverage_threshold=self.coverage_threshold,
        )
        right = ContinuousPreferenceSampler(
            task_order=self.task_order,
            alpha=self.alpha.tolist(),
            seed=self.seed,
            coverage_threshold=self.coverage_threshold,
        )
        max_error = 0.0
        for _ in range(int(sample_count)):
            max_error = max(max_error, float(np.max(np.abs(left.sample_numpy() - right.sample_numpy()))))
        return max_error

    def diagnostics(self, *, reproduction_samples: int = 128) -> dict[str, Any]:
        if self.samples:
            array = np.stack(self.samples, axis=0)
            simplex_error = np.abs(array.sum(axis=1) - 1.0)
            coverage = (array >= self.coverage_threshold).mean(axis=0)
            nonzero = (array > 0).sum(axis=0)
            mean = array.mean(axis=0)
            min_values = array.min(axis=0)
            max_values = array.max(axis=0)
            max_error = float(simplex_error.max())
        else:
            mean = np.zeros(len(self.task_order), dtype=np.float64)
            min_values = np.zeros(len(self.task_order), dtype=np.float64)
            max_values = np.zeros(len(self.task_order), dtype=np.float64)
            coverage = np.zeros(len(self.task_order), dtype=np.float64)
            nonzero = np.zeros(len(self.task_order), dtype=np.int64)
            max_error = 0.0
        return {
            "distribution": "Dirichlet",
            "alpha": self.alpha.tolist(),
            "seed": self.seed,
            "task_order": list(self.task_order),
            "summary": PreferenceSampleSummary(
                count=int(len(self.samples)),
                mean=mean.astype(float).tolist(),
                min=min_values.astype(float).tolist(),
                max=max_values.astype(float).tolist(),
                max_simplex_sum_error=max_error,
                coverage_threshold=self.coverage_threshold,
                coverage_fraction=coverage.astype(float).tolist(),
                coordinate_nonzero_count=[int(value) for value in nonzero.tolist()],
                deterministic_reproduction_max_abs_error=self.deterministic_reproduction_error(
                    sample_count=reproduction_samples
                ),
            ).__dict__,
        }
