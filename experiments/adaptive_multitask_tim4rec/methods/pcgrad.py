"""PCGrad projection utilities for MultitaskTiM4Rec."""

from __future__ import annotations

import random
from typing import Any

import torch

from experiments.adaptive_multitask_tim4rec.methods.common import (
    TASK_ORDER,
    conflict_summary,
    cosine_matrix,
    gradient_norms,
)


class PCGradProjector:
    """Project conflicting task gradients on shared parameters."""

    def __init__(self, *, mode: str = "ranking_anchored", seed: int = 2026, eps: float = 1e-12) -> None:
        if mode not in {"ranking_anchored", "all_tasks"}:
            raise ValueError(f"Unsupported PCGrad mode: {mode}")
        self.mode = mode
        self.seed = int(seed)
        self.eps = float(eps)

    def _project_one(self, source: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, bool]:
        dot = torch.dot(source.float(), reference.float())
        if float(dot.detach().cpu().item()) >= 0.0:
            return source, False
        denom = torch.dot(reference.float(), reference.float()).clamp_min(self.eps)
        return source - (dot / denom) * reference, True

    def project(self, vectors: dict[str, torch.Tensor], task_order: tuple[str, ...] = TASK_ORDER) -> dict[str, Any]:
        before_matrix = cosine_matrix(vectors, task_order)
        before_conflicts = conflict_summary(before_matrix, task_order)

        adjusted: dict[str, torch.Tensor] = {}
        projection_events: list[dict[str, Any]] = []
        if self.mode == "ranking_anchored":
            rank = vectors["rank"]
            adjusted["rank"] = rank.clone()
            for task in task_order:
                if task == "rank":
                    continue
                projected, changed = self._project_one(vectors[task].clone(), rank)
                adjusted[task] = projected
                if changed:
                    projection_events.append({"source": task, "reference": "rank"})
        else:
            rng = random.Random(self.seed)
            for task in task_order:
                current = vectors[task].clone()
                references = [other for other in task_order if other != task]
                rng.shuffle(references)
                for reference_task in references:
                    current, changed = self._project_one(current, vectors[reference_task])
                    if changed:
                        projection_events.append({"source": task, "reference": reference_task})
                adjusted[task] = current

        after_matrix = cosine_matrix(adjusted, task_order)
        after_conflicts = conflict_summary(after_matrix, task_order)
        combined = torch.stack([adjusted[task] for task in task_order]).sum(dim=0)
        original_combined = torch.stack([vectors[task] for task in task_order]).sum(dim=0)
        return {
            "mode": self.mode,
            "vectors": adjusted,
            "combined_gradient": combined,
            "cosine_matrix_before": before_matrix,
            "cosine_matrix_after": after_matrix,
            "conflicts_before": before_conflicts,
            "conflicts_after": after_conflicts,
            "gradient_norms_before": gradient_norms(vectors, task_order),
            "gradient_norms_after": gradient_norms(adjusted, task_order),
            "projection_events": projection_events,
            "projection_event_count": len(projection_events),
            "combined_gradient_norm_before": float(torch.linalg.vector_norm(original_combined.float()).cpu().item()),
            "combined_gradient_norm_after": float(torch.linalg.vector_norm(combined.float()).cpu().item()),
        }
