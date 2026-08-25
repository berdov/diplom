"""MetaBalance-style auxiliary gradient magnitude scaling."""

from __future__ import annotations

from typing import Any

import torch

from experiments.adaptive_multitask_tim4rec.methods.common import (
    TASK_ORDER,
    ParameterEntry,
    autograd_grads,
    conflict_summary,
    cosine_matrix,
    flatten_grads,
    gradient_norms,
)


class MetaBalanceAuxiliaryBalancer:
    """MetaBalance-Fix for ranking-primary auxiliary learning."""

    def __init__(self, *, relax_factor: float = 0.7, beta: float = 0.9, eps: float = 1e-12) -> None:
        if not 0.0 <= relax_factor < 1.0:
            raise ValueError(f"relax_factor must be in [0, 1), got {relax_factor}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.relax_factor = float(relax_factor)
        self.beta = float(beta)
        self.eps = float(eps)
        self.moving_norms: dict[str, list[torch.Tensor]] = {}

    def _moving_for(self, task: str, grads: tuple[torch.Tensor | None, ...], entries: list[ParameterEntry]) -> list[torch.Tensor]:
        current = []
        for grad, entry in zip(grads, entries):
            if grad is None:
                current.append(entry.parameter.new_zeros(()))
            else:
                current.append(torch.linalg.vector_norm(grad.detach().float()))
        if task not in self.moving_norms:
            self.moving_norms[task] = [value.detach().new_zeros(()) for value in current]
        self.moving_norms[task] = [
            previous.to(value.device) * self.beta + value.detach() * (1.0 - self.beta)
            for previous, value in zip(self.moving_norms[task], current)
        ]
        return self.moving_norms[task]

    def balanced_shared_gradients(
        self,
        task_losses: dict[str, torch.Tensor],
        shared_entries: list[ParameterEntry],
        task_order: tuple[str, ...] = TASK_ORDER,
    ) -> dict[str, Any]:
        grads_by_task: dict[str, tuple[torch.Tensor | None, ...]] = {}
        moving_by_task: dict[str, list[torch.Tensor]] = {}
        before_vectors: dict[str, torch.Tensor] = {}
        after_vectors: dict[str, torch.Tensor] = {}
        adjusted_by_task: dict[str, list[torch.Tensor]] = {}
        scale_values: dict[str, list[float]] = {task: [] for task in task_order}

        for index, task in enumerate(task_order):
            grads = autograd_grads(task_losses[task], shared_entries, retain_graph=True, create_graph=False)
            grads_by_task[task] = grads
            moving_by_task[task] = self._moving_for(task, grads, shared_entries)
            before_vectors[task] = flatten_grads(grads, shared_entries, detach=True)

        target_moving = moving_by_task["rank"]
        for task in task_order:
            adjusted: list[torch.Tensor] = []
            task_moving = moving_by_task[task]
            for param_index, (grad, entry) in enumerate(zip(grads_by_task[task], shared_entries)):
                base = torch.zeros_like(entry.parameter) if grad is None else grad.detach()
                if task == "rank":
                    scaled = base
                    scale = 1.0
                else:
                    ratio = target_moving[param_index].to(base.device) / task_moving[param_index].to(base.device).clamp_min(self.eps)
                    scaled = self.relax_factor * ratio * base + (1.0 - self.relax_factor) * base
                    scale = float((self.relax_factor * ratio + (1.0 - self.relax_factor)).detach().cpu().item())
                adjusted.append(scaled)
                scale_values[task].append(scale)
            adjusted_by_task[task] = adjusted
            after_vectors[task] = flatten_grads(adjusted, shared_entries, detach=True)

        combined = []
        for param_index, entry in enumerate(shared_entries):
            total = torch.zeros_like(entry.parameter)
            for task in task_order:
                total = total + adjusted_by_task[task][param_index]
            combined.append(total)

        before_matrix = cosine_matrix(before_vectors, task_order)
        after_matrix = cosine_matrix(after_vectors, task_order)
        return {
            "method": "MetaBalance-Fix",
            "relax_factor": self.relax_factor,
            "beta": self.beta,
            "combined_gradients": combined,
            "vectors_before": before_vectors,
            "vectors_after": after_vectors,
            "cosine_matrix_before": before_matrix,
            "cosine_matrix_after": after_matrix,
            "conflicts_before": conflict_summary(before_matrix, task_order),
            "conflicts_after": conflict_summary(after_matrix, task_order),
            "gradient_norms_before": gradient_norms(before_vectors, task_order),
            "gradient_norms_after": gradient_norms(after_vectors, task_order),
            "scale_summary": {
                task: {
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "mean": sum(values) / len(values) if values else None,
                }
                for task, values in scale_values.items()
            },
        }
