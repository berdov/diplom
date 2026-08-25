"""GradNorm auxiliary-loss balancer for MultitaskTiM4Rec."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from experiments.adaptive_multitask_tim4rec.methods.common import (
    AUX_TARGETS,
    ParameterEntry,
    autograd_grads,
    finite_tensor,
    tensor_to_float,
)


class GradNormAuxiliaryBalancer(nn.Module):
    """GradNorm over auxiliary losses while keeping the ranking loss fixed."""

    def __init__(
        self,
        initial_weights: dict[str, float],
        *,
        alpha: float = 1.5,
        learning_rate: float = 0.025,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.targets = AUX_TARGETS
        self.alpha = float(alpha)
        self.eps = float(eps)
        ordered = torch.tensor([float(initial_weights[target]) for target in self.targets], dtype=torch.float32)
        ordered = ordered.clamp_min(self.eps)
        ordered = ordered * (len(self.targets) / ordered.sum())
        self.log_weights = nn.Parameter(ordered.log())
        self.optimizer = torch.optim.Adam([self.log_weights], lr=float(learning_rate))
        self.initial_losses: torch.Tensor | None = None

    def weights_tensor(self) -> torch.Tensor:
        raw = torch.exp(self.log_weights)
        return raw * (len(self.targets) / raw.sum().clamp_min(self.eps))

    def weights_dict(self) -> dict[str, float]:
        values = self.weights_tensor().detach().cpu()
        return {target: float(values[index].item()) for index, target in enumerate(self.targets)}

    def renormalize_(self) -> None:
        with torch.no_grad():
            normalized = self.weights_tensor().detach().clamp_min(self.eps)
            self.log_weights.copy_(normalized.log())

    def maybe_init_losses(self, aux_losses: dict[str, torch.Tensor]) -> None:
        if self.initial_losses is not None:
            return
        losses = torch.stack([aux_losses[target].detach().clamp_min(self.eps) for target in self.targets])
        self.initial_losses = losses

    def weighted_auxiliary_sum(self, aux_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = self.weights_tensor().to(next(iter(aux_losses.values())).device)
        losses = torch.stack([aux_losses[target] for target in self.targets])
        return torch.sum(weights * losses)

    def gradnorm_loss(
        self,
        aux_losses: dict[str, torch.Tensor],
        shared_entries: list[ParameterEntry],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.maybe_init_losses(aux_losses)
        if self.initial_losses is None:
            raise RuntimeError("GradNorm initial losses were not initialized.")
        weights = self.weights_tensor().to(next(iter(aux_losses.values())).device)
        current_losses = torch.stack([aux_losses[target] for target in self.targets])
        weighted_losses = weights * current_losses
        norms = []
        for weighted_loss in weighted_losses:
            grads = autograd_grads(weighted_loss, shared_entries, retain_graph=True, create_graph=True)
            squared = torch.zeros((), dtype=weighted_loss.dtype, device=weighted_loss.device)
            for grad in grads:
                if grad is not None:
                    squared = squared + torch.sum(grad.pow(2))
            norms.append(torch.sqrt(squared + self.eps))
        grad_norms = torch.stack(norms)

        loss_ratios = current_losses.detach().clamp_min(self.eps) / self.initial_losses.to(current_losses.device)
        relative_inverse_rates = loss_ratios / loss_ratios.mean().clamp_min(self.eps)
        target_norms = grad_norms.detach().mean() * relative_inverse_rates.pow(self.alpha)
        grad_loss = torch.nn.functional.l1_loss(grad_norms, target_norms.detach(), reduction="sum")
        diagnostics = {
            "alpha": self.alpha,
            "gradnorm_loss": tensor_to_float(grad_loss),
            "weights": self.weights_dict(),
            "gradient_norms": {
                target: tensor_to_float(grad_norms[index]) for index, target in enumerate(self.targets)
            },
            "target_gradient_norms": {
                target: tensor_to_float(target_norms[index]) for index, target in enumerate(self.targets)
            },
            "relative_inverse_training_rates": {
                target: tensor_to_float(relative_inverse_rates[index]) for index, target in enumerate(self.targets)
            },
            "loss_ratios": {
                target: tensor_to_float(loss_ratios[index]) for index, target in enumerate(self.targets)
            },
        }
        return grad_loss, diagnostics

    def step_weights(
        self,
        aux_losses: dict[str, torch.Tensor],
        shared_entries: list[ParameterEntry],
    ) -> dict[str, Any]:
        before = self.weights_dict()
        self.optimizer.zero_grad(set_to_none=True)
        grad_loss, diagnostics = self.gradnorm_loss(aux_losses, shared_entries)
        grad_loss.backward()
        finite = self.log_weights.grad is not None and finite_tensor(self.log_weights.grad)
        if not finite:
            raise RuntimeError("Non-finite GradNorm task-weight gradient.")
        self.optimizer.step()
        self.renormalize_()
        after = self.weights_dict()
        diagnostics.update(
            {
                "weights_before": before,
                "weights_after": after,
                "weights_changed": any(abs(after[target] - before[target]) > 1e-12 for target in self.targets),
                "log_weight_gradient_finite": bool(finite),
            }
        )
        return diagnostics
