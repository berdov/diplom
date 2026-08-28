"""FAMO adaptive task weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import TASK_ORDER, require_torch, tensor_to_float


@dataclass
class FAMOUpdate:
    effective_weights: list[float]
    min_losses: list[float]
    previous_losses: list[float] | None
    current_losses: list[float] | None


class FAMO:
    """Fast Adaptive Multitask Optimization following the official implementation."""

    def __init__(
        self,
        *,
        n_tasks: int = len(TASK_ORDER),
        device: Any | None = None,
        gamma: float = 0.01,
        w_lr: float = 0.025,
        max_norm: float = 1.0,
        eps: float = 1e-8,
    ):
        th = require_torch()
        self.n_tasks = int(n_tasks)
        self.gamma = float(gamma)
        self.w_lr = float(w_lr)
        self.max_norm = float(max_norm)
        self.eps = float(eps)
        self.min_losses = th.zeros(self.n_tasks, device=device)
        self.w = th.zeros(self.n_tasks, device=device, requires_grad=True)
        self.w_opt = th.optim.Adam([self.w], lr=self.w_lr, weight_decay=self.gamma)
        self.prev_loss = None
        self.last_update: FAMOUpdate | None = None

    def effective_weights(self) -> Any:
        th = require_torch()
        return th.softmax(self.w, dim=-1)

    def get_weighted_loss(self, losses: Any) -> Any:
        th = require_torch()
        if losses.ndim != 1 or losses.numel() != self.n_tasks:
            raise ValueError(f"Expected {self.n_tasks} task losses, got {tuple(losses.shape)}")
        self.prev_loss = losses.detach()
        z = self.effective_weights()
        denom = (losses - self.min_losses.to(losses.device)).clamp_min(self.eps)
        c = (z / denom).sum().detach()
        return (denom.log() * z / c.clamp_min(self.eps)).sum()

    def backward(self, losses: Any, shared_parameters: Any | None = None) -> Any:
        th = require_torch()
        loss = self.get_weighted_loss(losses)
        loss.backward()
        if shared_parameters is not None and self.max_norm > 0:
            th.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss

    def update(self, current_losses: Any) -> FAMOUpdate:
        th = require_torch()
        if self.prev_loss is None:
            raise RuntimeError("FAMO.update() requires get_weighted_loss/backward before optimizer.step().")
        current = current_losses.detach().to(self.w.device)
        prev = self.prev_loss.detach().to(self.w.device)
        delta = (
            (prev - self.min_losses).clamp_min(self.eps).log()
            - (current - self.min_losses).clamp_min(self.eps).log()
        )
        weights = self.effective_weights()
        grad = th.autograd.grad(weights, self.w, grad_outputs=delta.detach(), retain_graph=False)[0]
        self.w_opt.zero_grad(set_to_none=True)
        self.w.grad = grad
        self.w_opt.step()
        self.last_update = FAMOUpdate(
            effective_weights=[tensor_to_float(value) for value in self.effective_weights()],
            min_losses=[tensor_to_float(value) for value in self.min_losses],
            previous_losses=[tensor_to_float(value) for value in prev],
            current_losses=[tensor_to_float(value) for value in current],
        )
        return self.last_update

    def state_dict(self) -> dict[str, Any]:
        state = {
            "n_tasks": self.n_tasks,
            "gamma": self.gamma,
            "w_lr": self.w_lr,
            "max_norm": self.max_norm,
            "w": [tensor_to_float(value) for value in self.w.detach()],
            "effective_weights": [tensor_to_float(value) for value in self.effective_weights()],
            "min_losses": [tensor_to_float(value) for value in self.min_losses],
        }
        if self.last_update is not None:
            state["last_update"] = self.last_update.__dict__
        return state

