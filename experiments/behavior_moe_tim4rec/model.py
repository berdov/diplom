#!/usr/bin/env python
"""Behavior-specialized MoE extension for tuned MultitaskTiM4Rec."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS


EXPERTS: tuple[str, ...] = ("interest", "consumption", "positive", "shared")
ROUTING_TASKS: tuple[str, ...] = ("rank", *TARGETS)
TASK_TO_SEMANTIC_EXPERT: dict[str, str | None] = {
    "rank": None,
    "is_click": "interest",
    "long_view": "consumption",
    "is_like": "positive",
    "is_profile_enter": "positive",
}


def config_get(config: Any, key: str, default: Any) -> Any:
    try:
        value = config[key]
    except Exception:
        value = default
    return default if value is None else value


class BehaviorExpert(nn.Module):
    """Small residual expert over the shared TiM4Rec representation."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class BehaviorMoETiM4Rec(MultitaskTiM4Rec):
    """TiM4Rec backbone plus task-conditioned behavior-specialized MoE."""

    def __init__(self, config: Any, dataset: Any):
        super().__init__(config, dataset)
        moe_config = dict(config_get(config, "behavior_moe", {}) or {})
        self.expert_names = tuple(moe_config.get("experts", EXPERTS))
        if self.expert_names != EXPERTS:
            raise ValueError(f"Unsupported expert set: {self.expert_names}")
        self.routing_tasks = ROUTING_TASKS
        self.router_temperature = float(moe_config.get("router_temperature", 1.0))
        self.residual_scale = float(moe_config.get("residual_scale", 0.1))
        self.load_balance_weight = float(moe_config.get("load_balance_weight", 0.0))
        self.router_semantic_bias = float(moe_config.get("router_semantic_bias", 0.05))
        dropout = float(moe_config.get("expert_dropout", config_get(config, "dropout_prob", 0.0)))

        self.experts = nn.ModuleDict(
            {name: BehaviorExpert(self.hidden_size, dropout=dropout) for name in self.expert_names}
        )
        self.router_heads = nn.ModuleDict(
            {task: nn.Linear(self.hidden_size, len(self.expert_names)) for task in self.routing_tasks}
        )
        self._init_behavior_moe()

    def _init_behavior_moe(self) -> None:
        for expert in self.experts.values():
            expert.apply(self._init_weights)
        for task, router in self.router_heads.items():
            nn.init.zeros_(router.weight)
            nn.init.zeros_(router.bias)
            expert = TASK_TO_SEMANTIC_EXPERT[task]
            if expert is not None:
                router.bias.data[self.expert_names.index(expert)] = self.router_semantic_bias

    def expert_outputs(self, seq_output: torch.Tensor) -> torch.Tensor:
        outputs = [self.experts[name](seq_output) for name in self.expert_names]
        return torch.stack(outputs, dim=1)

    def routing_weights_from_shared(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = {}
        for task, router in self.router_heads.items():
            logits = router(seq_output) / self.router_temperature
            weights[task] = torch.softmax(logits, dim=-1)
        return weights

    def task_representations_from_shared(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        expert_outputs = self.expert_outputs(seq_output)
        weights = self.routing_weights_from_shared(seq_output)
        representations = {}
        for task in self.routing_tasks:
            delta = torch.einsum("be,beh->bh", weights[task], expert_outputs)
            representations[task] = seq_output + self.residual_scale * delta
        return representations

    def representation_for_task(self, seq_output: torch.Tensor, task: str) -> torch.Tensor:
        if task not in self.routing_tasks:
            raise KeyError(f"Unknown routing task: {task}")
        return self.task_representations_from_shared(seq_output)[task]

    def scores_from_task_representation(self, task_representation: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bh,nh->bn", task_representation, self.item_embedding.weight)

    def ranking_logits_from_representation(self, seq_output: torch.Tensor) -> torch.Tensor:
        rank_representation = self.representation_for_task(seq_output, "rank")
        return self.scores_from_task_representation(rank_representation)

    def auxiliary_logits_from_representation(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        representations = self.task_representations_from_shared(seq_output)
        heads = self.auxiliary_heads()
        return {
            target: heads[target](representations[target]).squeeze(-1)
            for target in self.multitask_targets
        }

    def full_sort_predict(self, interaction: Any) -> torch.Tensor:
        seq_output = self.shared_representation(interaction)
        rank_representation = self.representation_for_task(seq_output, "rank")
        return self.scores_from_task_representation(rank_representation)

    def predict(self, interaction: Any) -> torch.Tensor:
        seq_output = self.shared_representation(interaction)
        rank_representation = self.representation_for_task(seq_output, "rank")
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        return torch.einsum("bh,bh->b", rank_representation, test_item_emb)

    def load_balance_loss_from_shared(self, seq_output: torch.Tensor) -> torch.Tensor:
        weights = self.routing_weights_from_shared(seq_output)
        usage = torch.stack([value.mean(dim=0) for value in weights.values()]).mean(dim=0)
        target = torch.full_like(usage, 1.0 / len(self.expert_names))
        return torch.mean((usage - target) ** 2)

    def calculate_multitask_loss(
        self,
        interaction: Any,
        *,
        lambda_aux: float,
        pos_weights: dict[str, torch.Tensor] | None,
        task_weights: dict[str, float] | None = None,
        load_balance_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_output = self.shared_representation(interaction)
        pos_items = interaction[self.POS_ITEM_ID]
        rank_logits = self.ranking_logits_from_representation(seq_output)
        rank_loss = self.loss_fct(rank_logits, pos_items)

        aux_logits = self.auxiliary_logits_from_representation(seq_output)
        aux_losses: dict[str, torch.Tensor] = {}
        weighted_aux = torch.zeros((), dtype=rank_loss.dtype, device=rank_loss.device)
        unweighted_aux = torch.zeros((), dtype=rank_loss.dtype, device=rank_loss.device)
        for target, logits in aux_logits.items():
            labels = interaction[target].float()
            weight = None if pos_weights is None else pos_weights[target].to(logits.device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
            aux_losses[target] = loss
            unweighted_aux = unweighted_aux + loss
            multiplier = 1.0 if task_weights is None else float(task_weights[target])
            weighted_aux = weighted_aux + multiplier * loss

        balance_weight = self.load_balance_weight if load_balance_weight is None else float(load_balance_weight)
        load_balance = self.load_balance_loss_from_shared(seq_output)
        total = rank_loss + float(lambda_aux) * weighted_aux + balance_weight * load_balance
        result = {
            "total": total,
            "rank": rank_loss,
            "aux_sum": unweighted_aux,
            "weighted_aux_sum": weighted_aux,
            "load_balance_loss": load_balance,
            "load_balance_contribution": balance_weight * load_balance,
        }
        result.update({f"{target}_loss": loss for target, loss in aux_losses.items()})
        result.update(
            {
                f"{target}_scaled_contribution": float(lambda_aux)
                * (1.0 if task_weights is None else float(task_weights[target]))
                * loss
                for target, loss in aux_losses.items()
            }
        )
        return result

    def calculate_loss(self, interaction: Any) -> torch.Tensor:
        lambda_aux = float(getattr(self, "lambda_aux", 0.0))
        pos_weights = getattr(self, "pos_weights", None)
        task_weights = getattr(self, "task_weights", None)
        return self.calculate_multitask_loss(
            interaction,
            lambda_aux=lambda_aux,
            pos_weights=pos_weights,
            task_weights=task_weights,
        )["total"]
