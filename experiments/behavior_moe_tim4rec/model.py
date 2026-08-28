#!/usr/bin/env python
"""Behavior-specialized MoE extension for tuned MultitaskTiM4Rec."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS


GENERIC_EXPERTS: tuple[str, ...] = ("interest", "consumption", "positive", "shared")
STRUCTURED_EXPERTS: tuple[str, ...] = ("interest", "consumption", "engagement", "shared")
EXPERTS = GENERIC_EXPERTS
ROUTING_TASKS: tuple[str, ...] = ("rank", *TARGETS)
PLE_SPECIFIC_EXPERTS: dict[str, str] = {
    "rank": "ranking_specific",
    "is_click": "click_specific",
    "long_view": "long_view_specific",
    "is_like": "like_specific",
    "is_profile_enter": "profile_specific",
}
PLE_SHARED_EXPERTS: tuple[str, ...] = ("shared_0", "shared_1")
PLE_EXPERTS: tuple[str, ...] = (
    "ranking_specific",
    "click_specific",
    "long_view_specific",
    "like_specific",
    "profile_specific",
    *PLE_SHARED_EXPERTS,
)
GENERIC_TASK_TO_SEMANTIC_EXPERT: dict[str, str | None] = {
    "rank": None,
    "is_click": "interest",
    "long_view": "consumption",
    "is_like": "positive",
    "is_profile_enter": "positive",
}
STRUCTURED_TASK_TO_SEMANTIC_EXPERT: dict[str, str | None] = {
    "rank": None,
    "is_click": "interest",
    "long_view": "consumption",
    "is_like": "engagement",
    "is_profile_enter": "engagement",
}
TASK_TO_SEMANTIC_EXPERT = GENERIC_TASK_TO_SEMANTIC_EXPERT
EXPERT_SETS: dict[str, tuple[str, ...]] = {
    "generic": GENERIC_EXPERTS,
    "structured": STRUCTURED_EXPERTS,
}
GENERIC_ALLOWED_EXPERTS: dict[str, tuple[str, ...]] = {
    task: GENERIC_EXPERTS for task in ROUTING_TASKS
}
STRUCTURED_ALLOWED_EXPERTS: dict[str, tuple[str, ...]] = {
    "rank": STRUCTURED_EXPERTS,
    "is_click": ("interest", "shared"),
    "long_view": ("consumption", "shared"),
    "is_like": ("engagement", "shared"),
    "is_profile_enter": ("engagement", "shared"),
}
ALLOWED_EXPERTS_BY_MODE: dict[str, dict[str, tuple[str, ...]]] = {
    "generic": GENERIC_ALLOWED_EXPERTS,
    "structured": STRUCTURED_ALLOWED_EXPERTS,
}
SEMANTIC_EXPERT_BY_MODE: dict[str, dict[str, str | None]] = {
    "generic": GENERIC_TASK_TO_SEMANTIC_EXPERT,
    "structured": STRUCTURED_TASK_TO_SEMANTIC_EXPERT,
}


def config_get(config: Any, key: str, default: Any) -> Any:
    try:
        value = config[key]
    except Exception:
        value = default
    return default if value is None else value


class BehaviorExpert(nn.Module):
    """Small MLP expert over the shared TiM4Rec representation."""

    def __init__(self, hidden_size: int, dropout: float, expert_hidden_size: int | None = None):
        super().__init__()
        inner_size = int(expert_hidden_size or hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class BehaviorMoETiM4Rec(MultitaskTiM4Rec):
    """TiM4Rec backbone plus task-conditioned behavior-specialized MoE."""

    def __init__(self, config: Any, dataset: Any):
        super().__init__(config, dataset)
        moe_config = dict(config_get(config, "behavior_moe", {}) or {})
        self.routing_mode = str(moe_config.get("routing_mode", "generic"))
        if self.routing_mode not in EXPERT_SETS:
            raise ValueError(f"Unsupported routing mode: {self.routing_mode}")
        expected_experts = EXPERT_SETS[self.routing_mode]
        self.expert_names = tuple(moe_config.get("experts", expected_experts))
        if self.expert_names != expected_experts:
            raise ValueError(
                f"Unsupported expert set for routing_mode={self.routing_mode}: {self.expert_names}"
            )
        self.routing_tasks = ROUTING_TASKS
        configured_allowed = dict(moe_config.get("allowed_experts", {}) or {})
        default_allowed = ALLOWED_EXPERTS_BY_MODE[self.routing_mode]
        self.allowed_experts = {
            task: tuple(configured_allowed.get(task, default_allowed[task]))
            for task in self.routing_tasks
        }
        for task, allowed in self.allowed_experts.items():
            unknown = [expert for expert in allowed if expert not in self.expert_names]
            if unknown:
                raise ValueError(f"Unknown experts in allowed_experts[{task}]: {unknown}")
            if not allowed:
                raise ValueError(f"allowed_experts[{task}] must not be empty.")
        self.allowed_expert_indices = {
            task: tuple(self.expert_names.index(expert) for expert in allowed)
            for task, allowed in self.allowed_experts.items()
        }
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
            expert = SEMANTIC_EXPERT_BY_MODE[self.routing_mode][task]
            if expert is not None:
                router.bias.data[self.expert_names.index(expert)] = self.router_semantic_bias

    def expert_outputs(self, seq_output: torch.Tensor) -> torch.Tensor:
        outputs = [self.experts[name](seq_output) for name in self.expert_names]
        return torch.stack(outputs, dim=1)

    def routing_logits_from_shared(
        self,
        seq_output: torch.Tensor,
        *,
        masked: bool = False,
    ) -> dict[str, torch.Tensor]:
        logits_by_task = {}
        for task, router in self.router_heads.items():
            logits = router(seq_output) / self.router_temperature
            logits_by_task[task] = self._masked_logits(task, logits) if masked else logits
        return logits_by_task

    def _masked_logits(self, task: str, logits: torch.Tensor) -> torch.Tensor:
        allowed = self.allowed_expert_indices[task]
        if len(allowed) == len(self.expert_names):
            return logits
        mask = torch.zeros(len(self.expert_names), dtype=torch.bool, device=logits.device)
        mask[list(allowed)] = True
        return logits.masked_fill(~mask.unsqueeze(0), torch.finfo(logits.dtype).min)

    def routing_weights_from_shared(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = {}
        for task, logits in self.routing_logits_from_shared(seq_output, masked=True).items():
            weights[task] = torch.softmax(logits, dim=-1)
        return weights

    def local_routing_weights_from_shared(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        expanded = self.routing_weights_from_shared(seq_output)
        return {
            task: expanded[task][:, list(self.allowed_expert_indices[task])]
            for task in self.routing_tasks
        }

    def routing_metadata(self) -> dict[str, Any]:
        return {
            "routing_mode": self.routing_mode,
            "experts": list(self.expert_names),
            "allowed_experts": {
                task: list(experts) for task, experts in self.allowed_experts.items()
            },
        }

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


class StructuredBehaviorMoETiM4Rec(BehaviorMoETiM4Rec):
    """Behavior-MoE probe with structured task-to-expert access masks."""

    def __init__(self, config: Any, dataset: Any):
        super().__init__(config, dataset)
        if self.routing_mode != "structured":
            raise ValueError(
                "StructuredBehaviorMoETiM4Rec requires behavior_moe.routing_mode=structured."
            )


class PLETiM4Rec(MultitaskTiM4Rec):
    """One-level CGC/PLE-style task-specific/shared experts over TiM4Rec h."""

    def __init__(self, config: Any, dataset: Any):
        super().__init__(config, dataset)
        ple_config = dict(config_get(config, "ple_tim4rec", {}) or {})
        self.routing_mode = "ple"
        self.extraction_levels = int(ple_config.get("extraction_levels", 1))
        if self.extraction_levels != 1:
            raise ValueError("PLETiM4Rec currently implements the canonical one-level CGC baseline.")
        self.specific_experts_per_task = int(ple_config.get("specific_experts_per_task", 1))
        if self.specific_experts_per_task != 1:
            raise ValueError("PLETiM4Rec baseline is fixed to one specific expert per task.")
        self.shared_expert_count = int(ple_config.get("shared_experts", len(PLE_SHARED_EXPERTS)))
        if self.shared_expert_count != len(PLE_SHARED_EXPERTS):
            raise ValueError(f"PLETiM4Rec baseline expects {len(PLE_SHARED_EXPERTS)} shared experts.")

        self.routing_tasks = ROUTING_TASKS
        self.task_specific_experts = dict(PLE_SPECIFIC_EXPERTS)
        self.shared_experts = tuple(PLE_SHARED_EXPERTS)
        self.expert_names = tuple(PLE_EXPERTS)
        self.allowed_experts = {
            task: (self.task_specific_experts[task], *self.shared_experts)
            for task in self.routing_tasks
        }
        self.allowed_expert_indices = {
            task: tuple(self.expert_names.index(expert) for expert in allowed)
            for task, allowed in self.allowed_experts.items()
        }
        self.router_temperature = float(ple_config.get("router_temperature", 1.0))
        dropout = float(ple_config.get("expert_dropout", config_get(config, "dropout_prob", 0.0)))
        expert_hidden_size = int(ple_config.get("expert_hidden_size", max(1, self.hidden_size)))

        self.experts = nn.ModuleDict(
            {
                name: BehaviorExpert(
                    self.hidden_size,
                    dropout=dropout,
                    expert_hidden_size=expert_hidden_size,
                )
                for name in self.expert_names
            }
        )
        self.router_heads = nn.ModuleDict(
            {
                task: nn.Linear(self.hidden_size, len(self.allowed_experts[task]))
                for task in self.routing_tasks
            }
        )
        self._init_ple()

    def _init_ple(self) -> None:
        for expert in self.experts.values():
            expert.apply(self._init_weights)
        for router in self.router_heads.values():
            nn.init.zeros_(router.weight)
            nn.init.zeros_(router.bias)

    def expert_outputs(self, seq_output: torch.Tensor) -> torch.Tensor:
        outputs = [self.experts[name](seq_output) for name in self.expert_names]
        return torch.stack(outputs, dim=1)

    def _expanded_local_values(
        self,
        task: str,
        local_values: torch.Tensor,
        *,
        fill_value: float,
    ) -> torch.Tensor:
        expanded = torch.full(
            (local_values.shape[0], len(self.expert_names)),
            fill_value,
            dtype=local_values.dtype,
            device=local_values.device,
        )
        expanded[:, list(self.allowed_expert_indices[task])] = local_values
        return expanded

    def local_routing_logits_from_shared(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            task: self.router_heads[task](seq_output) / self.router_temperature
            for task in self.routing_tasks
        }

    def routing_logits_from_shared(
        self,
        seq_output: torch.Tensor,
        *,
        masked: bool = False,
    ) -> dict[str, torch.Tensor]:
        fill_value = torch.finfo(seq_output.dtype).min if masked else 0.0
        return {
            task: self._expanded_local_values(task, logits, fill_value=fill_value)
            for task, logits in self.local_routing_logits_from_shared(seq_output).items()
        }

    def routing_weights_from_shared(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = {}
        for task, logits in self.local_routing_logits_from_shared(seq_output).items():
            local_weights = torch.softmax(logits, dim=-1)
            weights[task] = self._expanded_local_values(task, local_weights, fill_value=0.0)
        return weights

    def local_routing_weights_from_shared(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            task: torch.softmax(logits, dim=-1)
            for task, logits in self.local_routing_logits_from_shared(seq_output).items()
        }

    def routing_metadata(self) -> dict[str, Any]:
        return {
            "routing_mode": self.routing_mode,
            "extraction_levels": self.extraction_levels,
            "experts": list(self.expert_names),
            "task_specific_experts": dict(self.task_specific_experts),
            "shared_experts": list(self.shared_experts),
            "allowed_experts": {
                task: list(experts) for task, experts in self.allowed_experts.items()
            },
        }

    def task_representations_from_shared(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        expert_outputs = self.expert_outputs(seq_output)
        weights = self.routing_weights_from_shared(seq_output)
        return {
            task: torch.einsum("be,beh->bh", weights[task], expert_outputs)
            for task in self.routing_tasks
        }

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
        return seq_output.new_zeros(())

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

        load_balance = self.load_balance_loss_from_shared(seq_output)
        balance_weight = 0.0 if load_balance_weight is None else float(load_balance_weight)
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
