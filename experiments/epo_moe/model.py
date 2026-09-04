"""Task-routed Mixture-of-Experts adapter for MultitaskTiM4Rec."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from experiments.moo_8families.strategies.base import TASK_ORDER
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported MoE activation: {name}")


class MoEExpert(nn.Module):
    """Small shared MLP expert over the final TiM4Rec representation."""

    def __init__(self, hidden_size: int, expert_hidden_size: int, dropout: float, activation: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden_size),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(expert_hidden_size, hidden_size),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class MoEMultitaskTiM4Rec(MultitaskTiM4Rec):
    """TiM4Rec backbone with shared experts and task-specific gates.

    The insertion point is deliberately after `shared_representation()`:
    TiM4Rec still builds the sequence representation `h`, then each task gets
    its own gated mixture over shared MLP experts before the existing head.
    """

    def __init__(self, config: Any, dataset: Any):
        super().__init__(config, dataset)
        self.moe_num_experts = int(config["moe_num_experts"])
        self.moe_expert_hidden_size = int(config["moe_expert_hidden_size"])
        self.moe_dropout = float(config["moe_dropout"])
        self.moe_gate_temperature = float(config["moe_gate_temperature"])
        self.moe_residual = bool(config["moe_residual"])
        self.moe_residual_scale = float(config["moe_residual_scale"])
        self.moe_activation = str(config["moe_activation"])
        self.moe_task_order = tuple(TASK_ORDER)
        if self.moe_task_order != ("rank", *self.multitask_targets):
            raise ValueError(
                f"MoE task order must match rank + multitask targets: "
                f"{self.moe_task_order} vs {('rank', *self.multitask_targets)}"
            )
        if self.moe_num_experts < 1:
            raise ValueError("MoEMultitaskTiM4Rec requires moe_num_experts >= 1.")
        if self.moe_expert_hidden_size < 1:
            raise ValueError("moe_expert_hidden_size must be positive.")
        if self.moe_gate_temperature <= 0.0:
            raise ValueError("moe_gate_temperature must be positive.")

        self.moe_experts = nn.ModuleList(
            [
                MoEExpert(
                    hidden_size=self.hidden_size,
                    expert_hidden_size=self.moe_expert_hidden_size,
                    dropout=self.moe_dropout,
                    activation=self.moe_activation,
                )
                for _ in range(self.moe_num_experts)
            ]
        )
        self.moe_gates = nn.ModuleDict(
            {task: nn.Linear(self.hidden_size, self.moe_num_experts) for task in self.moe_task_order}
        )
        for module in self.moe_experts:
            module.apply(self._init_weights)
        for gate in self.moe_gates.values():
            self._init_weights(gate)

    def moe_enabled(self) -> bool:
        return True

    def _expert_outputs(self, seq_output: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(seq_output) for expert in self.moe_experts], dim=1)

    def gate_probabilities(self, seq_output: torch.Tensor, task: str) -> torch.Tensor:
        if task not in self.moe_gates:
            raise KeyError(f"Unknown MoE task gate: {task}")
        logits = self.moe_gates[task](seq_output) / self.moe_gate_temperature
        return torch.softmax(logits, dim=-1)

    def task_representation(
        self,
        seq_output: torch.Tensor,
        task: str,
        expert_outputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expert_outputs = self._expert_outputs(seq_output) if expert_outputs is None else expert_outputs
        probabilities = self.gate_probabilities(seq_output, task)
        mixture = torch.einsum("be,beh->bh", probabilities, expert_outputs)
        if self.moe_residual:
            return seq_output + self.moe_residual_scale * mixture
        return mixture

    def task_representations_from_shared(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        expert_outputs = self._expert_outputs(seq_output)
        return {
            task: self.task_representation(seq_output, task, expert_outputs)
            for task in self.moe_task_order
        }

    def routing_probabilities_from_representation(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        return {task: self.gate_probabilities(seq_output, task) for task in self.moe_task_order}

    def ranking_logits_from_representation(self, seq_output: torch.Tensor) -> torch.Tensor:
        rank_output = self.task_representation(seq_output, "rank")
        return torch.einsum("bh,nh->bn", rank_output, self.item_embedding.weight)

    def auxiliary_logits_from_representation(self, seq_output: torch.Tensor) -> dict[str, torch.Tensor]:
        heads = self.auxiliary_heads()
        representations = self.task_representations_from_shared(seq_output)
        return {
            target: heads[target](representations[target]).squeeze(-1)
            for target in self.multitask_targets
        }

    def predict(self, interaction: Any) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        time_stamp = interaction["timestamp_list"]
        seq_output = self.forward(item_seq, item_seq_len, time_stamp)
        rank_output = self.task_representation(seq_output, "rank")
        test_item_emb = self.item_embedding(test_item)
        return torch.einsum("bh,bh->b", rank_output, test_item_emb)

    def full_sort_predict(self, interaction: Any) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamp = interaction["timestamp_list"]
        seq_output = self.forward(item_seq, item_seq_len, time_stamp)
        rank_output = self.task_representation(seq_output, "rank")
        return torch.einsum("bh,nh->bn", rank_output, self.item_embedding.weight)

    def extra_parameter_summary(self) -> dict[str, Any]:
        expert_params = sum(param.numel() for param in self.moe_experts.parameters())
        gate_params = sum(param.numel() for param in self.moe_gates.parameters())
        return {
            "moe_enabled": True,
            "num_experts": int(self.moe_num_experts),
            "expert_hidden_size": int(self.moe_expert_hidden_size),
            "expert_dropout": float(self.moe_dropout),
            "gate_temperature": float(self.moe_gate_temperature),
            "residual": bool(self.moe_residual),
            "residual_scale": float(self.moe_residual_scale),
            "activation": self.moe_activation,
            "task_order": list(self.moe_task_order),
            "expert_parameters": int(expert_params),
            "gate_parameters": int(gate_params),
            "total_moe_parameters": int(expert_params + gate_params),
        }

    def architecture_record(self) -> dict[str, Any]:
        return {
            "insertion_point": "after MultitaskTiM4Rec.shared_representation and before ranking/auxiliary heads",
            "experts_shared_between_tasks": True,
            "task_specific_gates": list(self.moe_task_order),
            **self.extra_parameter_summary(),
        }


def model_architecture_record(model: Any) -> dict[str, Any]:
    if hasattr(model, "architecture_record"):
        return model.architecture_record()
    return {
        "moe_enabled": False,
        "num_experts": 0,
        "task_order": list(TASK_ORDER),
        "insertion_point": None,
        "experts_shared_between_tasks": False,
        "task_specific_gates": [],
    }


def moe_config_from_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "moe_num_experts": int(config.get("num_experts", 0)),
        "moe_expert_hidden_size": int(config.get("expert_hidden_size", 64)),
        "moe_dropout": float(config.get("dropout", 0.0)),
        "moe_gate_temperature": float(config.get("gate_temperature", 1.0)),
        "moe_residual": bool(config.get("residual", True)),
        "moe_residual_scale": float(config.get("residual_scale", 1.0)),
        "moe_activation": str(config.get("activation", "gelu")),
    }
