#!/usr/bin/env python
"""Multitask TiM4Rec model with behavior heads.

The backbone is the validated upstream TiM4Rec implementation used by
`experiments/tim4rec_baseline`. The only architectural change is four linear
auxiliary heads over the same next-interaction representation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tim4rec import TiM4Rec  # noqa: E402


TARGETS: tuple[str, ...] = ("is_click", "long_view", "is_like", "is_profile_enter")


class MultitaskTiM4Rec(TiM4Rec):
    """TiM4Rec backbone plus four same-row behavior prediction heads."""

    input_fields_used: tuple[str, ...] = ("item_id_list", "item_length", "timestamp_list")

    def __init__(self, config: Any, dataset: Any):
        super().__init__(config, dataset)
        try:
            configured_targets = config["multitask_targets"]
        except Exception:
            configured_targets = TARGETS
        self.multitask_targets = tuple(configured_targets)
        unexpected = sorted(set(self.multitask_targets).difference(TARGETS))
        if unexpected:
            raise ValueError(f"Unsupported multitask targets: {unexpected}")

        self.click_head = nn.Linear(self.hidden_size, 1)
        self.long_view_head = nn.Linear(self.hidden_size, 1)
        self.like_head = nn.Linear(self.hidden_size, 1)
        self.profile_enter_head = nn.Linear(self.hidden_size, 1)
        for head in self.auxiliary_heads().values():
            self._init_weights(head)

    def auxiliary_heads(self) -> dict[str, nn.Linear]:
        return {
            "is_click": self.click_head,
            "long_view": self.long_view_head,
            "is_like": self.like_head,
            "is_profile_enter": self.profile_enter_head,
        }

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor, time_stamp: torch.Tensor) -> torch.Tensor:
        if item_seq_len.dtype != torch.long:
            item_seq_len = item_seq_len.long()
        return super().forward(item_seq, item_seq_len, time_stamp)

    def shared_representation(self, interaction: Any) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamp = interaction["timestamp_list"]
        return self.forward(item_seq, item_seq_len, time_stamp)

    def ranking_logits_from_representation(self, seq_output: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bh,nh->bn", seq_output, self.item_embedding.weight)

    def auxiliary_logits_from_representation(
        self, seq_output: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        heads = self.auxiliary_heads()
        return {target: heads[target](seq_output).squeeze(-1) for target in self.multitask_targets}

    def auxiliary_logits(self, interaction: Any) -> dict[str, torch.Tensor]:
        return self.auxiliary_logits_from_representation(self.shared_representation(interaction))

    def calculate_multitask_loss(
        self,
        interaction: Any,
        *,
        lambda_aux: float,
        pos_weights: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        seq_output = self.shared_representation(interaction)
        pos_items = interaction[self.POS_ITEM_ID]
        rank_logits = self.ranking_logits_from_representation(seq_output)
        rank_loss = self.loss_fct(rank_logits, pos_items)

        aux_logits = self.auxiliary_logits_from_representation(seq_output)
        aux_losses: dict[str, torch.Tensor] = {}
        for target, logits in aux_logits.items():
            labels = interaction[target].float()
            weight = None if pos_weights is None else pos_weights[target].to(logits.device)
            aux_losses[target] = nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=weight,
            )

        aux_sum = sum(aux_losses.values())
        total = rank_loss + float(lambda_aux) * aux_sum
        return {
            "total": total,
            "rank": rank_loss,
            "aux_sum": aux_sum,
            **{f"{target}_loss": loss for target, loss in aux_losses.items()},
        }

    def calculate_loss(self, interaction: Any) -> torch.Tensor:
        lambda_aux = float(getattr(self, "lambda_aux", 0.0))
        pos_weights = getattr(self, "pos_weights", None)
        return self.calculate_multitask_loss(
            interaction,
            lambda_aux=lambda_aux,
            pos_weights=pos_weights,
        )["total"]
