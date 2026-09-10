"""Vanilla Mamba-3 sequential recommender baseline for KuaiRand Protocol B.

Scientific role:
- primary task only: next-item recommendation;
- official Mamba-3 SISO block as the sequence mixer;
- no TiM4Rec time-aware mechanisms, prototypes, MoE, MTL, or MOO;
- full-sequence forward only (no incremental ``step`` path).

This is a baseline for the new architecture stage, not a proposed novel method.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender

try:
    from mamba_ssm import Mamba3
except Exception as exc:  # pragma: no cover - exercised on cluster if env is incomplete
    raise ImportError(
        "Mamba3Rec requires the official state-spaces/mamba source build with Mamba-3. "
        "See experiments/mamba3_baseline/ENVIRONMENT.md."
    ) from exc


_DTYPE_MAP: Final[dict[str, torch.dtype]] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported mamba3_dtype={name!r}; choose one of {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


class FeedForward(nn.Module):
    """Small residual FFN matching the outer TiM4Rec-style capacity."""

    def __init__(self, d_model: int, inner_size: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, inner_size)
        self.activation = nn.Hardswish()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class Mamba3ResidualBlock(nn.Module):
    """Pre-norm residual wrapper around the official Mamba-3 block."""

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        expand: int,
        headdim: int,
        ngroups: int,
        rope_fraction: float,
        chunk_size: int,
        is_mimo: bool,
        mimo_rank: int,
        is_outproj_norm: bool,
        mixer_dtype: torch.dtype,
        dropout: float,
        use_ffn: bool,
        ffn_inner_size: int,
        norm_eps: float,
        layer_idx: int,
    ) -> None:
        super().__init__()
        if (d_model * expand) % headdim != 0:
            raise ValueError(
                f"d_model*expand={d_model * expand} must be divisible by headdim={headdim}"
            )

        self.mixer_dtype = mixer_dtype
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.mixer = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            rope_fraction=rope_fraction,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            is_outproj_norm=is_outproj_norm,
            layer_idx=layer_idx,
            dtype=mixer_dtype,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
            self.ffn = FeedForward(d_model, ffn_inner_size, dropout)
            self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep embeddings/residual stream in fp32 while allowing the official
        # Mamba-3 mixer to use bf16/fp16 on GPU.
        mixed = self.mixer(self.norm1(x).to(dtype=self.mixer_dtype))
        x = x + self.dropout1(mixed.to(dtype=x.dtype))

        if self.use_ffn:
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class Mamba3Rec(SequentialRecommender):
    """Primary-only next-item recommender with a stacked Mamba-3 backbone."""

    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)

        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_layers"])
        self.dropout_prob = float(config["dropout_prob"])
        self.norm_eps = float(config["norm_eps"])

        self.d_state = int(config["mamba3_d_state"])
        self.expand = int(config["mamba3_expand"])
        self.headdim = int(config["mamba3_headdim"])
        self.ngroups = int(config["mamba3_ngroups"])
        self.rope_fraction = float(config["mamba3_rope_fraction"])
        self.chunk_size = int(config["mamba3_chunk_size"])
        self.is_mimo = bool(config["mamba3_is_mimo"])
        self.mimo_rank = int(config["mamba3_mimo_rank"])
        self.is_outproj_norm = bool(config["mamba3_is_outproj_norm"])
        self.mixer_dtype = _resolve_dtype(config["mamba3_dtype"])

        self.use_ffn = bool(config["use_ffn"])
        self.ffn_inner_size = int(config["ffn_inner_size"])

        self.item_embedding = nn.Embedding(
            self.n_items,
            self.hidden_size,
            padding_idx=0,
        )
        self.input_dropout = nn.Dropout(self.dropout_prob)
        self.input_norm = nn.LayerNorm(self.hidden_size, eps=self.norm_eps)

        self.layers = nn.ModuleList(
            [
                Mamba3ResidualBlock(
                    d_model=self.hidden_size,
                    d_state=self.d_state,
                    expand=self.expand,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    rope_fraction=self.rope_fraction,
                    chunk_size=self.chunk_size,
                    is_mimo=self.is_mimo,
                    mimo_rank=self.mimo_rank,
                    is_outproj_norm=self.is_outproj_norm,
                    mixer_dtype=self.mixer_dtype,
                    dropout=self.dropout_prob,
                    use_ffn=self.use_ffn,
                    ffn_inner_size=self.ffn_inner_size,
                    norm_eps=self.norm_eps,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(self.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_size, eps=self.norm_eps)
        self.loss_fct = nn.CrossEntropyLoss()

        # Do not recursively reinitialize Mamba-3: it has architecture-specific
        # initializers for dt/A/B/C. Only initialize the recommender-owned layers.
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        self._init_owned_layers()

    def _init_owned_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, FeedForward):
                for child in module.modules():
                    if isinstance(child, nn.Linear):
                        nn.init.normal_(child.weight, mean=0.0, std=0.02)
                        if child.bias is not None:
                            nn.init.zeros_(child.bias)

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        hidden = self.item_embedding(item_seq)
        hidden = self.input_norm(self.input_dropout(hidden))

        for layer in self.layers:
            hidden = layer(hidden)

        hidden = self.output_norm(hidden)
        return self.gather_indexes(hidden, item_seq_len - 1)

    def calculate_loss(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        logits = torch.einsum("bh,nh->bn", seq_output, self.item_embedding.weight)
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        return torch.einsum("bh,bh->b", seq_output, test_item_emb)

    def full_sort_predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(item_seq, item_seq_len)
        return torch.einsum("bh,nh->bn", seq_output, self.item_embedding.weight)
