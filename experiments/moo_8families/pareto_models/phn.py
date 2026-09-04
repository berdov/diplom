"""PHN-style preference hypernetwork adapter for MultitaskTiM4Rec."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.moo_8families.strategies.base import TASK_ORDER, preference_tensor, require_torch  # noqa: E402

try:  # noqa: E402
    from experiments.multitask_tim4rec.model import MultitaskTiM4Rec
except ModuleNotFoundError:  # pragma: no cover - synthetic unit tests can run without RecBole.
    MultitaskTiM4Rec = object  # type: ignore[assignment]


class PHNAdapterTiM4Rec(MultitaskTiM4Rec):
    """TiM4Rec with a hypernetwork-generated compact representation adapter.

    This is intentionally named adapter: full PHN over all TiM4Rec parameters is
    not hidden behind this class. The hypernetwork generates per-preference FiLM
    parameters for the shared representation used by ranking and auxiliary heads.
    """

    def __init__(
        self,
        config: Any,
        dataset: Any,
        *,
        adapter_hidden_size: int = 64,
        adapter_scale: float = 0.1,
    ):
        if torch is None or nn is None:  # pragma: no cover
            require_torch()
        super().__init__(config, dataset)
        self.adapter_scale = float(adapter_scale)
        self.preference_hypernetwork = nn.Sequential(
            nn.Linear(len(TASK_ORDER), int(adapter_hidden_size)),
            nn.ReLU(),
            nn.Linear(int(adapter_hidden_size), 2 * self.hidden_size),
        )
        nn.init.zeros_(self.preference_hypernetwork[-1].weight)
        nn.init.zeros_(self.preference_hypernetwork[-1].bias)
        self.register_buffer("current_preference", torch.ones(len(TASK_ORDER)) / len(TASK_ORDER))

    def set_preference(self, preference: Sequence[float] | Any) -> None:
        pref = preference_tensor(preference, device=self.current_preference.device, dtype=self.current_preference.dtype)
        self.current_preference.copy_(pref)

    def preference(self, reference: Any | None = None) -> Any:
        if reference is None:
            return self.current_preference
        return self.current_preference.to(device=reference.device, dtype=reference.dtype)

    def conditioned_representation(self, seq_output: Any, preference: Sequence[float] | Any | None = None) -> Any:
        pref = self.preference(seq_output) if preference is None else preference_tensor(
            preference,
            device=seq_output.device,
            dtype=seq_output.dtype,
        )
        generated = self.preference_hypernetwork(pref).view(2, self.hidden_size)
        gamma, beta = generated[0], generated[1]
        return seq_output * (1.0 + self.adapter_scale * torch.tanh(gamma)) + self.adapter_scale * beta

    def shared_representation(self, interaction: Any, preference: Sequence[float] | Any | None = None) -> Any:
        base = super().shared_representation(interaction)
        return self.conditioned_representation(base, preference)

    def predict(self, interaction: Any) -> Any:
        seq_output = self.shared_representation(interaction)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        return torch.einsum("bh,bh->b", seq_output, test_item_emb)

    def full_sort_predict(self, interaction: Any) -> Any:
        seq_output = self.shared_representation(interaction)
        return self.ranking_logits_from_representation(seq_output)

    def extra_parameter_summary(self) -> dict[str, int]:
        params = sum(param.numel() for param in self.preference_hypernetwork.parameters())
        return {"phn_adapter_parameters": int(params)}
