"""COSMOS-style direct preference-conditioned TiM4Rec."""

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


class COSMOSTiM4Rec(MultitaskTiM4Rec):
    """Direct preference-conditioned recommender, not a hypernetwork."""

    def __init__(
        self,
        config: Any,
        dataset: Any,
        *,
        preference_hidden_size: int = 64,
        conditioning_scale: float = 0.1,
    ):
        if torch is None or nn is None:  # pragma: no cover
            require_torch()
        super().__init__(config, dataset)
        self.conditioning_scale = float(conditioning_scale)
        self.preference_encoder = nn.Sequential(
            nn.Linear(len(TASK_ORDER), int(preference_hidden_size)),
            nn.ReLU(),
        )
        self.preference_fusion = nn.Linear(self.hidden_size + int(preference_hidden_size), self.hidden_size)
        nn.init.zeros_(self.preference_fusion.weight)
        nn.init.zeros_(self.preference_fusion.bias)
        self.register_buffer("current_preference", torch.ones(len(TASK_ORDER)) / len(TASK_ORDER))

    def set_preference(self, preference: Sequence[float] | Any) -> None:
        pref = preference_tensor(preference, device=self.current_preference.device, dtype=self.current_preference.dtype)
        self.current_preference.copy_(pref)

    def preference(self, reference: Any | None = None) -> Any:
        if reference is None:
            return self.current_preference
        return self.current_preference.to(device=reference.device, dtype=reference.dtype)

    def shared_representation(self, interaction: Any, preference: Sequence[float] | Any | None = None) -> Any:
        base = super().shared_representation(interaction)
        pref = self.preference(base) if preference is None else preference_tensor(
            preference,
            device=base.device,
            dtype=base.dtype,
        )
        pref_embedding = self.preference_encoder(pref).expand(base.shape[0], -1)
        delta = torch.tanh(self.preference_fusion(torch.cat([base, pref_embedding], dim=-1)))
        return base + self.conditioning_scale * delta

    def predict(self, interaction: Any) -> Any:
        seq_output = self.shared_representation(interaction)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        return torch.einsum("bh,bh->b", seq_output, test_item_emb)

    def full_sort_predict(self, interaction: Any) -> Any:
        seq_output = self.shared_representation(interaction)
        return self.ranking_logits_from_representation(seq_output)

    def cosmos_regularized_loss(self, loss_vector: Any, preference: Sequence[float] | Any, lambda_cosine: float) -> Any:
        pref = preference_tensor(preference, device=loss_vector.device, dtype=loss_vector.dtype)
        weighted = (pref * loss_vector).sum()
        cosine = torch.nn.functional.cosine_similarity(loss_vector.unsqueeze(0), pref.unsqueeze(0), dim=-1).mean()
        return weighted - float(lambda_cosine) * cosine

    def extra_parameter_summary(self) -> dict[str, int]:
        params = sum(param.numel() for param in self.preference_encoder.parameters())
        params += sum(param.numel() for param in self.preference_fusion.parameters())
        return {"cosmos_conditioning_parameters": int(params)}
