"""PaLoRA low-rank preference combination for TiM4Rec modules."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.moo_8families.strategies.base import TASK_ORDER, preference_tensor, require_torch  # noqa: E402

try:  # noqa: E402
    from experiments.multitask_tim4rec.model import MultitaskTiM4Rec
except ModuleNotFoundError:  # pragma: no cover - synthetic unit tests can run without RecBole.
    MultitaskTiM4Rec = object  # type: ignore[assignment]


class PaLoRALinear(nn.Module):
    """Linear layer with preference-weighted per-task low-rank adapters."""

    def __init__(self, base: Any, *, task_count: int, rank: int = 1, alpha: float = 1.0):
        if torch is None or nn is None or F is None:  # pragma: no cover
            require_torch()
        super().__init__()
        if int(rank) <= 0:
            raise ValueError(f"PaLoRA rank must be positive, got {rank}")
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.task_count = int(task_count)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.weight = nn.Parameter(base.weight.detach().clone())
        self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone())
        self.lora_A = nn.Parameter(torch.empty(self.task_count, self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.task_count, self.out_features, self.rank))
        self.reset_lora_parameters()
        self.register_buffer("current_preference", torch.ones(self.task_count) / self.task_count)

    def reset_lora_parameters(self) -> None:
        for task_idx in range(self.task_count):
            nn.init.kaiming_uniform_(self.lora_A[task_idx], a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_preference(self, preference: Sequence[float] | Any) -> None:
        pref = preference_tensor(preference, device=self.current_preference.device, dtype=self.current_preference.dtype)
        self.current_preference.copy_(pref)

    def forward(self, input_tensor: Any) -> Any:
        base = F.linear(input_tensor, self.weight, self.bias)
        pref = self.current_preference.to(device=input_tensor.device, dtype=input_tensor.dtype)
        delta = torch.zeros_like(base)
        for task_idx in range(self.task_count):
            low_rank = F.linear(F.linear(input_tensor, self.lora_A[task_idx]), self.lora_B[task_idx])
            delta = delta + pref[task_idx] * low_rank
        return base + self.scaling * delta

    def extra_parameters(self) -> int:
        return int(self.lora_A.numel() + self.lora_B.numel())


def _set_submodule(root: Any, path: str, module: Any) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def _get_submodule(root: Any, path: str) -> Any:
    module = root
    for part in path.split("."):
        module = getattr(module, part) if not part.isdigit() else module[int(part)]
    return module


class PaLoRATiM4Rec(MultitaskTiM4Rec):
    """TiM4Rec with PaLoRA modules in selected late backbone projections."""

    def __init__(
        self,
        config: Any,
        dataset: Any,
        *,
        rank: int = 1,
        alpha: float = 1.0,
        target_modules: Sequence[str] | None = None,
    ):
        if torch is None or nn is None:  # pragma: no cover
            require_torch()
        super().__init__(config, dataset)
        symbolic = list(target_modules or ["last_ssd_out_proj", "last_ffn_fc2"])
        concrete = self.resolve_target_modules(symbolic)
        self.palora_module_names: list[str] = []
        for name in concrete:
            base = _get_submodule(self, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"PaLoRA target must be nn.Linear, got {type(base).__name__} at {name}")
            replacement = PaLoRALinear(base, task_count=len(TASK_ORDER), rank=rank, alpha=alpha)
            _set_submodule(self, name, replacement)
            self.palora_module_names.append(name)
        self.register_buffer("current_preference", torch.ones(len(TASK_ORDER)) / len(TASK_ORDER))

    def resolve_target_modules(self, names: Sequence[str]) -> list[str]:
        result = []
        last = int(self.num_layers) - 1
        for name in names:
            if name == "last_ssd_out_proj":
                result.append(f"ssd_layers.{last}.ssd.out_proj")
            elif name == "last_ffn_fc2":
                if not bool(self.is_ffn):
                    continue
                result.append(f"ssd_layers.{last}.ffn.fc2")
            else:
                result.append(str(name))
        return result

    def set_preference(self, preference: Sequence[float] | Any) -> None:
        pref = preference_tensor(preference, device=self.current_preference.device, dtype=self.current_preference.dtype)
        self.current_preference.copy_(pref)
        for module in self.modules():
            if isinstance(module, PaLoRALinear):
                module.set_preference(pref)

    def predict(self, interaction: Any) -> Any:
        seq_output = self.shared_representation(interaction)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        return torch.einsum("bh,bh->b", seq_output, test_item_emb)

    def full_sort_predict(self, interaction: Any) -> Any:
        seq_output = self.shared_representation(interaction)
        return self.ranking_logits_from_representation(seq_output)

    def extra_parameter_summary(self) -> dict[str, Any]:
        modules = []
        total_extra = 0
        for name in self.palora_module_names:
            module = _get_submodule(self, name)
            extra = module.extra_parameters()
            total_extra += extra
            modules.append(
                {
                    "name": name,
                    "rank": module.rank,
                    "alpha": module.alpha,
                    "extra_parameters": extra,
                    "base_shape": [module.out_features, module.in_features],
                }
            )
        return {
            "palora_modules": modules,
            "palora_extra_parameters": int(total_extra),
            "formula": "W + alpha/r * sum_t lambda_t B_t A_t",
        }
