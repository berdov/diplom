"""Shared MOO task utilities.

The benchmark uses five minimization objectives:
ranking cross-entropy plus four behavior BCE objectives.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local syntax checks may not have torch.
    torch = None  # type: ignore[assignment]


AUX_TARGETS: tuple[str, ...] = ("is_click", "long_view", "is_like", "is_profile_enter")
TASK_ORDER: tuple[str, ...] = ("rank", *AUX_TARGETS)


def require_torch() -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for MOO benchmark execution.")
    return torch


def tensor_to_float(value: Any) -> float:
    th = require_torch()
    if isinstance(value, th.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def validate_task_order(task_order: Sequence[str]) -> tuple[str, ...]:
    task_order = tuple(task_order)
    if task_order != TASK_ORDER:
        raise ValueError(f"Expected task order {TASK_ORDER}, got {task_order}")
    return task_order


def preference_tensor(
    preference: Sequence[float] | Any,
    *,
    device: Any | None = None,
    dtype: Any | None = None,
    task_order: Sequence[str] = TASK_ORDER,
    eps: float = 1e-12,
) -> Any:
    th = require_torch()
    validate_task_order(task_order)
    pref = preference if isinstance(preference, th.Tensor) else th.tensor(preference, dtype=dtype or th.float32)
    pref = pref.to(device=device, dtype=dtype or th.float32)
    if pref.ndim != 1 or pref.numel() != len(task_order):
        raise ValueError(f"Preference must be a vector with {len(task_order)} entries, got shape {tuple(pref.shape)}")
    if not th.isfinite(pref).all():
        raise ValueError(f"Preference contains non-finite values: {pref}")
    if bool((pref < 0).any()):
        raise ValueError(f"Preference must be non-negative: {pref}")
    total = pref.sum()
    if tensor_to_float(total) <= eps:
        raise ValueError(f"Preference sum must be positive: {pref}")
    return pref / total.clamp_min(eps)


def losses_to_vector(
    losses: Mapping[str, Any],
    *,
    task_order: Sequence[str] = TASK_ORDER,
) -> Any:
    th = require_torch()
    validate_task_order(task_order)
    values = []
    for task in task_order:
        key = "rank" if task == "rank" else f"{task}_loss"
        if key not in losses:
            key = task
        if key not in losses:
            raise KeyError(f"Missing task loss {task}; available keys: {sorted(losses)}")
        value = losses[key]
        if not isinstance(value, th.Tensor):
            value = th.tensor(float(value), dtype=th.float32)
        values.append(value)
    return th.stack(values)


def normalize_loss_vector(loss_vector: Any, scales: Sequence[float] | Any | None, eps: float = 1e-8) -> Any:
    th = require_torch()
    vector = loss_vector if isinstance(loss_vector, th.Tensor) else th.tensor(loss_vector, dtype=th.float32)
    if scales is None:
        return vector
    scale_tensor = scales if isinstance(scales, th.Tensor) else th.tensor(scales, dtype=vector.dtype, device=vector.device)
    scale_tensor = scale_tensor.to(device=vector.device, dtype=vector.dtype).clamp_min(eps)
    if scale_tensor.shape != vector.shape:
        raise ValueError(f"Scale shape {tuple(scale_tensor.shape)} does not match loss shape {tuple(vector.shape)}")
    return vector / scale_tensor


def weighted_sum(loss_vector: Any, preference: Sequence[float] | Any) -> Any:
    pref = preference_tensor(preference, device=loss_vector.device, dtype=loss_vector.dtype)
    return (pref * loss_vector).sum()


def finite_scalar_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    return {key: tensor_to_float(value) for key, value in values.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def round_list(values: Iterable[float], digits: int = 8) -> list[float]:
    return [round(float(value), digits) for value in values]

