"""Shared gradient diagnostics for adaptive MultitaskTiM4Rec smoke tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


AUX_TARGETS: tuple[str, ...] = ("is_click", "long_view", "is_like", "is_profile_enter")
TASK_ORDER: tuple[str, ...] = ("rank", *AUX_TARGETS)
HEAD_TOKENS: tuple[str, ...] = ("click_head", "long_view_head", "like_head", "profile_enter_head")


@dataclass(frozen=True)
class ParameterEntry:
    """A named parameter that participates in a shared-gradient calculation."""

    name: str
    parameter: torch.nn.Parameter


def task_loss_key(task: str) -> str:
    return "rank" if task == "rank" else f"{task}_loss"


def tensor_to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def head_parameter_ids(model: Any) -> set[int]:
    ids: set[int] = set()
    for head in model.auxiliary_heads().values():
        for param in head.parameters():
            ids.add(id(param))
    return ids


def shared_parameter_entries(model: Any, selector: str = "all_backbone") -> list[ParameterEntry]:
    """Return trainable TiM4Rec shared parameters, excluding task-specific heads."""

    head_ids = head_parameter_ids(model)
    entries = [
        ParameterEntry(name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in head_ids and not any(token in name for token in HEAD_TOKENS)
    ]
    if selector == "all_backbone":
        selected = entries
    elif selector == "last_shared_block":
        num_layers = int(getattr(model, "num_layers", 0))
        prefix = f"ssd_layers.{max(num_layers - 1, 0)}."
        selected = [entry for entry in entries if entry.name.startswith(prefix)]
        if not selected:
            selected = entries
    else:
        raise ValueError(f"Unknown shared parameter selector: {selector}")
    if not selected:
        raise RuntimeError(f"No shared parameters selected for selector={selector}")
    return selected


def parameter_group_summary(entries: Iterable[ParameterEntry]) -> dict[str, Any]:
    names = [entry.name for entry in entries]
    total = sum(entry.parameter.numel() for entry in entries)
    by_prefix: dict[str, int] = {}
    for entry in entries:
        prefix = entry.name.split(".", 1)[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + entry.parameter.numel()
    return {
        "parameter_tensors": len(names),
        "parameter_count": int(total),
        "first_names": names[:8],
        "last_names": names[-8:],
        "parameter_count_by_top_module": {key: int(value) for key, value in sorted(by_prefix.items())},
    }


def autograd_grads(
    loss: torch.Tensor,
    entries: list[ParameterEntry],
    *,
    retain_graph: bool,
    create_graph: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    params = [entry.parameter for entry in entries]
    return torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )


def flatten_grads(
    grads: Iterable[torch.Tensor | None],
    entries: list[ParameterEntry],
    *,
    detach: bool = True,
) -> torch.Tensor:
    chunks = []
    for grad, entry in zip(grads, entries):
        if grad is None:
            chunk = entry.parameter.new_zeros(entry.parameter.numel())
        else:
            chunk = grad.reshape(-1)
        chunks.append(chunk.detach() if detach else chunk)
    if not chunks:
        raise RuntimeError("Cannot flatten empty gradient list.")
    return torch.cat(chunks)


def grad_vector(
    loss: torch.Tensor,
    entries: list[ParameterEntry],
    *,
    retain_graph: bool,
    create_graph: bool = False,
    detach: bool = True,
) -> torch.Tensor:
    grads = autograd_grads(loss, entries, retain_graph=retain_graph, create_graph=create_graph)
    return flatten_grads(grads, entries, detach=detach)


def task_gradient_vectors(
    task_losses: dict[str, torch.Tensor],
    entries: list[ParameterEntry],
    task_order: Iterable[str] = TASK_ORDER,
) -> dict[str, torch.Tensor]:
    vectors: dict[str, torch.Tensor] = {}
    ordered = list(task_order)
    for index, task in enumerate(ordered):
        vectors[task] = grad_vector(
            task_losses[task],
            entries,
            retain_graph=index < len(ordered) - 1,
            detach=True,
        )
    return vectors


def vector_norm(value: torch.Tensor, eps: float = 1e-12) -> float:
    norm = float(torch.linalg.vector_norm(value.detach().float()).cpu().item())
    return 0.0 if norm < eps else norm


def cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float | None:
    left_norm = torch.linalg.vector_norm(left.detach().float())
    right_norm = torch.linalg.vector_norm(right.detach().float())
    denom = left_norm * right_norm
    if float(denom.cpu().item()) <= eps:
        return None
    value = torch.dot(left.detach().float(), right.detach().float()) / denom
    if not torch.isfinite(value):
        return None
    return float(value.cpu().item())


def cosine_matrix(vectors: dict[str, torch.Tensor], task_order: Iterable[str] = TASK_ORDER) -> dict[str, dict[str, float | None]]:
    order = list(task_order)
    return {left: {right: cosine(vectors[left], vectors[right]) for right in order} for left in order}


def gradient_norms(vectors: dict[str, torch.Tensor], task_order: Iterable[str] = TASK_ORDER) -> dict[str, float]:
    return {task: vector_norm(vectors[task]) for task in task_order}


def conflict_summary(matrix: dict[str, dict[str, float | None]], task_order: Iterable[str] = TASK_ORDER) -> dict[str, Any]:
    order = list(task_order)
    total = 0
    negative = 0
    negative_pairs: list[dict[str, Any]] = []
    for i, left in enumerate(order):
        for right in order[i + 1 :]:
            value = matrix[left][right]
            if value is None:
                continue
            total += 1
            if value < 0:
                negative += 1
                negative_pairs.append({"left": left, "right": right, "cosine": value})
    return {
        "pairs": total,
        "negative_pairs": negative,
        "fraction_conflicting": negative / total if total else 0.0,
        "negative_pairs_detail": negative_pairs,
    }


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def finite_named_scalars(values: dict[str, torch.Tensor]) -> dict[str, Any]:
    nonfinite = [name for name, value in values.items() if not finite_tensor(value)]
    return {"all_finite": not nonfinite, "nonfinite": nonfinite}


def assign_flat_gradient(entries: list[ParameterEntry], vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for entry in entries:
            param = entry.parameter
            size = param.numel()
            grad_piece = vector[offset : offset + size].view_as(param).to(param.device)
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.copy_(grad_piece)
            offset += size
    if offset != vector.numel():
        raise RuntimeError(f"Gradient vector size mismatch: used {offset}, vector has {vector.numel()}")


def assign_gradient_tensors(entries: list[ParameterEntry], grads: list[torch.Tensor]) -> None:
    if len(entries) != len(grads):
        raise RuntimeError(f"Expected {len(entries)} gradients, got {len(grads)}")
    with torch.no_grad():
        for entry, grad in zip(entries, grads):
            if entry.parameter.grad is None:
                entry.parameter.grad = torch.zeros_like(entry.parameter)
            entry.parameter.grad.copy_(grad.to(entry.parameter.device))


def max_cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0, "max_allocated_bytes": 0, "max_reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def ensure_finite_gradients(entries: Iterable[ParameterEntry]) -> dict[str, Any]:
    checked = 0
    nonfinite: list[str] = []
    for entry in entries:
        grad = entry.parameter.grad
        if grad is None:
            continue
        checked += 1
        if not torch.isfinite(grad).all().item():
            nonfinite.append(entry.name)
    return {
        "checked_tensors": checked,
        "nonfinite_tensor_count": len(nonfinite),
        "nonfinite_tensors": nonfinite[:10],
        "all_finite": len(nonfinite) == 0,
    }


def safe_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)
