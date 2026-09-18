"""Durable, owned evidence with finite numerical diagnostics."""

import hashlib
import json
import os
from pathlib import Path
import torch


def create_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def update_record(path, value):
    """Only for a file previously reserved by this process, never old evidence."""
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    create_record(tmp, value)
    os.replace(tmp, path)


def tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_records(state):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                     sha256=hashlib.sha256(tensor_bytes(value)).hexdigest())
            for key, value in sorted(state.items())}


def state_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def tensor_difference(left, right):
    rows = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            rows.append(dict(key=key, missing="left" if a is None else "right"))
        elif a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            rows.append(dict(key=key, left_shape=list(a.shape), right_shape=list(b.shape),
                left_dtype=str(a.dtype), right_dtype=str(b.dtype),
                max_abs_diff=(a.double()-b.double()).abs().max().item() if a.shape == b.shape else None))
    return rows


def compare(a, b, atol=1e-6, rtol=1e-5, relative_norm_limit=None):
    if a is None or b is None:
        return dict(passed=False, finite=False, reason="missing tensor/gradient")
    if a.shape != b.shape:
        return dict(passed=False, finite=False, reason="shape mismatch",
                    left_shape=list(a.shape), right_shape=list(b.shape))
    a, b = a.detach().double(), b.detach().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(passed=False, finite=False, reason="NaN/Inf", max_abs=None, mean_abs=None, relative_norm=None)
    delta = a-b
    relative = delta.norm().item() / max(b.norm().item(), 1e-12)
    passed = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    if relative_norm_limit is not None:
        passed = passed and relative <= relative_norm_limit
    return dict(passed=passed, finite=True, max_abs=delta.abs().max().item(),
        mean_abs=delta.abs().mean().item(), relative_norm=relative,
        atol=atol, rtol=rtol, relative_norm_limit=relative_norm_limit,
        elements=a.numel(), left_dtype="float64 diagnostic conversion")


def case(name, oracle, checks, *, dtype="bf16 mixer / fp32 outer and DT"):
    return dict(name=name, oracle=oracle, dtype=dtype, checks=checks,
        checked_tensors=len(checks), failed_keys=[k for k, v in checks.items() if not v["passed"]],
        passed=bool(checks) and all(v["passed"] for v in checks.values()))
