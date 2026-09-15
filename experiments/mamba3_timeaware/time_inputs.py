"""Causal temporal inputs, independent of RecBole and CUDA kernels."""

import math

import torch
from torch import nn


def history_gaps(timestamps, valid_mask):
    """Adjacent history gaps; negative gaps clamp to zero, padding is neutral."""
    if timestamps.ndim != 2 or timestamps.shape != valid_mask.shape:
        raise ValueError("Expected timestamps and valid_mask [B,L]")
    if valid_mask.dtype != torch.bool or timestamps.device != valid_mask.device:
        raise ValueError("Expected boolean mask on the timestamp device")
    if not torch.isfinite(timestamps[valid_mask]).all():
        raise ValueError("Nonfinite history timestamp")
    times = torch.where(valid_mask, timestamps.double(), 0.0)
    active = torch.zeros_like(valid_mask)
    active[:, 1:] = valid_mask[:, 1:] & valid_mask[:, :-1]
    gaps = torch.zeros_like(times)
    gaps[:, 1:] = (times[:, 1:] - times[:, :-1]).clamp_min(0)
    gaps = torch.where(active, gaps, 0.0)
    if not torch.isfinite(gaps).all():
        raise ValueError("History gap overflow")
    return gaps, active


class TimeCalibrator(nn.Module):
    def __init__(self, n_heads, time_scale_reference, max_log_scale=math.log(2)):
        super().__init__()
        if n_heads < 1 or int(n_heads) != n_heads:
            raise ValueError("n_heads must be a positive integer")
        if time_scale_reference is None or not math.isfinite(time_scale_reference) or time_scale_reference <= 0:
            raise ValueError("Explicit positive TRAIN-only time_scale_reference required")
        if not math.isfinite(max_log_scale) or not 0 < max_log_scale <= 20:
            raise ValueError("max_log_scale must be in (0,20]")
        self.n_heads = int(n_heads)
        self.register_buffer("reference", torch.tensor(time_scale_reference, dtype=torch.float64))
        self.max_log_scale = float(max_log_scale)
        self.first = nn.Linear(1, 16)
        self.last = nn.Linear(16, self.n_heads)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, gaps, active):
        if gaps.ndim != 2 or gaps.shape != active.shape or active.dtype != torch.bool:
            raise ValueError("Expected gaps and boolean active [B,L]")
        if not torch.isfinite(gaps).all() or (gaps < 0).any():
            raise ValueError("Gaps must be finite and nonnegative")
        # Equivalent to log1p(gap/reference), without overflowing the ratio.
        log_ratio = gaps.double().clamp_min(torch.finfo(torch.float64).tiny).log() - self.reference.log()
        tau = torch.logaddexp(log_ratio, torch.zeros_like(log_ratio))
        tau = torch.where(gaps > 0, tau, 0.0).unsqueeze(-1).to(self.first.weight.dtype)
        raw = self.last(torch.nn.functional.silu(self.first(tau)))
        scale = torch.exp(self.max_log_scale * torch.tanh(raw))
        return torch.where(active.unsqueeze(-1), scale, torch.ones_like(scale))


def condition_dt(dt_base, a, time_scale):
    if dt_base.ndim != 3 or dt_base.shape != a.shape or dt_base.shape != time_scale.shape:
        raise ValueError("DT, A and time_scale must have identical [B,L,H] shapes")
    if time_scale.device != dt_base.device:
        raise ValueError("time_scale must be on the DT device")
    if not torch.isfinite(time_scale).all() or (time_scale <= 0).any():
        raise ValueError("time_scale must be finite and positive")
    dt_real = dt_base * time_scale.to(dt_base.dtype)
    return dt_real, a * dt_real
