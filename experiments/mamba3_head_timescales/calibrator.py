"""Reuse both Linear layers; only alpha is new, before the log1p input."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LearnedReference(nn.Module):
    def __init__(self, original, variant):
        super().__init__()
        if variant not in ('shared_tau', 'head_tau') or original.n_heads != 2:
            raise ValueError('Expected learned mode and two heads')
        self.variant = variant
        self.n_heads = original.n_heads
        self.max_log_scale = original.max_log_scale
        self.register_buffer('reference', original.reference)
        self.first, self.last = original.first, original.last
        self.alpha = nn.Parameter(torch.zeros(1 if variant == 'shared_tau' else 2,
                                              dtype=self.first.weight.dtype, device=self.first.weight.device))

    def log_reference_ratio(self):
        return math.log(4) * torch.tanh(self.alpha.double())

    def forward(self, gaps, active):
        if gaps.ndim != 2 or gaps.shape != active.shape or active.dtype != torch.bool:
            raise ValueError('Expected gaps and boolean active [B,L]')
        if gaps.device != active.device or gaps.device != self.reference.device:
            raise ValueError('Inputs and calibrator must share a device')
        if not torch.isfinite(gaps).all() or (gaps < 0).any():
            raise ValueError('Gaps must be finite and nonnegative')
        log_g = gaps.double().clamp_min(torch.finfo(torch.float64).tiny).log().unsqueeze(-1)
        log_ratio = log_g - self.reference.log() - self.log_reference_ratio()
        z = torch.logaddexp(log_ratio, torch.zeros_like(log_ratio))
        z = torch.where(gaps.unsqueeze(-1) > 0, z, 0.).to(self.first.weight.dtype)
        if self.variant == 'shared_tau':
            raw = self.last(F.silu(self.first(z)))
        else:
            # Each head sees only its scalar z and its own old output row.
            # Keeping the original Linear calls/shapes minimizes numerical drift.
            raw = torch.stack([self.last(F.silu(self.first(z[..., h:h+1])))[..., h]
                               for h in range(self.n_heads)], dim=-1)
        scale = torch.exp(self.max_log_scale * torch.tanh(raw))
        return torch.where(active.unsqueeze(-1), scale, torch.ones_like(scale))
