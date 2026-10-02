"""Rational centered gap feature, with a floating-point upper-endpoint guard."""
import torch
from ..modulation import GapTrap as ParentGapTrap, attach_projection
from experiments.mamba3_three_time.calibrators import REFERENCE_MS


def centered_q(gaps, active, dtype=torch.float32):
    if gaps.ndim != 2 or gaps.shape != active.shape or active.dtype != torch.bool or gaps.device != active.device:
        raise ValueError('Expected gaps and boolean active [B,L] on one device')
    if not torch.isfinite(gaps).all() or (gaps < 0).any():
        raise ValueError('Finite nonnegative history gaps required')
    g = gaps.double()
    q = (g - REFERENCE_MS) / (g + REFERENCE_MS)
    # Finite huge gaps can round to +1 in fp64 and again on the fp32 cast.
    # Retain the analytic open upper endpoint without a tuned epsilon.
    one = torch.ones((), device=g.device, dtype=torch.float64)
    q = q.clamp_max(torch.nextafter(one, torch.zeros_like(one))).to(dtype)
    one = one.to(dtype)
    q = q.clamp_max(torch.nextafter(one, torch.zeros_like(one)))
    return torch.where(active, q, 0.)


class GapTrap(ParentGapTrap):
    def forward(self, gaps, active):
        return self.alpha.clamp(0, 1) * centered_q(gaps, active, self.alpha.dtype)
