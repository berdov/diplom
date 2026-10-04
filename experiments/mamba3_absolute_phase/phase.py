"""Periodic observed-event clocks; no target time or additional random draws."""

import math

import torch
from torch import nn

from experiments.mamba3_timeaware.time_inputs import history_gaps


MODES = ("baseline_dual", "relative_phase", "absolute_phase")
PERIODS_MS = (21600000, 86400000)
K = len(PERIODS_MS)


def phase_clock(timestamps, valid, mode, *, gaps=None):
    """Return sanitized FP64 clocks; relative time restarts at this window."""
    if mode not in MODES[1:]:
        raise ValueError("A phase clock requires relative_phase or absolute_phase")
    if (timestamps.ndim != 2 or timestamps.shape != valid.shape
            or valid.dtype != torch.bool or timestamps.device != valid.device):
        raise ValueError("Expected timestamps and boolean valid mask [B,L]")
    if not torch.isfinite(timestamps[valid]).all():
        raise ValueError("Nonfinite history timestamp")
    # Sanitize before remainder or trigonometry, not only their final output.
    clean = torch.where(valid, timestamps.double(), 0.0)
    if mode == "absolute_phase":
        return clean
    if gaps is None:
        gaps, _ = history_gaps(timestamps, valid)
    if (gaps.shape != timestamps.shape or gaps.device != timestamps.device
            or not torch.isfinite(gaps).all() or (gaps < 0).any()):
        raise ValueError("Finite nonnegative inherited history gaps required")
    elapsed = gaps.double().cumsum(1)
    if not torch.isfinite(elapsed[valid]).all():
        raise ValueError("Relative phase clock overflow")
    return torch.where(valid, elapsed, 0.0)


def features_from_clock(clock, *, dtype=torch.float32):
    """[sin6h, cos6h, sin24h, cos24h]/sqrt(2), FP64 until final cast."""
    if not torch.isfinite(clock).all():
        raise ValueError("Sanitized finite clock required")
    clock = clock.double()
    periods = clock.new_tensor(PERIODS_MS)
    reduced = torch.remainder(clock.unsqueeze(-1), periods)
    theta = reduced * (2.0 * math.pi) / periods
    return torch.stack((theta.sin(), theta.cos()), -1).flatten(-2).div(math.sqrt(K)).to(dtype)


def periodic_features(timestamps, valid, mode, *, gaps=None, dtype=torch.float32):
    return features_from_clock(phase_clock(timestamps, valid, mode, gaps=gaps), dtype=dtype)


def phase_correction(features, weight, active):
    """Bounded raw-angle correction, shared by heads and layers."""
    if (features.ndim != 3 or features.shape[-1] != 2*K or weight.ndim != 2
            or weight.shape[-1] != 2*K or features.shape[:2] != active.shape
            or active.dtype != torch.bool):
        raise ValueError("Expected features [B,L,4], W [A,4], active [B,L]")
    # Linear has no bias and consumes no RNG; W is the only new parameter.
    delta = torch.nn.functional.linear(features.to(weight.dtype), weight).tanh()
    return torch.where(active.unsqueeze(-1), delta, torch.zeros_like(delta))


class PeriodicPhase(nn.Module):
    def __init__(self, num_angles, mode):
        super().__init__()
        if mode not in MODES[1:] or int(num_angles) != num_angles or num_angles < 1:
            raise ValueError("Positive angle count and modified phase mode required")
        self.mode = mode
        self.W = nn.Parameter(torch.zeros(int(num_angles), 2*K, dtype=torch.float32))

    def components(self, timestamps, valid, *, gaps=None, active=None):
        if gaps is None or active is None:
            inherited_gaps, inherited_active = history_gaps(timestamps, valid)
            gaps = inherited_gaps if gaps is None else gaps
            active = inherited_active if active is None else active
        expected_active = torch.zeros_like(valid)
        expected_active[:, 1:] = valid[:, 1:] & valid[:, :-1]
        if not torch.equal(active, expected_active):
            raise ValueError("Phase active mask must preserve real zero-gap events")
        phi = periodic_features(timestamps, valid, self.mode, gaps=gaps, dtype=self.W.dtype)
        return phase_correction(phi, self.W, active), phi, active

    def forward(self, timestamps, valid, *, gaps=None, active=None):
        return self.components(timestamps, valid, gaps=gaps, active=active)[0]


def reference_correction(timestamps, valid, mode, weight):
    """Independent scalar FP64 oracle for the new operation (small fixtures).

    Deliberately does not call production clocks/features/linear/history_gaps.
    The inherited negative-gap clamp and first/padding mask are reproduced.
    """
    if mode not in MODES[1:] or timestamps.shape != valid.shape:
        raise ValueError("Invalid reference inputs")
    rows = []
    for b in range(timestamps.shape[0]):
        elapsed = timestamps.new_zeros((), dtype=torch.float64)
        positions = []
        for t in range(timestamps.shape[1]):
            active = bool(valid[b, t]) and t > 0 and bool(valid[b, t-1])
            if bool(valid[b, t]) and not bool(torch.isfinite(timestamps[b, t])):
                raise ValueError("Nonfinite history timestamp")
            if not active:
                positions.append(weight.double().sum(-1) * 0.0)
                continue
            elapsed = elapsed + (timestamps[b, t].double() - timestamps[b, t-1].double()).clamp_min(0)
            clock = timestamps[b, t].double() if mode == "absolute_phase" else elapsed
            basis = []
            for period in PERIODS_MS:
                angle = clock.remainder(period) / period * (2 * math.pi)
                basis.extend((angle.sin() / math.sqrt(2), angle.cos() / math.sqrt(2)))
            angles = []
            for row in weight.double():
                angles.append(sum(row[k] * basis[k] for k in range(4)).tanh())
            positions.append(torch.stack(angles))
        rows.append(torch.stack(positions))
    return torch.stack(rows)


clock = phase_clock
features = periodic_features
apply_correction = phase_correction
