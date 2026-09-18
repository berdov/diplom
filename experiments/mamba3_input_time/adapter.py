"""Event-preserving causal input adapters. No RecBole, CUDA or Mamba imports."""
import math

import torch
from torch import nn

from experiments.mamba3_timeaware.time_inputs import history_gaps

MODES = ('separate_replay', 'time_add', 'attention_content', 'attention_time')
ADAPTER_COUNTS = dict(zip(MODES, (0, 1120, 8192, 8240)))
COUNTS = {mode: 610572 + n for mode, n in ADAPTER_COUNTS.items()}
REFERENCE = 838393


def log_interval(gaps):
    """Float64 log1p(gap/reference), including exact zero without overflow."""
    x = gaps.double()
    positive = x.clamp_min(torch.finfo(torch.float64).tiny).log() - math.log(REFERENCE)
    return torch.where(x > 0, torch.logaddexp(positive, torch.zeros_like(x)), 0.)


def allowed_pairs(valid):
    length = valid.shape[1]
    causal = torch.arange(length, device=valid.device)[:, None] >= torch.arange(length, device=valid.device)[None, :]
    return valid[:, :, None] & valid[:, None, :] & causal


def pair_intervals(times, valid, allowed):
    clean = torch.where(valid, times, 0.)
    gaps = torch.where(allowed, (clean[:, :, None] - clean[:, None, :]).clamp_min(0), 0.)
    if not torch.isfinite(gaps).all():
        raise ValueError('Nonfinite allowed pair interval')
    return log_interval(gaps)


class InputAdapter(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        if mode == 'time_add':
            self.first = nn.Linear(1, 16)
            self.last = nn.Linear(16, 64)
            nn.init.zeros_(self.last.weight)
            nn.init.zeros_(self.last.bias)
        elif mode.startswith('attention_'):
            self.q = nn.Linear(64, 32, bias=False)
            self.k = nn.Linear(64, 32, bias=False)
            self.v = nn.Linear(64, 32, bias=False)
            self.out = nn.Linear(32, 64, bias=False)
            nn.init.zeros_(self.out.weight)
            if mode == 'attention_time':
                self.time_first = nn.Linear(1, 8)
                self.time_last = nn.Linear(8, 4, bias=False)
                nn.init.zeros_(self.time_last.weight)
        if sum(p.numel() for p in self.parameters()) != ADAPTER_COUNTS[mode]:
            raise ValueError('Adapter parameter count mismatch')

    def forward(self, u, timestamps, valid, *, details=False):
        if (u.ndim != 3 or u.shape[-1] != 64 or timestamps.shape != u.shape[:2]
                or valid.shape != timestamps.shape or valid.dtype != torch.bool
                or timestamps.dtype != torch.float64 or u.dtype not in (torch.float32, torch.float64)
                or not (u.device == timestamps.device == valid.device)):
            raise ValueError('Expected embeddings [B,L,64], float64 times, bool mask on same device')
        lengths = valid.sum(1)
        expected = torch.arange(valid.shape[1], device=valid.device)[None, :] < lengths[:, None]
        if (lengths < 1).any() or not torch.equal(valid, expected):
            raise ValueError('Nonempty right-padded histories required')
        if not torch.isfinite(timestamps[valid]).all() or not torch.isfinite(u[valid]).all():
            raise ValueError('Nonfinite valid history')
        extra = {}
        if self.mode == 'separate_replay':
            delta = torch.zeros_like(u)
        elif self.mode == 'time_add':
            gaps, active = history_gaps(timestamps, valid)
            tau = log_interval(gaps).to(u.dtype).unsqueeze(-1)
            delta = self.last(torch.nn.functional.silu(self.first(tau)))
            delta = torch.where(active[:, :, None], delta, 0.)
        else:
            clean = torch.where(valid[:, :, None], u, 0.)
            batch, length = valid.shape
            def heads(projection):
                return projection(clean).reshape(batch, length, 4, 8).transpose(1, 2)
            q, k, v = heads(self.q), heads(self.k), heads(self.v)
            allowed = allowed_pairs(valid)
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(8)
            bias = None
            if self.mode == 'attention_time':
                tau = pair_intervals(timestamps, valid, allowed).to(u.dtype)
                bias = self.time_last(torch.nn.functional.silu(self.time_first(tau.unsqueeze(-1))))
                bias = bias.permute(0, 3, 1, 2)
                logits = logits + bias
            logits = logits.masked_fill(~allowed[:, None], -torch.inf)
            # Invalid query rows use finite dummy logits only; all their weights are then zero.
            logits = torch.where(valid[:, None, :, None], logits, 0.)
            weights = logits.softmax(-1).masked_fill(~allowed[:, None], 0.)
            attended = (weights @ v).transpose(1, 2).reshape(batch, length, 32)
            delta = self.out(attended).masked_fill(~valid[:, :, None], 0.)
            if details:
                extra = dict(weights=weights.detach(), allowed=allowed, bias=None if bias is None else bias.detach())
        output = u + delta
        if details:
            return output, dict(delta=delta.detach(), **extra)
        return output

    def common_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('time_')}
