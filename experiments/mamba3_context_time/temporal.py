"""Item-local bounded log-scale mixtures; no recurrent state or target inputs."""
import math

import torch
from torch import nn
from experiments.mamba3_timeaware.time_inputs import TimeCalibrator

MODES = ('separate_replay', 'dense11', 'dense12', 'uniform', 'routed')
TEMPORAL_COUNTS = dict(separate_replay=132, dense11=774, dense12=844, uniform=528, routed=792)
COUNTS = {k: 610440 + v for k, v in TEMPORAL_COUNTS.items()}
REFERENCE = 838393.0
BOUND = math.log(2)


def gap_tau(gaps):
    if gaps.ndim != 2 or not torch.isfinite(gaps).all() or (gaps < 0).any():
        raise ValueError('Finite nonnegative [B,L] gaps required')
    reference = torch.tensor(REFERENCE, dtype=torch.float64, device=gaps.device)
    ratio = gaps.double().clamp_min(torch.finfo(torch.float64).tiny).log() - reference.log()
    tau = torch.logaddexp(ratio, torch.zeros_like(ratio))
    return torch.where(gaps > 0, tau, 0.0).unsqueeze(-1).float()


def mix_log_scales(q, probabilities):
    return (probabilities[..., None, None] * q).sum(dim=-3)


class ContextTime(nn.Module):
    def __init__(self, mode, *, experts=4):
        super().__init__()
        if mode not in MODES[1:] or experts not in (1, 4):
            raise ValueError('Unknown mode or synthetic expert count')
        if mode.startswith('dense') and experts != 4:
            raise ValueError('Expert override applies only to synthetic banks')
        self.mode, self.experts = mode, experts
        if mode.startswith('dense'):
            width = int(mode.removeprefix('dense'))
            self.first, self.last = nn.Linear(65, width), nn.Linear(width, 4)
            nn.init.zeros_(self.last.weight)
            nn.init.zeros_(self.last.bias)
        else:
            self.bank = nn.ModuleList([
                nn.ModuleDict({p: TimeCalibrator(2, REFERENCE, BOUND) for p in ('decay', 'scan')})
                for _ in range(experts)
            ])
            if mode == 'routed':
                self.router = nn.Linear(65, experts)
                nn.init.normal_(self.router.weight, std=.01)
                nn.init.zeros_(self.router.bias)
        if experts == 4 and sum(p.numel() for p in self.parameters()) != TEMPORAL_COUNTS[mode]:
            raise ValueError('Unexpected temporal parameter count')

    def forward(self, u, gaps, active):
        if (u.shape != (*gaps.shape, 64) or u.dtype != torch.float32 or
                active.shape != gaps.shape or active.dtype != torch.bool or
                u.device != gaps.device or active.device != gaps.device or not torch.isfinite(u).all()):
            raise ValueError('Expected fp32 item-local embeddings and aligned gap/mask tensors')
        tau = gap_tau(gaps)
        details = {'tau': tau}
        if self.mode.startswith('dense'):
            raw = self.last(torch.nn.functional.silu(self.first(torch.cat((u, tau), -1))))
            ell = BOUND * raw.reshape(*gaps.shape, 2, 2).tanh()
        else:
            q = torch.stack([
                torch.stack([BOUND * pair[p].last(torch.nn.functional.silu(pair[p].first(tau))).tanh()
                             for p in ('decay', 'scan')], dim=-2)
                for pair in self.bank
            ], dim=-3)
            if self.mode == 'uniform':
                pi = torch.full((*gaps.shape, self.experts), 1 / self.experts, device=u.device)
            else:
                pi = self.router(torch.cat((u, tau), -1)).softmax(-1)
            ell = mix_log_scales(q, pi)
            details.update(probabilities=pi, log_experts=q)
        scales = torch.where(active[..., None, None], ell.exp(), torch.ones_like(ell))
        return scales[..., 0, :], scales[..., 1, :], details
