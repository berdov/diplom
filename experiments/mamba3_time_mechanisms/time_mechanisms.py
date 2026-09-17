"""Independent positive decay/scan scales; no kernel dependencies."""

import copy
import math
import torch
from torch import nn
from experiments.mamba3_timeaware.time_inputs import TimeCalibrator

MODES = ('vanilla', 'decay_only', 'scan_only', 'shared', 'separate')
COUNTS = dict(vanilla=610440, decay_only=610506, scan_only=610506,
              shared=610506, separate=610572)


class TimeMechanisms(nn.Module):
    def __init__(self, mode, n_heads=2, reference=838393.0, bound=math.log(2)):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown time mechanism mode')
        if (n_heads, reference, bound) != (2, 838393.0, math.log(2)):
            raise ValueError('Frozen calibration settings required')
        self.mode, self.n_heads = mode, n_heads
        self.calibrators = nn.ModuleDict()
        names = dict(vanilla=(), decay_only=('decay',), scan_only=('scan',),
                     shared=('shared',), separate=('decay', 'scan'))[mode]
        for name in names:
            self.calibrators[name] = (copy.deepcopy(self.calibrators['decay'])
                                     if name == 'scan' and mode == 'separate'
                                     else TimeCalibrator(n_heads, reference, bound))
        assert sum(p.numel() for p in self.parameters()) == COUNTS[mode] - COUNTS['vanilla']

    def forward(self, gaps, active):
        one = torch.ones(*gaps.shape, self.n_heads, device=gaps.device, dtype=torch.float32)
        if self.mode == 'vanilla':
            return one, one
        scales = {name: module(gaps, active) for name, module in self.calibrators.items()}
        if self.mode == 'shared':
            return scales['shared'], scales['shared']
        return scales.get('decay', one), scales.get('scan', one)


def mechanism_dt(dt_base, a, decay_scale, scan_scale):
    if dt_base.ndim != 3 or any(t.shape != dt_base.shape for t in (a, decay_scale, scan_scale)):
        raise ValueError('Expected identical [B,L,H] shapes')
    for scale in (decay_scale, scan_scale):
        if scale.device != dt_base.device or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Scales must be finite, positive and on the DT device')
    decay_dt = dt_base * decay_scale.to(dt_base.dtype)
    # Preserve the frozen shared graph: accumulate scan/decay gradients before scaling.
    scan_dt = decay_dt if scan_scale is decay_scale else dt_base * scan_scale.to(dt_base.dtype)
    return scan_dt, a * decay_dt
