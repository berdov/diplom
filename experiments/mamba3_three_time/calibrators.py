"""One observed gap, three optional bounded functions; no new time features."""

import copy
import math
import torch
from torch import nn
from experiments.mamba3_timeaware.time_inputs import TimeCalibrator

MODES = ("base", "dual", "triple")
REFERENCE_MS = 838393


class ThreeTimes(nn.Module):
    def __init__(self, mode, n_heads=2):
        super().__init__()
        if mode not in MODES or n_heads != 2:
            raise ValueError("Expected base/dual/triple and two temporal heads")
        self.mode, self.n_heads = mode, n_heads
        self.calibrators = nn.ModuleDict()
        # These CPU-created modules must not advance the backbone/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            if mode != "base":
                first = TimeCalibrator(n_heads, REFERENCE_MS, math.log(2))
                self.calibrators["decay"] = first
                for name in (("scan",) if mode == "dual" else ("write", "phase")):
                    self.calibrators[name] = copy.deepcopy(first)

    def forward(self, gaps, active):
        if self.mode == "base":
            one = torch.ones(*gaps.shape, self.n_heads, device=gaps.device)
            return one, one, one
        decay = self.calibrators["decay"](gaps, active)
        if self.mode == "dual":
            scan = self.calibrators["scan"](gaps, active)
            return decay, scan, scan
        return decay, self.calibrators["write"](gaps, active), self.calibrators["phase"](gaps, active)


def split_dt(d, a, decay, write, phase):
    if any(t.shape != d.shape for t in (a, decay, write, phase)):
        raise ValueError("All time inputs must have shape [B,L,H]")
    adt = (a * (d * decay)).permute(0, 2, 1)
    dt_write = (d * write).permute(0, 2, 1)
    # Preserve the same autograd object in dual, not merely equal values.
    dt_phase = dt_write if phase is write else (d * phase).permute(0, 2, 1)
    return adt, dt_write, dt_phase
