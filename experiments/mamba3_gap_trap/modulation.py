"""One projected scalar; deterministic zero initialization and no RNG draws."""
import torch
from torch import nn
from experiments.mamba3_three_time.calibrators import REFERENCE_MS


class GapTrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, gaps, active):
        if gaps.ndim != 2 or gaps.shape != active.shape or active.dtype != torch.bool:
            raise ValueError('Expected gaps and boolean active [B,L]')
        if not torch.isfinite(gaps).all() or (gaps < 0).any():
            raise ValueError('Finite nonnegative history gaps required')
        g = gaps.double()
        q = (g / (g + REFERENCE_MS)).to(self.alpha.dtype)
        # Clamp has derivative 1 at both boundaries in the pinned PyTorch.
        # Project after each Adam step so its flat exterior cannot trap alpha.
        return self.alpha.clamp(0, 1) * torch.where(active, q, 0.)

    @torch.no_grad()
    def project(self):
        if not torch.isfinite(self.alpha).all():
            raise ValueError('Nonfinite gap-Trap parameter')
        self.alpha.clamp_(0, 1)


def attach_projection(model, optimizer):
    if model.gap_trap_mode == 'fixed_replay':
        return None
    return optimizer.register_step_post_hook(lambda _opt, _args, _kwargs: model.gap_trap.project())
