"""Unchanged trainer with analytic centered and kernel-boundary diagnostics."""
import math
import torch
from .. import trainer as parent
from . import config as c
from .reuse import bind


def diagnostics(model):
    alpha = float(model.gap_trap.alpha.detach()) if model.gap_trap_mode == 'centered_gap_trap' else 0.
    grid, logits = c.plan()['diagnostic_grid'], c.plan()['diagnostic_content_logits']
    q = [(g-1)/(g+1) for g in grid]
    shifts = [alpha*x for x in q]
    raw = torch.tensor(logits, dtype=torch.bfloat16)[:, None]
    # Match modulation's q->fp32 cast followed by fp32 alpha multiplication.
    shift32 = (torch.tensor(q, dtype=torch.float32)*torch.tensor(alpha, dtype=torch.float32))[None, :]
    effective = (raw.float()+shift32).to(raw.dtype).float()-raw.float()
    eligible = shift32.expand_as(effective) != 0
    zeros = int(((effective == 0) & eligible).sum())
    count = int(eligible.sum())
    return dict(alpha=alpha, alpha_bounds=[0.,1.], at_lower_bound=alpha==0., at_upper_bound=alpha==1.,
                shared_across_heads_and_layers=True, gaps_over_R0=grid, q_centered=q,
                logit_shift=shifts, odds_multiplier=[math.exp(x) for x in shifts], content_logits=logits,
                current_fraction=[[1/(1+math.exp(-t-s)) for s in shifts] for t in logits],
                bf16=dict(shift_before_cast=shift32[0].tolist(), shift_cast_alone=shift32.to(torch.bfloat16)[0].float().tolist(),
                          effective_shift_after_add_and_cast=effective.tolist(), nonzero_shift_grid_cases=count,
                          rounded_to_zero_cases=zeros, rounded_to_zero_fraction=zeros/count if count else None),
                interpretation='Analytic fixed content logits, active gaps; first event/padding are neutral. No dataset forward.')


trainer_class = bind(parent, {'diagnostics': diagnostics})['trainer_class']
