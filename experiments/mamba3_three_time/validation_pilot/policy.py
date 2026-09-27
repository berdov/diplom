"""Residual reporting without clipping, denominator floors, or historical rewrites."""
import json
import math
from .config import HERE

EPS32 = 1.1920928955078125e-7


def policy():
    p = json.loads((HERE / 'numeric_acceptance_v1.json').read_text())
    if p['future_gradient']['maximum_rho'] != EPS32 or p['future_gradient']['denominator_floor'] != 0:
        raise ValueError('Frozen SISO residual policy changed')
    return p


def from_norms(prefix_l2, future_l2):
    p, f = float(prefix_l2), float(future_l2)
    if not all(math.isfinite(x) and x >= 0 for x in (p, f)):
        return dict(status='NONFINITE', informative=False, rho=None, finite_precision_pass=False)
    if p == f == 0:
        return dict(status='ZERO_SIGNAL', informative=False, rho=None, finite_precision_pass=False)
    if not math.isfinite(p + f):
        return dict(status='NONFINITE', informative=False, rho=None, finite_precision_pass=False)
    rho = f / (p + f)
    return dict(status='MEASURED', informative=p > 0, rho=rho,
                finite_precision_pass=p > 0 and rho <= EPS32)


def positional(gradient, prefix):
    import torch
    g = gradient.detach().double()
    if g.ndim != 3 or not 0 < prefix < g.shape[1]:
        raise ValueError('Expected informative [batch, position, channel] prefix fixture')
    gp, gf = g[:, :prefix], g[:, prefix:]
    p, f = gp.norm().item(), gf.norm().item()
    stats = from_norms(p, f)
    finite_number = lambda v: v if math.isfinite(v) else None
    stats.update(prefix_l2=finite_number(p), future_l2=finite_number(f),
                 max_abs=finite_number(gf.abs().max().item()),
                 mean_abs=finite_number(gf.abs().mean().item()), nonzero_count=int(torch.count_nonzero(gf)),
                 legacy_exact_zero=bool(torch.count_nonzero(gf) == 0),
                 norm_dtype='float64', gradient_dtype=str(gradient.dtype), threshold=EPS32)
    return stats
