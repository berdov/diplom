"""Independent differentiable recurrence, not a replacement production kernel.

Full-forward kernel's shifted source weight is gamma[t] + beta[t+1].
Equivalently h[t] = exp(ADT[t]) * (h[t-1] + beta[t] * KV[t-1])
                  + gamma[t] * KV[t], with h[-1] = KV[-1] = 0.
This reference uses ordinary PyTorch elementary functions without approximate
Triton math or internal bf16 rounding; those differences get separate tolerances.
"""

import math
import torch
from torch.nn import functional as F


def intermediates(adt, dw, dp, trap, angles):
    theta = torch.remainder((angles.tanh() * math.pi * dp.permute(0, 2, 1)[..., None]).cumsum(1), 2*math.pi)
    gate = trap.sigmoid()
    return dict(theta=theta, decay=adt.exp(), gamma=dw*gate, beta=dw*(1-gate))


def rotary(x, theta, mimo=False):
    """x [B,L,H,R,N], theta [B,L,H,A]; native layouts differ."""
    a, n = theta.shape[-1], x.shape[-1]
    c, s = theta.cos().unsqueeze(-2), theta.sin().unsqueeze(-2)
    if mimo:
        left, right = x[..., :a], x[..., n//2:n//2+a]
        return torch.cat((left*c-right*s, x[..., a:n//2], left*s+right*c, x[..., n//2+a:]), -1)
    left, right = x[..., :2*a:2], x[..., 1:2*a:2]
    paired = torch.stack((left*c-right*s, left*s+right*c), -1).flatten(-2)
    return torch.cat((paired, x[..., 2*a:]), -1)


def recurrence(q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z,
               mv=None, mz=None, mo=None, *, return_intermediates=False):
    mimo = mv is not None
    h = v.shape[2]
    if mimo:
        # Native inputs [B,L,R,G,N], biases [H,R,N].
        q, k = q.permute(0, 1, 3, 2, 4), k.permute(0, 1, 3, 2, 4)
    else:
        q, k, qb, kb = q.unsqueeze(-2), k.unsqueeze(-2), qb.unsqueeze(1), kb.unsqueeze(1)
    q, k = (t.repeat_interleave(h // t.shape[2], 2) for t in (q, k))
    q, k = q + qb, k + kb
    parts = intermediates(adt, dw, dp, trap, angles)
    qr, kr = rotary(q, parts["theta"], mimo), rotary(k, parts["theta"], mimo)
    vp = v.unsqueeze(-2) if not mimo else v.unsqueeze(-2)*mv
    state = v.new_zeros(v.shape[0], h, k.shape[-1], v.shape[-1])
    previous = torch.zeros_like(state)
    outputs = []
    for t in range(v.shape[1]):
        kv = torch.einsum("bhrn,bhrp->bhnp", kr[:, t], vp[:, t])
        past = parts["decay"][:, :, t, None, None] * (
            state + parts["beta"][:, :, t, None, None]*previous)
        state = past + parts["gamma"][:, :, t, None, None]*kv
        # Same-position rotation cancels exactly. Match the upstream explicit
        # diagonal (unrotated QK) to avoid spurious tiny phase derivatives.
        diagonal = torch.einsum("bhrn,bhsn,bhsp->bhrp", q[:, t], k[:, t], vp[:, t])
        y = torch.einsum("bhrn,bhnp->bhrp", qr[:, t], past)
        y = y + parts["gamma"][:, :, t, None, None]*diagonal
        if d is not None:
            y = y + d[None, :, None, None]*vp[:, t]
        if z is not None:
            gate = z[:, t].unsqueeze(-2) if not mimo else z[:, t].unsqueeze(-2)*mz
            y = y * F.silu(gate)
        outputs.append(y.squeeze(-2) if not mimo else (y*mo).sum(-2))
        previous = kv
    out = torch.stack(outputs, 1)
    return (out, parts) if return_intermediates else out
