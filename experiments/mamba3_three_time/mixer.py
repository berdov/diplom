"""Pinned full-forward projection order; only the three DT consumers differ.

Copyright (c) 2026, Dao AI Lab, Goombalab. Apache-2.0; LICENSE.upstream.
Derived from modules/mamba3.py at e9594ce1c732d97440f0332fdc43170a2294dbfa.
"""

import torch
from torch.nn import functional as F
from .calibrators import split_dt
from . import kernels


def three_time_forward(mixer, u, decay, write, phase, *, official_tied=False):
    from mamba_ssm.modules.mamba3 import heavy_tail_activation
    if mixer.is_outproj_norm or mixer.rotary_dim_divisor != 4:
        raise NotImplementedError("Only frozen no-outnorm/half-rotary configuration")
    b, length, _ = u.shape
    h, p, n, g = mixer.nheads, mixer.headdim, mixer.d_state, mixer.num_bc_heads
    rank = mixer.mimo_rank if mixer.is_mimo else 1
    z, x, B, C, dd_dt, dd_a, trap, angles = torch.split(mixer.in_proj(u),
        [mixer.d_inner, mixer.d_inner, n*g*rank, n*g*rank, h, h, h, mixer.num_rope_angles], -1)
    z, x = z.reshape(b, length, h, p), x.reshape(b, length, h, p)
    B, C = B.reshape(b, length, rank, g, n), C.reshape(b, length, rank, g, n)
    a = (-heavy_tail_activation(dd_a.float())).clamp(max=-mixer.A_floor)
    adt, dw, dp = split_dt(F.softplus(dd_dt + mixer.dt_bias), a, decay, write, phase)
    trap = trap.permute(0, 2, 1)
    angles = angles.unsqueeze(-2).expand(-1, -1, h, -1).float()
    B, C = mixer.B_norm(B), mixer.C_norm(C)
    if official_tied and dw is not dp:
        raise ValueError("Official two-path oracle requires the same scan tensor")
    if mixer.is_mimo:
        if official_tied:
            from .length_adapter import official
            y = official(dict(q=C,k=B,v=x,adt=adt,dw=dw,dp=dp,trap=trap,
                qb=mixer.C_bias,kb=mixer.B_bias,mv=mixer.mimo_x,mz=mixer.mimo_z,
                mo=mixer.mimo_o,angles=angles,d=mixer.D,z=z))
        else:
            y = kernels.mimo(C, B, x, adt, dw, dp, trap, mixer.C_bias, mixer.B_bias,
                angles, mixer.D, z, mixer.mimo_x, mixer.mimo_z, mixer.mimo_o, chunk=mixer.chunk_size)
    else:
        y = kernels.siso(C.squeeze(2), B.squeeze(2), x, adt, dw, dp, trap,
            mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), angles, mixer.D, z, chunk=mixer.chunk_size)
    return mixer.out_proj(y.reshape(b, length, -1).to(x.dtype))
