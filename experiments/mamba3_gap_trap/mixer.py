"""Pinned projection order; only raw Trap logits receive the gap shift.

Copyright (c) 2026, Dao AI Lab, Goombalab. Apache-2.0; LICENSE.upstream.
Derived from modules/mamba3.py at e9594ce1c732d97440f0332fdc43170a2294dbfa.
"""

import torch
from torch.nn import functional as F
from experiments.mamba3_three_time.calibrators import split_dt
from experiments.mamba3_three_time import kernels


def gap_trap_forward(mixer, u, decay, write, phase, shift):
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
    trap = (trap.float() + shift.unsqueeze(-1)).to(trap.dtype).permute(0, 2, 1)
    angles = angles.unsqueeze(-2).expand(-1, -1, h, -1).float()
    B, C = mixer.B_norm(B), mixer.C_norm(C)
    if not mixer.is_mimo:
        raise NotImplementedError("MIMO rank4/chunk8 pilot only")
    y = kernels.mimo(C, B, x, adt, dw, dp, trap, mixer.C_bias, mixer.B_bias,
        angles, mixer.D, z, mixer.mimo_x, mixer.mimo_z, mixer.mimo_o, chunk=mixer.chunk_size)
    return mixer.out_proj(y.reshape(b, length, -1).to(x.dtype))
