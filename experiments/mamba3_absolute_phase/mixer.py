"""Pinned projections with one additive correction before native angle_dt.

Copyright (c) 2026, Dao AI Lab, Goombalab. Apache-2.0;
see experiments/mamba3_three_time/LICENSE.upstream.
Projection order derived from e9594ce1c732d97440f0332fdc43170a2294dbfa.
"""

import torch
from torch.nn import functional as F

from experiments.mamba3_three_time.calibrators import split_dt
from experiments.mamba3_three_time import kernels


NATIVE_KEYS = ("q", "k", "v", "adt", "dw", "dp", "trap", "qb", "kb",
               "angles", "d", "z", "mv", "mz", "mo")


def project_inputs(mixer, u, decay, write, phase, raw_angle_correction=None):
    from mamba_ssm.modules.mamba3 import heavy_tail_activation
    if (not mixer.is_mimo or mixer.is_outproj_norm or mixer.rotary_dim_divisor != 4
            or mixer.mimo_rank != 4 or mixer.chunk_size != 8):
        raise NotImplementedError("Frozen MIMO rank4/chunk8/half-rotary configuration only")
    b, length, _ = u.shape
    h, p, n, g, rank = (mixer.nheads, mixer.headdim, mixer.d_state,
                         mixer.num_bc_heads, mixer.mimo_rank)
    z, x, B, C, dd_dt, dd_a, trap, raw = torch.split(mixer.in_proj(u),
        [mixer.d_inner, mixer.d_inner, n*g*rank, n*g*rank,
         h, h, h, mixer.num_rope_angles], -1)
    z, x = z.reshape(b, length, h, p), x.reshape(b, length, h, p)
    B, C = B.reshape(b, length, rank, g, n), C.reshape(b, length, rank, g, n)
    a = (-heavy_tail_activation(dd_a.float())).clamp(max=-mixer.A_floor)
    adt, dw, dp = split_dt(F.softplus(dd_dt + mixer.dt_bias), a, decay, write, phase)
    before = raw.unsqueeze(-2).expand(-1, -1, h, -1).float()
    if raw_angle_correction is None:
        angles = before
        raw_angle_correction = before.new_zeros(b, length, mixer.num_rope_angles)
    else:
        if raw_angle_correction.shape != (b, length, mixer.num_rope_angles):
            raise ValueError("Expected shared raw-angle correction [B,L,A]")
        angles = before + raw_angle_correction.unsqueeze(-2)
    B, C = mixer.B_norm(B), mixer.C_norm(C)
    return dict(q=C, k=B, v=x, adt=adt, dw=dw, dp=dp,
                trap=trap.permute(0, 2, 1), qb=mixer.C_bias, kb=mixer.B_bias,
                angles=angles, d=mixer.D, z=z, mv=mixer.mimo_x,
                mz=mixer.mimo_z, mo=mixer.mimo_o,
                raw_angles_before=before, raw_angle_correction=raw_angle_correction,
                raw_angles_corrected=angles)


def phase_forward(mixer, u, decay, write, phase, raw_angle_correction=None,
                  *, reference=False, observer=None, metadata=None):
    values = project_inputs(mixer, u, decay, write, phase, raw_angle_correction)
    if observer is not None:
        observer(dict(values, **(metadata or {})))
    if reference:
        from experiments.mamba3_three_time.reference import recurrence
        # Explicit diagnostic path. Never selected by the scientific runner.
        args = {key: value.double() if value is not None else None
                for key, value in values.items() if key in NATIVE_KEYS}
        y = recurrence(**args)
    else:
        y = kernels.mimo(*(values[key] for key in NATIVE_KEYS), chunk=mixer.chunk_size)
    return mixer.out_proj(y.reshape(u.shape[0], u.shape[1], -1).to(values["v"].dtype))
