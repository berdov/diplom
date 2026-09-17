"""SISO full forward adapted from pinned Mamba, Apache-2.0.

Copyright Dao AI Lab / Goombalab (2026).
Pin e9594ce1c732d97440f0332fdc43170a2294dbfa; license in
../mamba3_timeaware/LICENSE.upstream. Only decay_dt vs scan_dt differs.
"""

import torch
from einops import rearrange
from torch.nn import functional as F
from experiments.mamba3_timeaware.time_mamba3 import require_siso
from .time_mechanisms import mechanism_dt


def mechanism_mamba3_forward(mixer, u, decay_scale, scan_scale):
    require_siso(mixer)
    from mamba_ssm.modules.mamba3 import heavy_tail_activation, mamba3_siso_combined
    if u.ndim != 3 or any(s.shape != (*u.shape[:2], mixer.nheads) for s in (decay_scale, scan_scale)):
        raise ValueError('Expected u [B,L,D] and scales [B,L,H]')
    projected = mixer.in_proj(u)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        projected, [mixer.d_inner, mixer.d_inner,
                    mixer.d_state * mixer.num_bc_heads, mixer.d_state * mixer.num_bc_heads,
                    mixer.nheads, mixer.nheads, mixer.nheads, mixer.num_rope_angles], dim=-1)
    z = rearrange(z, 'b l (h p) -> b l h p', p=mixer.headdim)
    x = rearrange(x, 'b l (h p) -> b l h p', p=mixer.headdim)
    B = rearrange(B, 'b l (r g n) -> b l r g n', r=1, g=mixer.num_bc_heads)
    C = rearrange(C, 'b l (r g n) -> b l r g n', r=1, g=mixer.num_bc_heads)
    trap = rearrange(trap, 'b l h -> b h l')
    a = -heavy_tail_activation(dd_A.to(torch.float32))
    a = torch.clamp(a, max=-mixer.A_floor)
    dt_base = F.softplus(dd_dt + mixer.dt_bias)
    dt, adt = mechanism_dt(dt_base, a, decay_scale, scan_scale)
    dt = rearrange(dt, 'b l n -> b n l')
    adt = rearrange(adt, 'b l n -> b n l')
    angles = angles.unsqueeze(-2).expand(-1, -1, mixer.nheads, -1).to(torch.float32)
    B, C = mixer.B_norm(B), mixer.C_norm(C)
    y = mamba3_siso_combined(
        Q=C.squeeze(2), K=B.squeeze(2), V=x, ADT=adt, DT=dt, Trap=trap,
        Q_bias=mixer.C_bias.squeeze(1), K_bias=mixer.B_bias.squeeze(1),
        Angles=angles, D=mixer.D, Z=z if not mixer.is_outproj_norm else None,
        chunk_size=mixer.chunk_size, Input_States=None, return_final_states=False, cu_seqlens=None)
    y = rearrange(y, 'b l h p -> b l (h p)')
    if mixer.is_outproj_norm:
        z = rearrange(z, 'b l h p -> b l (h p)')
        y = mixer.norm(y, z)
    return mixer.out_proj(y.to(x.dtype))
