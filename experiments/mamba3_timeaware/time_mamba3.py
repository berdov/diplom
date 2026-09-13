"""SISO full forward adapted from state-spaces/mamba, copyright Dao AI Lab,
Goombalab (2026), Apache-2.0; commit e9594ce1c732d97440f0332fdc43170a2294dbfa.

Only native DT is conditioned; no inference cache, MIMO or step path.
Official parameters, norms, activation and scan remain upstream-owned.
"""

import torch
from einops import rearrange
from torch.nn import functional as F

from .time_inputs import condition_dt


def require_siso(mixer):
    if mixer.is_mimo:
        raise ValueError("Only SISO full-sequence forward is supported")


def time_mamba3_forward(mixer, u, time_scale):
    require_siso(mixer)
    from mamba_ssm.modules.mamba3 import heavy_tail_activation, mamba3_siso_combined

    if u.ndim != 3 or time_scale.shape != (*u.shape[:2], mixer.nheads):
        raise ValueError("Expected u [B,L,D] and time_scale [B,L,nheads]")
    projected = mixer.in_proj(u)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        projected,
        [mixer.d_inner, mixer.d_inner,
         mixer.d_state * mixer.num_bc_heads, mixer.d_state * mixer.num_bc_heads,
         mixer.nheads, mixer.nheads, mixer.nheads, mixer.num_rope_angles], dim=-1,
    )
    z = rearrange(z, "b l (h p) -> b l h p", p=mixer.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=1, g=mixer.num_bc_heads)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=1, g=mixer.num_bc_heads)
    trap = rearrange(trap, "b l h -> b h l")
    a = -heavy_tail_activation(dd_A.to(torch.float32))
    a = torch.clamp(a, max=-mixer.A_floor)
    dt_base = F.softplus(dd_dt + mixer.dt_bias)
    # The only architectural change: condition DT before both ADT and rotary scan.
    dt, adt = condition_dt(dt_base, a, time_scale)
    dt = rearrange(dt, "b l n -> b n l")
    adt = rearrange(adt, "b l n -> b n l")
    angles = angles.unsqueeze(-2).expand(-1, -1, mixer.nheads, -1).to(torch.float32)
    B = mixer.B_norm(B)
    C = mixer.C_norm(C)
    y = mamba3_siso_combined(
        Q=C.squeeze(2), K=B.squeeze(2), V=x, ADT=adt, DT=dt, Trap=trap,
        Q_bias=mixer.C_bias.squeeze(1), K_bias=mixer.B_bias.squeeze(1),
        Angles=angles, D=mixer.D, Z=z if not mixer.is_outproj_norm else None,
        chunk_size=mixer.chunk_size, Input_States=None,
        return_final_states=False, cu_seqlens=None,
    )
    y = rearrange(y, "b l h p -> b l (h p)")
    if mixer.is_outproj_norm:
        z = rearrange(z, "b l h p -> b l (h p)")
        y = mixer.norm(y, z)
    return mixer.out_proj(y.to(x.dtype))
