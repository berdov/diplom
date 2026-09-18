"""Dense-only split-DT autograd adapters, derived from pinned Mamba3 wrappers.

Copyright (c) 2025-2026, Dao AI Lab, Goombalab. Apache-2.0; LICENSE.upstream.
Pin e9594ce1c732d97440f0332fdc43170a2294dbfa. Changes: DT_write and
DT_phase are distinct inputs/gradient returns; state/varlen/norm paths omitted
and rejected, low-level kernels unchanged. No site-packages modifications.
"""

import torch


def dense_only(cu_seqlens=None, input_states=None, return_final_states=False):
    if cu_seqlens is not None or input_states is not None or return_final_states:
        raise NotImplementedError("Only dense full sequences without cached/final states")


class SISO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, chunk):
        from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd
        theta, _ = angle_dt_fwd(angles, dp, chunk_size=chunk, return_output_state=True)
        out, ov, states, cs, total, qr, ks, qk, scale, gamma, _ = mamba3_siso_fwd(
            q, k, v, adt, dw, trap, qb, kb, theta, d, z, None,
            chunk_size=chunk, store_states_adt_outv=any(ctx.needs_input_grad),
            return_final_states=False, cu_seqlens=None)
        ctx.chunk = chunk
        ctx.save_for_backward(q, k, v, dw, dp, trap, qb, kb, angles, theta,
                              d, z, ov, states, cs, total, qr, ks, qk, scale, gamma)
        return out

    @staticmethod
    def backward(ctx, grad):
        from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_bwd
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import (
            compute_dzdo, compute_dqkv, compute_dqktheta, compute_ddt_dtrap_dinput_states)
        q, k, v, dw, dp, trap, qb, kb, angles, theta, d, z, ov, states, cs, total, qr, ks, qk, scale, gamma = ctx.saved_tensors
        dz, go = compute_dzdo(grad, z, ov, chunk_size=ctx.chunk) if z is not None else (None, grad)
        dq0, dk0, dv, da, dqk, dd, _ = compute_dqkv(
            q=qr, k=ks, v=v, da_cs=cs, da_cs_sum=total, qk_dot=qk,
            SSM_States=states, do=go, d_ossm_state=None, d_ov_state=None, D=d,
            chunk_size=ctx.chunk, has_input_state=False, Cu_Seqlens=None)
        dq, dk, dqb, dkb, dt, ds, dg = compute_dqktheta(
            q=q, k=k, scale=scale, gamma=gamma, q_bias=qb, k_bias=kb, angles=theta,
            dq_in=dq0, dk_in=dk0, dqk=dqk, d_ok_state=None,
            chunk_size=ctx.chunk, Cu_Seqlens=None)
        ddw, dtrap, _, _, _ = compute_ddt_dtrap_dinput_states(
            dscale=ds, dgamma=dg, dt=dw, trap=trap.float(), d_issm_state=None,
            input_k_state=None, input_v_state=None, Cu_Seqlens=None)
        dang, ddp, _ = angle_dt_bwd(grad_out=dt, angle=angles, dt=dp,
            has_init_state=False, chunk_size=ctx.chunk, grad_output_state=None, cu_seqlens=None)
        return dq, dk, dv, da, ddw, ddp, dtrap, dqb, dkb, dang, dd, dz, None


def siso(q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, *, chunk=64,
         cu_seqlens=None, input_states=None, return_final_states=False):
    dense_only(cu_seqlens, input_states, return_final_states)
    # Pinned public SISO API forces these casts, including raw angles.
    q, k, v, trap, angles = (x.to(torch.bfloat16) for x in (q, k, v, trap, angles))
    z = z.to(torch.bfloat16) if z is not None else None
    return SISO.apply(q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, chunk)


class MIMO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, mv, mz, mo, chunk):
        from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
        from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd import mamba_mimo_forward
        q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, mv, mz, mo = (
            t.contiguous() if t is not None else None
            for t in (q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, mv, mz, mo))
        theta, _ = angle_dt_fwd(angles, dp, chunk_size=chunk, return_output_state=True)
        cs, rev, seg = compute_dacs_segsum_triton(adt, chunk)
        out, _, _ = mamba_mimo_forward(q, k, v, qb, kb, mv, mo, z, d, mz,
            theta, cs, rev, dw, trap, seg, chunk_size=chunk, rotary_dim_divisor=4,
            dtype=v.dtype, return_state=False, fuse_pregate_headwise_rms_norm=False,
            outproj_norm_weight=None, outproj_norm_eps=1e-5)
        ctx.chunk = chunk
        ctx.save_for_backward(q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, mv, mz, mo, theta)
        return out

    @staticmethod
    def backward(ctx, grad):
        from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_bwd
        from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import mamba_mimo_bwd_combined
        q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, mv, mz, mo, theta = ctx.saved_tensors
        cs, rev, seg = compute_dacs_segsum_triton(adt, ctx.chunk)
        dq, dk, dv, da, ddw, dt, dqb, dkb, dmv, dmz, dmo, dtheta, dd, dz, _ = mamba_mimo_bwd_combined(
            grad.contiguous(), q, k, v, qb, kb, mv, mo, z, mz, theta, cs, rev,
            dw, trap, d, seg, ctx.chunk, 4, v.dtype,
            fuse_pregate_headwise_rms_norm=False, outproj_norm_weight=None, outproj_norm_eps=1e-5)
        dang, ddp, _ = angle_dt_bwd(grad_out=dtheta, angle=angles, dt=dp,
            has_init_state=False, chunk_size=ctx.chunk, cu_seqlens=None)
        return dq, dk, dv, da, ddw, ddp, dt, dqb, dkb, dang, dd, dz, dmv, dmz, dmo, None


def mimo(q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, mv, mz, mo, *, chunk=16,
         cu_seqlens=None, input_states=None, return_final_states=False):
    dense_only(cu_seqlens, input_states, return_final_states)
    if q.shape[2] != 4 or chunk != 16 or angles.shape[-1] != q.shape[-1] // 4:
        raise NotImplementedError("Frozen rank4/chunk16/half-rotary scope only")
    return MIMO.apply(q, k, v, adt, dw, dp, trap, qb, kb, angles, d, z, mv, mz, mo, chunk)
