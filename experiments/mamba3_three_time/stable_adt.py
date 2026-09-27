"""Local stable backward candidate, derived ONLY from mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py.

Copyright (c) 2026, Dao AI Lab, Goombalab. Apache-2.0; LICENSE.upstream.
Pinned source e9594ce1c732d97440f0332fdc43170a2294dbfa.
Only cancellation-prone cumulative-sum arithmetic changed; no forward changes.
Masks, state carry, dtypes and elementary approximations remain upstream.
"""

from typing import Tuple, Optional
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w, maxnreg=r)
        for s in [1, 2, 3]
        for w in [2, 4, 8]
        for r in [None, 128, 256]
    ],
    key=["CHUNK_SIZE", "HEADDIM_QK", "HEADDIM_V", "IS_VARLEN"]
)
@triton.jit
def mamba3_siso_bwd_kernel_dqkv(
    # Input tensors
    Q, K, V, DA_CS, DA_CS_SUM, QK_Dot, D, SSM_States, dO, d_OSSM_State, Cu_Seqlens, # dO is scaled with Z
    # Output tensors
    dQ, dK, dV, dADT, dQK_Dot, dD, d_ISSM_State, # dQK_Dot is scaled with scale
    # Strides for Inputs
    # Strides for Q: (batch, seqlen, nheads_qk, HEADDIM_QK)
    stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_qkdim,
    # Strides for K: (batch, seqlen, nheads_qk, HEADDIM_QK)
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim,
    # Strides for V: (batch, seqlen, nheads, HEADDIM_V)
    stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_vdim,
    # Strides for DA_CS: (batch, nheads, seqlen)
    stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seqlen,
    # Strides for DA_CS_SUM: (batch, nheads, nchunks)
    stride_da_cs_sum_batch, stride_da_cs_sum_head, stride_da_cs_sum_seqlen,
    # Strides for QK (QK dot products): (batch, nheads, nchunks*CHUNK_SIZE)
    stride_qk_dot_batch, stride_qk_dot_head, stride_qk_dot_seqlen,
    # Strides for D: (nheads,)
    stride_d_head,
    # Strides for SSM_States: (batch, nheads, HEADDIM_V, nchunks*HEADDIM_QK)
    stride_ssm_states_batch, stride_ssm_states_head, stride_ssm_states_vdim, stride_ssm_states_qkdim,
    # Strides for dO: (batch, seqlen, nheads, HEADDIM_V)
    stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_vdim,
    # Strides for d_OSSM_State: (num_sequences, nheads, HEADDIM_V, HEADDIM_QK)
    stride_d_ossm_state_batch, stride_d_ossm_state_head, stride_d_ossm_state_vdim, stride_d_ossm_state_qkdim,
    # Strides for Cu_Seqlens: (num_sequences + 1,)
    stride_cu_seqlen,
    # Strides for Outputs
    # Strides for dQ: (batch, seqlen, nheads, HEADDIM_QK)
    stride_dq_batch, stride_dq_seqlen, stride_dq_head, stride_dq_qkdim,
    # Strides for dK: (batch, seqlen, nheads, HEADDIM_QK)
    stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_qkdim,
    # Strides for dV: (batch, seqlen, nheads, HEADDIM_V)
    stride_dv_batch, stride_dv_seqlen, stride_dv_head, stride_dv_vdim,
    # Strides for dAdt: (batch, nheads, seqlen)
    stride_dadt_batch, stride_dadt_head, stride_dadt_seqlen,
    # Strides for dQK_dot: (batch, nheads, seqlen)
    stride_dQK_dot_batch, stride_dQK_dot_head, stride_dQK_dot_seqlen,
    # Strides for dD: (nheads,)
    stride_dd_batch, stride_dd_head,
    # Strides for d_ISSM_State: (num_sequences, nheads, HEADDIM_V, HEADDIM_QK)
    stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_vdim, stride_d_issm_state_qkdim,
    # Dimensions
    seqlen, nheads_qk, headdim_qk, headdim_v,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    RECOMPUTE_MASK: tl.constexpr,
    HAS_D_OSSM_STATE: tl.constexpr,
    RETURN_D_ISSM_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Backward kernel for Mamba-3 attention mechanism.

    Each program instance handles one (head, batch/seq) pair and iterates through
    all chunks in reverse order. This reverse iteration is necessary because
    state gradients flow backward through the sequence.

    The kernel computes:
        - dQ, dK: Gradients for query/key from both intra-chunk attention and inter-chunk states
        - dV: Gradient for values
        - dADT: Gradient for the decay parameter (A * dt)
        - dQK_Dot: Gradient for the QK dot product term
        - dD: Gradient for the skip connection (if present)
        - dISSM_State: Gradient for the input SSM state (if present)

    Grid:
        - Normal mode: (nheads, batch)
        - Varlen mode: (nheads, num_sequences)
    """
    # ==================== Program Indexing ====================
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    if IS_VARLEN:
        pid_seq = pid_batch
        pid_batch = 0
        cu_seqlen = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_next = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seqlen = cu_seqlen_next - cu_seqlen
        cu_chunks = pid_seq + cu_seqlen // CHUNK_SIZE
    else:
        cu_seqlen = 0
        cu_chunks = 0
        pid_seq = 0

    # Compute Q/K head index for GQA (grouped query attention)
    # Multiple output heads may share the same Q/K head
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)

    # Input Pointer Offsets
    q_offset = pid_batch * stride_q_batch + head_idx_qk * stride_q_head + IS_VARLEN * cu_seqlen * stride_q_seqlen
    k_offset = pid_batch * stride_k_batch + head_idx_qk * stride_k_head + IS_VARLEN * cu_seqlen * stride_k_seqlen
    v_offset = pid_batch * stride_v_batch + pid_head * stride_v_head + IS_VARLEN * cu_seqlen * stride_v_seqlen
    da_cs_offset = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head + IS_VARLEN * cu_seqlen * stride_da_cs_seqlen
    da_cs_sum_offset = pid_batch * stride_da_cs_sum_batch + pid_head * stride_da_cs_sum_head + IS_VARLEN * cu_chunks * stride_da_cs_sum_seqlen
    qk_dot_offset = pid_batch * stride_qk_dot_batch + pid_head * stride_qk_dot_head + IS_VARLEN * cu_seqlen * stride_qk_dot_seqlen
    ssm_states_offset = pid_batch * stride_ssm_states_batch + pid_head * stride_ssm_states_head + IS_VARLEN * cu_chunks * HEADDIM_QK * stride_ssm_states_qkdim
    do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head + IS_VARLEN * cu_seqlen * stride_do_seqlen
    if HAS_D_OSSM_STATE:
        d_ossm_state_offset = (pid_batch + IS_VARLEN * pid_seq) * stride_d_ossm_state_batch + pid_head * stride_d_ossm_state_head

    # Load skip connection value D if present
    if D is not None:
        D_offset = pid_head * stride_d_head
        D_val = tl.load(D + D_offset)

    # Output Pointer Offsets
    dq_offset = pid_batch * stride_dq_batch + pid_head * stride_dq_head + IS_VARLEN * cu_seqlen * stride_dq_seqlen
    dk_offset = pid_batch * stride_dk_batch + pid_head * stride_dk_head + IS_VARLEN * cu_seqlen * stride_dk_seqlen
    dv_offset = pid_batch * stride_dv_batch + pid_head * stride_dv_head + IS_VARLEN * cu_seqlen * stride_dv_seqlen
    dadt_offset = pid_batch * stride_dadt_batch + pid_head * stride_dadt_head + IS_VARLEN * cu_seqlen * stride_dadt_seqlen
    dQK_dot_offset = pid_batch * stride_dQK_dot_batch + pid_head * stride_dQK_dot_head + IS_VARLEN * cu_seqlen * stride_dQK_dot_seqlen

    if D is not None:
        dD_offset = pid_head * stride_dd_head + pid_batch * stride_dd_batch + IS_VARLEN * pid_seq * stride_dd_batch
        dD_acc = tl.zeros([1], dtype=tl.float32)

    if RETURN_D_ISSM_STATE:
        d_issm_state_offset = (pid_batch + IS_VARLEN * pid_seq) * stride_d_issm_state_batch + pid_head * stride_d_issm_state_head

    # Accumulates gradients flowing backward through states across chunks
    if HAS_D_OSSM_STATE:
        d_ssm_ptrs =  d_OSSM_State + d_ossm_state_offset + tl.arange(0, HEADDIM_V)[:, None] * stride_d_ossm_state_vdim + tl.arange(0, HEADDIM_QK)[None, :] * stride_d_ossm_state_qkdim
        d_ssm_states_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk)
        d_ssm_states_acc = tl.load(d_ssm_ptrs, mask=d_ssm_states_mask, other=0.0).to(tl.float32)
    else:
        d_ssm_states_acc = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    #  TMA Descriptors for Efficient Memory Access
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset,
        shape=[seqlen, headdim_qk],
        strides=[stride_q_seqlen, stride_q_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset,
        shape=[seqlen, headdim_qk],
        strides=[stride_k_seqlen, stride_k_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset,
        shape=[seqlen, headdim_v],
        strides=[stride_v_seqlen, stride_v_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    ssm_states_desc = tl.make_tensor_descriptor(
        SSM_States + ssm_states_offset,
        shape=[headdim_v, num_chunks * headdim_qk],
        strides=[stride_ssm_states_vdim, stride_ssm_states_qkdim],
        block_shape=[HEADDIM_V, HEADDIM_QK],
    )
    do_desc = tl.make_tensor_descriptor(
        dO + do_offset,
        shape=[seqlen, headdim_v],
        strides=[stride_do_seqlen, stride_do_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + dq_offset,
        shape=[seqlen, headdim_qk],
        strides=[stride_dq_seqlen, stride_dq_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    dk_desc = tl.make_tensor_descriptor(
        dK + dk_offset,
        shape=[seqlen, headdim_qk],
        strides=[stride_dk_seqlen, stride_dk_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + dv_offset,
        shape=[seqlen, headdim_v],
        strides=[stride_dv_seqlen, stride_dv_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )

    for chunk_idx_loop in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_idx_loop  # Reverse order for backward pass
        chunk_start = chunk_idx * CHUNK_SIZE

        # Sequence-length mask for non-TMA loads/stores
        offs_cs = chunk_start + tl.arange(0, CHUNK_SIZE)
        seq_mask = offs_cs < seqlen

        # ============================================================
        # Load Decay Values
        # We load these first to overlap computation with TMA loads
        # ============================================================
        da_cs_ptrs = DA_CS + da_cs_offset + offs_cs * stride_da_cs_seqlen
        da_cs = tl.load(da_cs_ptrs, mask=seq_mask, other=0.0)  # Cumulative decay within chunk: (CHUNK_SIZE,)

        da_cs_sum_ptrs = DA_CS_SUM + da_cs_sum_offset + chunk_idx * stride_da_cs_sum_seqlen
        da_cs_chunk_sum = tl.load(da_cs_sum_ptrs)  # Total decay for this chunk: scalar

        # ============================================================
        # Load Q, K, V, dO, SSM_States via TMA
        # ============================================================
        do_block = do_desc.load([chunk_start, 0])  # (CHUNK_SIZE, HEADDIM_V)
        v_block = v_desc.load([chunk_start, 0])    # (CHUNK_SIZE, HEADDIM_V)
        q_block = q_desc.load([chunk_start, 0])    # (CHUNK_SIZE, HEADDIM_QK)
        k_block = k_desc.load([chunk_start, 0])    # (CHUNK_SIZE, HEADDIM_QK)
        ssm_states_block = ssm_states_desc.load([0, chunk_idx * headdim_qk])  # (HEADDIM_V, HEADDIM_QK)

        # ============================================================
        # Compute Decay Scaling Factors
        # ============================================================
        # Reverse cumsum: how much decay from position i to end of chunk
        da_cs_rev = da_cs_chunk_sum - da_cs
        exp_da_cs_rev = tl.math.exp2(da_cs_rev)  # For scaling inter-chunk contributions
        exp_da_cs = tl.math.exp2(da_cs)          # For scaling intra-chunk contributions

        # Compute strictly causal mask with exponential decay (this is L^T)
        if not RECOMPUTE_MASK:
            causal_decay_mask = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0)),
                0.0
            )

        # ============================================================
        # Compute dADT Gradient (Part 1): From Intra-chunk Attention
        # This is register-heavy so we compute it early before spilling
        # ============================================================
        # Gradient contribution from (QK^T ⊙ L) V term
        dAinv = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T
        if RECOMPUTE_MASK:
            dAinv *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dAinv = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                dAinv,
                0.0
            )
        else:
            dAinv *= causal_decay_mask
        dAinv *= tl.dot(k_block, tl.trans(q_block))  # Element-wise with K @ Q^T
        dM_rev_vector = tl.sum(dAinv, axis=0) - tl.sum(dAinv, axis=1)  # (CHUNK_SIZE,)

        # ============================================================
        # Compute dK: Key Gradient
        # dK = (V @ dO^T ⊙ mask)^T @ Q + V @ dStates * scale
        # ============================================================
        # Intra-chunk: dP^T @ Q where dP = dO @ V^T ⊙ mask
        dp_t_block = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T: (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            dp_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dp_t_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                dp_t_block,
                0.0
            )
        else:
            dp_t_block *= causal_decay_mask

        acc_dk = tl.dot(dp_t_block.to(q_block.dtype), q_block)  # (CHUNK_SIZE, HEADDIM_QK)

        # Inter-chunk: gradient flowing through accumulated states
        acc_dk += tl.dot(v_block, d_ssm_states_acc.to(v_block.dtype)) * exp_da_cs_rev[:, None]

        dk_desc.store([chunk_start, 0], acc_dk)

        # ============================================================
        # Compute dQ: Query Gradient
        # dQ = (V @ dO^T ⊙ mask) @ K + dO @ States * scale
        # ============================================================
        # Intra-chunk: S^T @ K where S = V @ dO^T ⊙ mask
        s_block = tl.dot(v_block, tl.trans(do_block))  # (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            s_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            s_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                s_block,
                0.0
            )
        else:
            s_block *= causal_decay_mask

        acc_dq = tl.dot(tl.trans(s_block).to(k_block.dtype), k_block)  # (CHUNK_SIZE, HEADDIM_QK)

        # Inter-chunk: gradient through states from previous chunks
        acc_dq += tl.dot(do_block, ssm_states_block) * exp_da_cs[:, None]

        dq_desc.store([chunk_start, 0], acc_dq)

        # ============================================================
        # Compute dV: Value Gradient
        # dV = (K @ Q^T ⊙ mask) @ dO + K @ dStates^T * scale + dO * (D + qk_dot)
        # ============================================================
        # Intra-chunk: P^T @ dO where P = Q @ K^T ⊙ mask
        p_t_block = tl.dot(k_block, tl.trans(q_block))  # K @ Q^T: (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            p_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            p_t_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                p_t_block,
                0.0
            )
        else:
            p_t_block *= causal_decay_mask

        acc_dv = tl.dot(p_t_block.to(do_block.dtype), do_block)  # (CHUNK_SIZE, HEADDIM_V)

        # Inter-chunk: gradient through states
        acc_dv += tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype)) * exp_da_cs_rev[:, None]

        # Skip connection gradient contribution
        # Load dO again with volatile to avoid cache conflicts
        dO_reloaded = tl.load(
            dO + do_offset + offs_cs[:, None] * stride_do_seqlen +
            tl.arange(0, HEADDIM_V)[None, :] * stride_do_vdim,
            mask=seq_mask[:, None] & (tl.arange(0, HEADDIM_V)[None, :] < headdim_v),
            other=0.0,
            volatile=True
        )

        qk_dot = tl.load(QK_Dot + qk_dot_offset + offs_cs * stride_qk_dot_seqlen, mask=seq_mask, other=0.0)
        if D is not None:
            acc_dv += dO_reloaded * (D_val + qk_dot[:, None])
        else:
            acc_dv += dO_reloaded * qk_dot[:, None]

        dv_desc.store([chunk_start, 0], acc_dv)

        # ============================================================
        # Compute dQK_Dot and dD: Skip Connection Gradients
        # ============================================================
        v_block_reloaded = tl.load(
            V + v_offset + offs_cs[:, None] * stride_v_seqlen +
            tl.arange(0, HEADDIM_V)[None, :] * stride_v_vdim,
            mask=seq_mask[:, None] & (tl.arange(0, HEADDIM_V)[None, :] < headdim_v),
            other=0.0,
            volatile=True
        )

        # dQK_dot = sum_v(dO * V) for each position
        dQK_dot_block = tl.dot(
            dO_reloaded * v_block_reloaded,
            tl.full([HEADDIM_V, 1], 1, dtype=dO_reloaded.dtype)
        )

        tl.store(
            dQK_Dot + dQK_dot_offset + offs_cs * stride_dQK_dot_seqlen,
            dQK_dot_block.reshape(CHUNK_SIZE),
            mask=seq_mask
        )

        # Accumulate dD gradient
        if D is not None:
            dD_acc += tl.dot(
                tl.full([1, CHUNK_SIZE], 1, dtype=tl.float32),
                dQK_dot_block
            ).reshape(1)

        # ============================================================
        # Compute dADT Gradient (Part 2): From Inter-chunk States
        # ============================================================
        # Gradient from Q @ States^T term
        QS = tl.dot(q_block, tl.trans(ssm_states_block))  # (CHUNK_SIZE, HEADDIM_V)
        dM_rev_vector += tl.sum(QS * dO_reloaded, axis=1) * exp_da_cs  # (CHUNK_SIZE,)

        # ============================================================
        # Compute dADT Gradient (Part 3): From State Accumulation
        # ============================================================
        # Gradient flowing through d_ssm_states_acc @ SSM_States
        SSM_States_ptrs = (SSM_States + ssm_states_offset +
                tl.arange(0, HEADDIM_V)[:, None] * stride_ssm_states_vdim +
                (chunk_idx * headdim_qk + tl.arange(0, HEADDIM_QK)[None, :]) * stride_ssm_states_qkdim)
        SSM_States_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & ((chunk_idx * headdim_qk + tl.arange(0, HEADDIM_QK)[None, :]) < num_chunks * headdim_qk)

        SSM_States_reloaded = tl.load(SSM_States_ptrs, volatile=True, mask=SSM_States_mask)  # (HEADDIM_V, HEADDIM_QK)
        dM_scalar = tl.sum(SSM_States_reloaded * d_ssm_states_acc) * tl.math.exp2(da_cs_chunk_sum)

        # ============================================================
        # Compute dADT Gradient (Part 4): From K @ dStates
        # ============================================================
        dSK = tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype))  # (CHUNK_SIZE, HEADDIM_V)
        dM_vector = tl.sum(dSK * v_block_reloaded, axis=1) * exp_da_cs_rev  # (CHUNK_SIZE,)

        # ============================================================
        # Combine dADT Gradient Components via Reverse Cumsum
        # ============================================================
        # r_i + sum(r) + c + sum_{j<=i}(v_j-r_j) - v_i
        # = sum_{j>=i} r_j + sum_{j<i} v_j + c. No total-minus-prefix.
        indices = tl.arange(0, CHUNK_SIZE)
        prefix_v = tl.cumsum(dM_vector, axis=0)
        exclusive_v = tl.where(indices > 0,
            tl.gather(prefix_v, tl.maximum(indices - 1, 0), axis=0), 0.0)
        dM_rev_vector = tl.cumsum(dM_rev_vector, axis=0, reverse=True) + exclusive_v + dM_scalar

        # Store dADT
        dadt_ptrs = dADT + dadt_offset + offs_cs * stride_dadt_seqlen
        tl.store(dadt_ptrs, dM_rev_vector, mask=seq_mask)

        # ============================================================
        # Accumulate State Gradients for Previous Chunks
        # ============================================================
        dO_reloaded *= exp_da_cs[:, None]
        d_ssm_states_acc = (tl.math.exp2(da_cs_chunk_sum) * d_ssm_states_acc +
                       tl.dot(tl.trans(dO_reloaded).to(q_block.dtype), q_block))

    # Store Final dD Gradient
    if D is not None:
        tl.store(dD + dD_offset + tl.arange(0, 1), dD_acc)

    # Store d_ISSM_State
    if RETURN_D_ISSM_STATE:
        d_ISSM_State_ptrs = d_ISSM_State + d_issm_state_offset + tl.arange(0, HEADDIM_V)[:, None] * stride_d_issm_state_vdim + tl.arange(0, HEADDIM_QK)[None, :] * stride_d_issm_state_qkdim
        d_ISSM_State_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk)
        tl.store(d_ISSM_State_ptrs, d_ssm_states_acc, mask=d_ISSM_State_mask)


def compute_dqkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    da_cs: torch.Tensor,
    da_cs_sum: torch.Tensor,
    qk_dot: torch.Tensor,
    SSM_States: torch.Tensor,
    do: torch.Tensor,
    d_ossm_state: Optional[torch.Tensor] = None,
    d_ov_state: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    has_input_state: bool = False,
    Cu_Seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute gradients dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, d_issm_state for Mamba-3 backward pass.

    This kernel operates on the rotated/scaled Q and K tensors (Q_mid, K_mid from forward).

    Args:
        q: Rotated query tensor Q_mid (batch, seqlen, headdim_qk, headdim_qk)
        k: Rotated+scaled key tensor K_mid (batch, seqlen, headdim_qk, headdim_qk)
        v: Value tensor (batch, seqlen, nheads, headdim_v)
        da_cs: Cumulative decay per chunk (batch, nheads, seqlen)
        da_cs_sum: Sum of decay per chunk (batch, nheads, nchunks)
        qk_dot: QK dot products from forward (batch, nheads, seqlen)
        SSM_States: SSM states from forward pass (batch, nheads, headdim_v, nchunks * headdim_qk)
        do: Output gradient, possibly scaled by Z (batch, seqlen, nheads, headdim_v)
        d_ossm_state: Gradient of output SSM states (num_sequences, nheads, headdim_v, headdim_qk)
        d_ov_state: Gradient of output V state (num_sequences, nheads, headdim_v) - added to last token of dV
        D: Optional skip connection weight (nheads,)
        chunk_size: Chunk size (default: 64)
        has_input_state: Whether to compute gradient for input states

    Returns:
        Tuple of (dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, d_issm_state)
        where d_issm_state is None if has_input_state=False
    """
    batch, seqlen, nheads_qk, headdim_qk = q.shape
    _, _, nheads, headdim_v = v.shape
    is_varlen = Cu_Seqlens is not None

    if is_varlen:
        num_sequences = Cu_Seqlens.shape[0] - 1
        assert batch == 1
        nchunks = num_sequences + seqlen // chunk_size
    else:
        num_sequences = batch
        nchunks = (seqlen + chunk_size - 1) // chunk_size

    assert nheads % nheads_qk == 0, "nheads must be divisible by nheads_qk (for GQA support)"
    assert q.is_cuda and k.is_cuda and v.is_cuda and da_cs.is_cuda and da_cs_sum.is_cuda and do.is_cuda, "All tensors must be on CUDA"

    assert k.shape == q.shape
    assert v.shape == (batch, seqlen, nheads, headdim_v)
    assert da_cs.shape == (batch, nheads, seqlen)
    assert da_cs_sum.shape == (batch, nheads, nchunks)
    assert qk_dot.shape == (batch, nheads, seqlen)
    assert SSM_States.shape == (batch, nheads, headdim_v, nchunks * headdim_qk)
    assert do.shape == (batch, seqlen, nheads, headdim_v)
    assert d_ossm_state is None or d_ossm_state.shape == (num_sequences, nheads, headdim_v, headdim_qk)
    assert d_ov_state is None or d_ov_state.shape == (num_sequences, nheads, headdim_v)
    if D is not None:
        assert D.shape == (nheads,)

    # Ensure all tensors satisfy TMA alignment constraints.
    #
    # TMA 2D requires the global stride (seqlen dimension, in bytes) to be a
    # multiple of 16.  For bfloat16 data this means stride_seqlen % 8 == 0.
    # Tensors that come from saved ctx.saved_tensors in the backward (e.g. V
    # extracted from a fused projection) can be non-contiguous with strides
    # that violate this constraint.  The safest fix is to make them contiguous.
    #
    # We always use .contiguous() for tensors that are passed through TMA
    # descriptors; other tensors just need innermost stride == 1.
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if da_cs.stride(-1) != 1:
        da_cs = da_cs.contiguous()
    if da_cs_sum.stride(-1) != 1:
        da_cs_sum = da_cs_sum.contiguous()
    if qk_dot.stride(-1) != 1:
        qk_dot = qk_dot.contiguous()
    if not SSM_States.is_contiguous():
        SSM_States = SSM_States.contiguous()
    if not do.is_contiguous():
        do = do.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if d_ossm_state is not None and not d_ossm_state.is_contiguous():
        d_ossm_state = d_ossm_state.contiguous()
    if d_ov_state is not None and not d_ov_state.is_contiguous():
        d_ov_state = d_ov_state.contiguous()

    # Allocate output tensors
    dq = torch.empty((batch, seqlen, nheads, headdim_qk), dtype=q.dtype, device=q.device)
    dk = torch.empty((batch, seqlen, nheads, headdim_qk), dtype=k.dtype, device=k.device)
    dv = torch.empty((batch, seqlen, nheads, headdim_v), dtype=v.dtype, device=v.device)
    dAdt = torch.empty_like(da_cs)
    dQK = torch.empty_like(da_cs)
    dD = torch.empty((num_sequences, nheads), dtype=torch.float32, device=q.device) if D is not None else None
    d_issm_state = torch.empty((num_sequences, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=q.device) if has_input_state else None

    # Round up head dimensions to power of 2 for efficient loading
    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    HEADDIM_V = triton.next_power_of_2(headdim_v)

    # Grid: each program handles one (head, batch/num_sequences) pair
    if is_varlen:
        grid = (nheads, num_sequences)
    else:
        grid = (nheads, batch)

    # Launch kernel
    mamba3_siso_bwd_kernel_dqkv[grid](
        q, k, v, da_cs, da_cs_sum, qk_dot, D, SSM_States, do, d_ossm_state, Cu_Seqlens,
        dq, dk, dv, dAdt, dQK, dD, d_issm_state,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # DA_CS strides
        da_cs.stride(0), da_cs.stride(1), da_cs.stride(2),
        # DA_CS_SUM strides
        da_cs_sum.stride(0), da_cs_sum.stride(1), da_cs_sum.stride(2),
        # QK_Dot strides
        qk_dot.stride(0), qk_dot.stride(1), qk_dot.stride(2),
        # D stride
        D.stride(0) if D is not None else 0,
        # SSM_States strides: (batch, nheads, headdim_v, nchunks*headdim_qk)
        SSM_States.stride(0), SSM_States.stride(1), SSM_States.stride(2),
        SSM_States.stride(3),
        # dO strides
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        # d_ossm_state strides
        d_ossm_state.stride(0) if d_ossm_state is not None else 0,
        d_ossm_state.stride(1) if d_ossm_state is not None else 0,
        d_ossm_state.stride(2) if d_ossm_state is not None else 0,
        d_ossm_state.stride(3) if d_ossm_state is not None else 0,
        # Cu_Seqlens strides
        Cu_Seqlens.stride(0) if Cu_Seqlens is not None else 0,
        # dQ strides
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        # dK strides
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        # dV strides
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        # dAdt strides
        dAdt.stride(0), dAdt.stride(1), dAdt.stride(2),
        # dQK strides
        dQK.stride(0), dQK.stride(1), dQK.stride(2),
        # dD strides
        dD.stride(0) if D is not None else 0,
        dD.stride(1) if D is not None else 0,
        # d_issm_state strides
        d_issm_state.stride(0) if d_issm_state is not None else 0,
        d_issm_state.stride(1) if d_issm_state is not None else 0,
        d_issm_state.stride(2) if d_issm_state is not None else 0,
        d_issm_state.stride(3) if d_issm_state is not None else 0,
        # Dimensions
        seqlen, nheads_qk, headdim_qk, headdim_v,
        # Compile-time constants
        CHUNK_SIZE=chunk_size,
        HEADDIM_QK=HEADDIM_QK,
        HEADDIM_V=HEADDIM_V,
        RECOMPUTE_MASK=False,
        HAS_D_OSSM_STATE=d_ossm_state is not None,
        RETURN_D_ISSM_STATE=has_input_state,
        IS_VARLEN=is_varlen,
    )

    # Add output V state gradients to the last token
    if d_ov_state is not None:
        if is_varlen:
            last_token_idx = Cu_Seqlens[1:] - 1
            dv[0, last_token_idx] += d_ov_state
        else:
            dv[:, -1, :, :] += d_ov_state

    dD = dD.sum(dim=0) if dD is not None else None
    return dq, dk, dv, dAdt, dQK, dD, d_issm_state
