"""Isolated diagnostic of the chunk64 reverse-scan/carry, without elementary math."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan(X, Y, L: tl.constexpr, CHANNELS: tl.constexpr):
    c = tl.program_id(0)
    offsets = tl.arange(0,64)
    carry = 0.0
    for block in range(tl.cdiv(L,64)-1,-1,-1):
        t = block*64+offsets
        x = tl.load(X+t*CHANNELS+c, t<L, 0.)
        y = tl.cumsum(x,axis=0,reverse=True)+carry
        tl.store(Y+t*CHANNELS+c,y,t<L)
        carry += tl.sum(x,axis=0)


def reverse_scan(values):
    values = values.contiguous()
    if values.shape[0] != 1 or values.dtype != torch.float32:
        raise ValueError("Fixed isolated scan fixture only")
    out = torch.empty_like(values)
    _scan[(values.shape[2]*values.shape[3],)](values,out,values.shape[1],values.shape[2]*values.shape[3])
    return out
