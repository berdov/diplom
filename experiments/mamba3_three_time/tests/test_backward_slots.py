"""CPU adapter wiring regression with sentinel derivatives, not GPU evidence."""

import sys
from types import SimpleNamespace
import torch
from experiments.mamba3_three_time.kernels import SISO, MIMO


def number(value):
    return torch.tensor(float(value))


def test_siso_returns_separate_slots_and_all_other_gradients(monkeypatch):
    saved = tuple(number(i+100) for i in range(21))
    observed = {}
    def angle(**kwargs):
        observed["phase"] = kwargs["dt"]
        return number(10), number(6), None
    def write(**kwargs):
        observed["write"] = kwargs["dt"]
        return number(5),number(7),None,None,None
    monkeypatch.setitem(sys.modules,"mamba_ssm.ops.triton.mamba3.angle_dt",SimpleNamespace(angle_dt_bwd=angle))
    monkeypatch.setitem(sys.modules,"mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd",SimpleNamespace(
        compute_dzdo=lambda *a,**k:(number(12),number(999)),
        compute_dqkv=lambda **k:tuple(number(i) for i in (91,92,3,4,93,11))+(None,),
        compute_dqktheta=lambda **k:tuple(number(i) for i in (1,2,8,9,94,95,96)),
        compute_ddt_dtrap_dinput_states=write))
    returned = SISO.backward(SimpleNamespace(saved_tensors=saved,chunk=64),number(1000))
    assert [v.item() for v in returned[:-1]] == list(range(1,13))
    assert returned[-1] is None
    assert observed["write"] is saved[3] and observed["phase"] is saved[4]


def test_mimo_returns_separate_slots_and_rank_gradients(monkeypatch):
    saved = tuple(number(i+100) for i in range(16))
    observed = {}
    def angle(**kwargs):
        observed["phase"] = kwargs["dt"]
        return number(10),number(6),None
    def backward(*args,**kwargs):
        observed["write"] = args[13]
        return tuple(number(i) for i in (1,2,3,4,5,7,8,9,13,14,15,99,11,12))+(None,)
    monkeypatch.setitem(sys.modules,"mamba_ssm.ops.triton.mamba3.angle_dt",SimpleNamespace(angle_dt_bwd=angle))
    monkeypatch.setitem(sys.modules,"mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils",SimpleNamespace(
        compute_dacs_segsum_triton=lambda *a:(number(1),number(2),number(3))))
    monkeypatch.setitem(sys.modules,"mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd",SimpleNamespace(mamba_mimo_bwd_combined=backward))
    returned = MIMO.backward(SimpleNamespace(saved_tensors=saved,chunk=16),number(1000))
    assert [v.item() for v in returned[:-1]] == list(range(1,16))
    assert returned[-1] is None
    assert observed["write"] is saved[4] and observed["phase"] is saved[5]
