import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from experiments.mamba3_time_mechanisms.time_mechanisms import MODES, COUNTS, TimeMechanisms, mechanism_dt
from experiments.mamba3_timeaware.time_inputs import history_gaps, condition_dt
from experiments.mamba3_time_mechanisms.diagnostics import TemporalDiagnostics


@pytest.mark.parametrize('mode', MODES)
def test_counts_initial_identity_and_formulas(mode):
    module = TimeMechanisms(mode)
    gaps = torch.tensor([[0.,3.,900000.,0.]])
    active = torch.tensor([[False,True,True,False]])
    decay, scan = module(gaps,active)
    assert torch.equal(decay,torch.ones_like(decay)) and torch.equal(scan,torch.ones_like(scan))
    assert sum(p.numel() for p in module.parameters()) + 610440 == COUNTS[mode]
    dt, a = torch.rand(1,4,2), -torch.rand(1,4,2)
    actual_dt, actual_adt = mechanism_dt(dt,a,decay,scan)
    assert torch.equal(actual_dt,dt) and torch.equal(actual_adt,a*dt)
    with torch.no_grad():
        for name,c in module.calibrators.items():
            c.last.bias.fill_(.5 if name != 'scan' else -.4)
    decay, scan = module(gaps,active)
    actual_dt, actual_adt = mechanism_dt(dt,a,decay,scan)
    assert torch.equal(actual_dt,dt*scan.to(dt.dtype))
    assert torch.equal(actual_adt,a*(dt*decay.to(dt.dtype)))
    if mode in ('vanilla','decay_only'):
        assert torch.equal(actual_dt,dt)
    if mode in ('vanilla','scan_only'):
        assert torch.equal(actual_adt,a*dt)
    if mode == 'shared':
        assert decay is scan and len(module.calibrators) == 1
        old_dt,old_adt = condition_dt(dt,a,decay)
        assert torch.equal(actual_dt,old_dt) and torch.equal(actual_adt,old_adt)
    assert torch.equal(decay[:,[0,3]],torch.ones_like(decay[:,[0,3]]))
    assert torch.equal(scan[:,[0,3]],torch.ones_like(scan[:,[0,3]]))


def test_separate_independence_and_gradients():
    module = TimeMechanisms('separate')
    d,s = module.calibrators.values()
    assert d is not s
    for left,right in zip(d.parameters(),s.parameters()):
        assert left is not right and left.data_ptr() != right.data_ptr()
        assert torch.equal(left,right)
    gaps,active = torch.tensor([[0.,100.,1e6]]),torch.tensor([[False,True,True]])
    before = module(gaps,active)
    with torch.no_grad(): d.last.bias.add_(.2)
    after = module(gaps,active)
    assert not torch.equal(before[0],after[0]) and torch.equal(before[1],after[1])
    with torch.no_grad(): s.last.bias.sub_(.3)
    final = module(gaps,active)
    assert torch.equal(after[0],final[0]) and not torch.equal(after[1],final[1])
    dt = torch.rand(1,3,2,requires_grad=True)
    scan,adt = mechanism_dt(dt,-torch.ones_like(dt),*final)
    (scan.square().sum() + adt.square().sum()).backward()
    for branch in (d,s):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        assert branch.last.weight.grad.abs().sum() > 0
    assert torch.isfinite(dt.grad).all()


def test_causal_history_contract():
    times = torch.tensor([[100.,90.,200.,float('nan')]],dtype=torch.float64)
    valid = torch.tensor([[True,True,True,False]])
    gaps,active = history_gaps(times,valid)
    assert gaps.tolist() == [[0.,0.,110.,0.]]
    assert active.tolist() == [[False,True,True,False]]
    changed = times.clone(); changed[0,2] = 999999
    assert torch.equal(history_gaps(changed,valid)[0][:,:2],gaps[:,:2])
    for mode in MODES:
        scales = TimeMechanisms(mode)(gaps,active)
        assert all(torch.isfinite(s).all() for s in scales)


@pytest.mark.parametrize('bad', ('decay','periodicity','',None))
def test_allowlist(bad):
    with pytest.raises(ValueError): TimeMechanisms(bad)


@pytest.mark.parametrize('value', (0.,-1.,float('nan'),float('inf')))
def test_invalid_scales(value):
    dt = torch.ones(1,2,2)
    with pytest.raises(ValueError): mechanism_dt(dt,-dt,dt,dt*value)


@pytest.mark.parametrize('mode',MODES)
def test_cpu_wrapper_identity_and_layer_sharing(mode):
    # Actual forward body, with an explicit CPU algebraic scan double, NOT GPU equivalence.
    path = Path(__file__).resolve().parents[1] / 'model.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef))
    forward = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == 'forward')
    calls = []
    def scan_double(mixer,u,decay,scan):
        calls.append((decay,scan))
        return u * (decay.mean(-1,keepdim=True)+scan.mean(-1,keepdim=True))/2
    ns = dict(torch=torch,history_gaps=history_gaps,mechanism_mamba3_forward=scan_double)
    exec(compile(ast.Module(body=[forward],type_ignores=[]),str(path),'exec'),ns)
    layers = [SimpleNamespace(mixer=object(),norm1=nn.Identity(),mixer_dtype=torch.float32,
                dropout1=nn.Identity(),use_ffn=True,norm2=nn.Identity(),ffn=nn.Linear(64,64),dropout2=nn.Identity()) for _ in range(2)]
    fake = SimpleNamespace(mechanisms=TimeMechanisms(mode),diagnostic_collector=None,
        item_embedding=nn.Embedding(16,64),input_norm=nn.Identity(),input_dropout=nn.Identity(),
        layers=layers,output_norm=nn.Identity(),gather_indexes=lambda h,i:h[torch.arange(len(h)),i])
    items,lengths = torch.tensor([[1,2,3,0]]),torch.tensor([3])
    times = torch.tensor([[100.,150.,900000.,0.]])
    actual = ns['forward'](fake,items,lengths,times)
    hidden = fake.item_embedding(items)
    for layer in layers:
        hidden = hidden + hidden
        hidden = hidden + layer.ffn(hidden)
    assert torch.equal(actual,hidden[:,2])
    assert calls[0][0] is calls[1][0] and calls[0][1] is calls[1][1]
    changed = ns['forward'](fake,items,lengths,times+1e10)
    assert torch.equal(changed,actual)


def test_diagnostics_are_detached_bounded_and_do_not_consume_training_rng():
    d = torch.exp(torch.randn(3,10,2,requires_grad=True)*.1)
    s = torch.exp(torch.randn(3,10,2,requires_grad=True)*.1)
    active = torch.ones(3,10,dtype=torch.bool); active[:,0] = False
    diag = TemporalDiagnostics('separate',capacity=8)
    state = torch.random.get_rng_state()
    diag.update(d,s,active)
    assert torch.equal(state,torch.random.get_rng_state())
    assert not diag.sample.requires_grad and diag.sample.grad_fn is None
    result = diag.result()
    assert result['active_history_gaps']==27 and result['sample_size']==8
    torch.testing.assert_close(torch.tensor(result['scales']['decay']['mean']),d.detach()[active].mean(0))
    ones = TemporalDiagnostics('separate'); ones.update(torch.ones_like(d),torch.ones_like(s),active)
    assert ones.result()['separate']['log_correlation'] is None


def test_shared_dt_backward_exactly_matches_frozen_graph():
    dt = torch.rand(2,5,2,requires_grad=True)
    a = (-torch.rand_like(dt)).requires_grad_()
    scale = torch.rand_like(dt).add(.5).requires_grad_()
    old = condition_dt(dt,a,scale)
    new = mechanism_dt(dt,a,scale,scale)
    signal = torch.randn_like(dt)
    grad_old = torch.autograd.grad((old[0]*signal+old[1].square()).sum(),(dt,a,scale),retain_graph=True)
    grad_new = torch.autograd.grad((new[0]*signal+new[1].square()).sum(),(dt,a,scale))
    assert all(torch.equal(x,y) for x,y in zip(grad_old,grad_new))


def test_actual_forward_wires_distinct_kernel_arguments(monkeypatch):
    import sys
    from types import ModuleType
    from experiments.mamba3_time_mechanisms.time_mamba3 import mechanism_mamba3_forward
    # Kernel spy checks actual wrapper plumbing, not numerical equivalence to Triton.
    module = ModuleType('mamba_ssm.modules.mamba3')
    module.heavy_tail_activation = torch.nn.functional.softplus
    captured = {}
    def kernel(**kwargs):
        captured.update(kwargs)
        return kwargs['V']*(kwargs['DT']+kwargs['ADT']).transpose(1,2).unsqueeze(-1)
    module.mamba3_siso_combined = kernel
    monkeypatch.setitem(sys.modules,'mamba_ssm.modules.mamba3',module)
    mixer = SimpleNamespace(is_mimo=False,nheads=2,d_inner=128,d_state=4,num_bc_heads=1,
        num_rope_angles=2,headdim=64,A_floor=.01,dt_bias=torch.zeros(2),B_norm=nn.Identity(),
        C_norm=nn.Identity(),B_bias=torch.zeros(2,1,4),C_bias=torch.zeros(2,1,4),D=torch.ones(2),
        is_outproj_norm=False,chunk_size=64,in_proj=nn.Linear(64,272),out_proj=nn.Linear(128,64))
    u=torch.randn(2,5,64,requires_grad=True)
    decay=torch.full((2,5,2),1.5,requires_grad=True)
    scan=torch.full((2,5,2),.7,requires_grad=True)
    result=mechanism_mamba3_forward(mixer,u,decay,scan)
    projected=mixer.in_proj(u)
    dt=torch.nn.functional.softplus(projected[...,264:266]+mixer.dt_bias)
    a=-torch.nn.functional.softplus(projected[...,266:268])
    a=a.clamp(max=-.01)
    assert torch.equal(captured['DT'],(dt*scan).transpose(1,2))
    assert torch.equal(captured['ADT'],(a*(dt*decay)).transpose(1,2))
    result.square().mean().backward()
    for value in (u,decay,scan):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
