import copy
import importlib.util
import math

import pytest
import torch

from experiments.mamba3_context_time.temporal import ContextTime, TEMPORAL_COUNTS, gap_tau, mix_log_scales
from experiments.mamba3_context_time.diagnostics import ContextDiagnostics
from experiments.mamba3_timeaware.time_inputs import TimeCalibrator, history_gaps
from experiments.mamba3_time_mechanisms.time_mechanisms import TimeMechanisms


def inputs():
    u = torch.randn(3, 4, 64)
    valid = torch.tensor([[1,1,1,0], [1,0,0,0], [1,1,1,1]], dtype=torch.bool)
    times = torch.tensor([[1e12,1e12,1e12+838393,float('nan')],
                          [1e12, float('nan'),0,0], [1e12,1e12+2,1e12+10,1e12+1e8]], dtype=torch.float64)
    return u, *history_gaps(times, valid)


def nonzero(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'last.' in name:
                p.copy_(torch.linspace(-.12, .17, p.numel()).reshape_as(p))


@pytest.mark.parametrize('mode', ('dense11','dense12','uniform','routed'))
def test_counts_identity_masks_bounds_extremes(mode):
    model = ContextTime(mode)
    assert sum(p.numel() for p in model.parameters()) == TEMPORAL_COUNTS[mode]
    u,gaps,active = inputs()
    decay,scan,details = model(u,gaps,active)
    assert decay.shape == scan.shape == (3,4,2)
    assert decay.dtype == scan.dtype == torch.float32
    assert torch.equal(decay,torch.ones_like(decay)) and torch.equal(scan,torch.ones_like(scan))
    nonzero(model)
    gaps[2,-1] = torch.finfo(torch.float64).max
    decay,scan,_ = model(u,gaps,active)
    for scale in (decay,scan):
        assert torch.isfinite(scale).all() and (scale>=.5).all() and (scale<=2).all()
        assert torch.equal(scale[~active],torch.ones_like(scale[~active]))
    assert active[0,1] and gaps[0,1] == 0
    assert not torch.equal(scan[0,1],torch.ones(2))
    if mode == 'uniform':
        assert not hasattr(model,'router')
        assert all(torch.equal(a,b) for a,b in zip(model(u,gaps,active)[:2],model(u+11,gaps,active)[:2]))
    elif mode == 'routed':
        torch.testing.assert_close(details['probabilities'].sum(-1),torch.ones(3,4))


@pytest.mark.parametrize('mode', ('', 'shared', 'output_moe', 'dense13', 'separate_replay'))
def test_mode_allowlist(mode):
    with pytest.raises(ValueError): ContextTime(mode)


def test_parameter_independence_and_paired_initialization():
    torch.manual_seed(2026); uniform=ContextTime('uniform')
    torch.manual_seed(2026); routed=ContextTime('routed')
    assert all(torch.equal(v,routed.bank.state_dict()[k]) for k,v in uniform.bank.state_dict().items())
    firsts=[pair[p].first.weight for pair in uniform.bank for p in ('decay','scan')]
    assert len({p.data_ptr() for p in firsts}) == 8
    assert all(not torch.equal(a,b) for i,a in enumerate(firsts) for b in firsts[i+1:])


def test_tau_matches_frozen_float64_conversion():
    u,gaps,active=inputs(); gaps[2,-1]=torch.finfo(torch.float64).max
    frozen=TimeCalibrator(2,838393); nonzero(frozen)
    actual=torch.exp(math.log(2)*torch.tanh(frozen.last(torch.nn.functional.silu(frozen.first(gap_tau(gaps))))))
    expected=frozen(gaps,active)
    assert torch.equal(actual[active],expected[active])


@pytest.mark.parametrize('experts', (1,4))
def test_separate_reductions_and_gradient_scaling(experts):
    ref=TimeMechanisms('separate'); nonzero(ref)
    model=ContextTime('uniform',experts=experts)
    for pair in model.bank:
        for p in ('decay','scan'): pair[p].load_state_dict(ref.calibrators[p].state_dict())
    u,gaps,active=inputs()
    a=ref(gaps,active); b=model(u,gaps,active)[:2]
    for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
    sum(x.square().sum() for x in a).backward()
    sum(x.square().sum() for x in b).backward()
    for p in ('decay','scan'):
        for key,parameter in ref.calibrators[p].named_parameters():
            grads=[dict(pair[p].named_parameters())[key].grad for pair in model.bank]
            torch.testing.assert_close(sum(grads),parameter.grad,atol=1e-6,rtol=1e-5)
            for g in grads: torch.testing.assert_close(g,parameter.grad/experts,atol=1e-6,rtol=1e-5)


def test_log_mixture_uniform_router_and_router_learning():
    torch.manual_seed(2026); uniform=ContextTime('uniform'); nonzero(uniform)
    with torch.no_grad():
        for i,pair in enumerate(uniform.bank): pair['decay'].last.bias.add_(i*.15)
    routed=ContextTime('routed'); routed.bank.load_state_dict(uniform.bank.state_dict())
    with torch.no_grad(): routed.router.weight.zero_(); routed.router.bias.zero_()
    u,gaps,active=inputs(); a=uniform(u,gaps,active); b=routed(u,gaps,active)
    for x,y in zip(a[:2],b[:2]): torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
    expected=a[2]['log_experts'].mean(-3).exp()
    torch.testing.assert_close(a[0][active],expected[...,0,:][active])
    wrong=a[2]['log_experts'].exp().mean(-3)[...,0,:]
    assert not torch.allclose(a[0][active],wrong[active],atol=1e-6,rtol=1e-5)
    b[0].square().sum().backward()
    assert routed.router.weight.grad.abs().sum()>0
    fresh=ContextTime('routed'); y=fresh(u,gaps,active)
    y[0].sum().backward()
    assert torch.equal(fresh.router.weight.grad,torch.zeros_like(fresh.router.weight))
    assert sum(p.grad.abs().sum() for p in fresh.bank.parameters())>0


def test_float64_log_mixture_gradcheck():
    q=torch.randn(1,2,4,2,2,dtype=torch.float64,requires_grad=True)*.1
    logits=torch.randn(1,2,4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda q,z:mix_log_scales(q,z.softmax(-1)).exp(),(q,logits))


@pytest.mark.parametrize('mode', ('dense11','dense12','uniform','routed'))
def test_causal_prefix_translation_and_diagnostics(mode):
    model=ContextTime(mode); nonzero(model)
    if mode=='routed':
        with torch.no_grad():
            for i,pair in enumerate(model.bank): pair['scan'].last.bias.add_(i*.2)
    u,gaps,active=inputs()
    y=model(u,gaps,active)
    changed_u=u.clone(); changed_u[:,2:]+=5
    changed_gaps=gaps.clone(); changed_gaps[:,2:]+=100
    z=model(changed_u,changed_gaps,active)
    for a,b in zip(y[:2],z[:2]): assert torch.equal(a[:,:2],b[:,:2])
    times=torch.tensor([[1e12,1e12,1e12+1000,0]],dtype=torch.float64)
    valid=torch.tensor([[True,True,True,False]])
    a=history_gaps(times,valid); b=history_gaps(times+8000000,valid)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    if mode!='uniform':
        assert any(not torch.equal(a,b) for a,b in zip(y[:2],model(u+5,gaps,active)[:2]))
    diag=ContextDiagnostics(mode,capacity=3)
    state=torch.random.get_rng_state()
    diag.update(*y[:2],active,y[2],gaps)
    assert torch.equal(state,torch.random.get_rng_state())
    assert not diag.sample.requires_grad and diag.sample.grad_fn is None
    assert diag.result()['active_history_gaps']==int(active.sum())
    assert len(diag.sample)<=3
    for a,b in zip(y[:2],model(u,gaps,active)[:2]): assert torch.equal(a,b)
    if mode.startswith('dense'): assert 'routing' not in diag.result()
    if mode=='uniform': assert diag.result()['routing']['learned'] is False


@pytest.mark.skipif(importlib.util.find_spec('mamba_ssm') is None or not torch.cuda.is_available(),
                    reason='Real pinned CUDA full-model gate runs only inside allocation; CPU is not GPU evidence')
def test_real_cuda_cases():
    from experiments.mamba3_context_time.gpu_checks import case,all_pass
    for training in (False,True):
        for length in (50,64): assert all_pass(case(training,length)['checks'])
