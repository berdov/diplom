import pytest
import torch
from experiments.mamba3_three_time.calibrators import ThreeTimes, split_dt
from experiments.mamba3_timeaware.time_inputs import history_gaps
from experiments.mamba3_three_time.kernels import dense_only


@pytest.mark.parametrize("mode,count",[("base",0),("dual",132),("triple",198)])
def test_rng_counts_identity(mode,count):
    state = torch.get_rng_state().clone()
    module = ThreeTimes(mode)
    assert torch.equal(state,torch.get_rng_state())
    assert sum(p.numel() for p in module.parameters()) == count
    times = torch.tensor([[1.6e12,1.6e12,1.6e12+1,0]],dtype=torch.float64)
    valid = torch.tensor([[True,True,True,False]])
    gaps,active = history_gaps(times,valid)
    scales = module(gaps,active)
    assert active.tolist() == [[False,True,True,False]]
    for scale in scales:
        assert torch.equal(scale,torch.ones_like(scale))
    if mode == "dual":
        assert scales[1] is scales[2]
    if mode == "triple":
        assert not ({id(p) for p in module.calibrators["write"].parameters()} &
                    {id(p) for p in module.calibrators["phase"].parameters()})


def test_bounds_zero_gap_and_delayed_gradients():
    torch.manual_seed(2026)
    module = ThreeTimes("triple").double()
    gaps = torch.tensor([[0,0,1,838393,3*838393]],dtype=torch.float64)
    active = torch.tensor([[False,True,True,True,True]])
    optimizer = torch.optim.SGD(module.parameters(),lr=.01)
    for step in range(3):
        optimizer.zero_grad()
        scales = module(gaps,active)
        sum(x.sum() for x in scales).backward()
        for calibrator in module.calibrators.values():
            assert calibrator.last.weight.grad.abs().sum()>0
            if step == 0:
                assert calibrator.first.weight.grad.abs().sum() == 0
            else:
                assert calibrator.first.weight.grad.abs().sum()>0
        optimizer.step()
    for scale in module(gaps,active):
        assert torch.isfinite(scale).all() and (scale>=.5).all() and (scale<=2).all()
        assert torch.equal(scale[:,0],torch.ones_like(scale[:,0]))
        assert (scale[:,1] != 1).any()
    extreme = gaps.clone()
    extreme[:,-1] = 1e300
    for scale in module(extreme,active):
        assert torch.isfinite(scale).all() and (scale>=.5).all() and (scale<=2).all()


def test_dual_alias_and_chain_rule():
    d = torch.ones(1,3,2,requires_grad=True)
    a = -torch.ones_like(d)
    scale = torch.full_like(d,1.2,requires_grad=True)
    adt,dw,dp = split_dt(d,a,scale,scale,scale)
    assert dw is dp
    (adt.sum()+2*dw.sum()+3*dp.sum()).backward()
    torch.testing.assert_close(scale.grad,4*d.detach())
    torch.testing.assert_close(d.grad,4*scale.detach())


@pytest.mark.parametrize("kw",[{"cu_seqlens":torch.tensor([0,1])},{"input_states":()},{"return_final_states":True}])
def test_reject_unsupported_scope(kw):
    with pytest.raises(NotImplementedError):
        dense_only(**kw)
