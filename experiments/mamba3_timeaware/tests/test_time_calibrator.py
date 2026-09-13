import math

import pytest
import torch

from experiments.mamba3_timeaware.time_inputs import TimeCalibrator, condition_dt


def test_identity_shape():
    model = TimeCalibrator(2, 1)
    gaps = torch.tensor([[0., 1., 100.], [4., 0., 6.]])
    out = model(gaps, torch.ones_like(gaps, dtype=torch.bool))
    assert out.shape == (2, 3, 2)
    assert torch.equal(out, torch.ones_like(out))
    assert torch.count_nonzero(model.last.weight) == 0
    assert torch.count_nonzero(model.last.bias) == 0


def test_bounded_huge_gaps():
    model = TimeCalibrator(2, 1e-200)
    with torch.no_grad():
        model.last.weight.fill_(100)
        model.last.bias.copy_(torch.tensor([-100., 100.]))
    gaps = torch.tensor([[0., 1e200, 1e300]], dtype=torch.float64)
    out = model(gaps, torch.ones_like(gaps, dtype=torch.bool))
    assert torch.isfinite(out).all()
    assert (out >= 0.5).all() and (out <= 2).all()


def test_gradients_through_dt():
    model = TimeCalibrator(2, 10)
    gaps = torch.tensor([[0., 10., 100.]])
    scale = model(gaps, gaps > 0)
    dt, adt = condition_dt(torch.ones_like(scale), -torch.ones_like(scale), scale)
    (dt.square().sum() + adt.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.last.weight.grad.abs().sum() > 0
    # Identity init blocks the first layer gradient initially, by design.
    assert model.first.weight.grad.abs().sum() == 0
    with torch.no_grad():
        model.last.weight.fill_(0.01)
    model.zero_grad()
    model(gaps, gaps > 0).sum().backward()
    assert model.first.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('reference', [None, 0, -1, math.inf, math.nan])
def test_reference_required(reference):
    with pytest.raises(ValueError):
        TimeCalibrator(2, reference)


@pytest.mark.parametrize('gap', [-1., math.inf, math.nan])
def test_bad_gap_rejected(gap):
    with pytest.raises(ValueError):
        TimeCalibrator(2, 1)(torch.tensor([[gap]]), torch.tensor([[True]]))


@pytest.mark.parametrize('bound', [0, -1, 21, math.inf])
def test_bad_bound_rejected(bound):
    with pytest.raises(ValueError):
        TimeCalibrator(2, 1, bound)
