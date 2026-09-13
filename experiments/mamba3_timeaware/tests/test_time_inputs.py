import inspect

import pytest
import torch

from experiments.mamba3_timeaware.time_inputs import history_gaps, TimeCalibrator, condition_dt


def test_history_padding_negative_and_first():
    timestamps = torch.tensor([[999., 10., 20., 15., float('nan')]])
    valid = torch.tensor([[False, True, True, True, False]])
    gaps, active = history_gaps(timestamps, valid)
    assert gaps.tolist() == [[0, 0, 10, 0, 0]]
    assert active.tolist() == [[False, False, True, True, False]]
    timestamps[0, 0] = -1000
    assert torch.equal(gaps, history_gaps(timestamps, valid)[0])


def test_neutral_mask_even_after_learning():
    calibrator = TimeCalibrator(2, 10)
    with torch.no_grad():
        calibrator.last.bias.fill_(1)
    gaps, active = history_gaps(torch.tensor([[10., 20., 0.]]), torch.tensor([[True, True, False]]))
    scale = calibrator(gaps, active)
    assert torch.equal(scale[:, [0, 2]], torch.ones(1, 2, 2))
    assert (scale[:, 1] > 1).all()


def test_history_prefix_causality():
    times = torch.tensor([[10., 20., 30., 40.]])
    valid = torch.ones_like(times, dtype=torch.bool)
    before = history_gaps(times, valid)[0]
    times[:, -1] = 10000
    after = history_gaps(times, valid)[0]
    assert torch.equal(before[:, :-1], after[:, :-1])
    assert list(inspect.signature(history_gaps).parameters) == ['timestamps', 'valid_mask']


def test_timestamp_precision():
    gaps, _ = history_gaps(torch.tensor([[1700000000000, 1700000000001]]), torch.tensor([[True, True]]))
    assert gaps[0, 1] == 1


@pytest.mark.parametrize('times,mask', [
    (torch.zeros(2), torch.ones(2, dtype=torch.bool)),
    (torch.zeros(1, 2), torch.ones(1, 2)),
    (torch.tensor([[float('inf')]]), torch.tensor([[True]])),
])
def test_bad_history(times, mask):
    with pytest.raises(ValueError):
        history_gaps(times, mask)


def test_dt_identity_and_consistency():
    dt = torch.rand(2, 4, 2)
    a = -torch.rand_like(dt)
    real, adt = condition_dt(dt, a, torch.ones_like(dt))
    assert torch.equal(real, dt)
    assert torch.equal(adt, a * dt)
    real, adt = condition_dt(dt, a, torch.full_like(dt, 2))
    assert torch.equal(real, 2 * dt)
    assert torch.equal(adt, a * real)


def test_dt_rejects_broadcast_heads():
    with pytest.raises(ValueError):
        condition_dt(torch.ones(2, 4, 2), torch.ones(2, 4, 2), torch.ones(2, 4, 1))
