"""CPU tests of physical clocks and the differentiable phase-only operation."""

import io
import math
import unittest

import torch

from experiments.mamba3_absolute_phase.phase import (
    PeriodicPhase, phase_clock, periodic_features, reference_correction,
)
from experiments.mamba3_timeaware.time_inputs import history_gaps


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.times = torch.tensor([[1660000000123., 1660000723456., 1660000723456., float('nan')],
                                   [1660000123000., 1660004322000., 1660024456000., 1660035567000.]],
                                  dtype=torch.float64)
        self.valid = torch.tensor([[True, True, True, False], [True]*4])

    def test_exact_zero_no_rng_and_only_weight(self):
        before = torch.get_rng_state().clone()
        module = PeriodicPhase(32, 'absolute_phase')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(list(module.state_dict()), ['W'])
        self.assertEqual(module.W.dtype, torch.float32)
        self.assertEqual(module.W.numel(), 128)
        self.assertEqual(module.W.count_nonzero().item(), 0)
        self.assertEqual(module(self.times, self.valid).count_nonzero().item(), 0)

    def test_independent_double_oracle_and_gradients(self):
        for mode in ('absolute_phase', 'relative_phase'):
            module = PeriodicPhase(3, mode).double()
            with torch.no_grad():
                module.W.copy_(torch.arange(12).reshape(3, 4)/31 - .1)
            got = module(self.times, self.valid)
            expected = reference_correction(self.times, self.valid, mode, module.W)
            torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
            g1, = torch.autograd.grad(got.square().sum(), module.W)
            g2, = torch.autograd.grad(expected.square().sum(), module.W)
            torch.testing.assert_close(g1, g2, rtol=1e-12, atol=1e-12)
            self.assertTrue((g1.abs().sum(0) > 0).all())

    def test_first_padding_and_zero_gap(self):
        module = PeriodicPhase(3, 'absolute_phase')
        with torch.no_grad():
            module.W.fill_(.2)
        correction, _, active = module.components(self.times, self.valid)
        self.assertTrue(active[0, 2])
        self.assertFalse(active[:, 0].any())
        self.assertEqual(correction[:, 0].count_nonzero().item(), 0)
        self.assertEqual(correction[0, 3].count_nonzero().item(), 0)
        self.assertGreater(correction[0, 2].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(correction).all())

    def test_negative_gap_policy_unchanged(self):
        times = torch.tensor([[1000., 400., 1300.]], dtype=torch.float64)
        valid = torch.ones_like(times, dtype=torch.bool)
        got = phase_clock(times, valid, 'relative_phase')
        torch.testing.assert_close(got, torch.tensor([[0., 0., 900.]], dtype=torch.float64), rtol=0, atol=0)
        gaps, active = history_gaps(times, valid)
        self.assertTrue(torch.equal(active, torch.tensor([[False, True, True]])))
        self.assertTrue(torch.equal(gaps, torch.tensor([[0., 0., 900.]], dtype=torch.float64)))

    def test_periods_and_absolute_relative_shift(self):
        for mode in ('absolute_phase', 'relative_phase'):
            original = periodic_features(self.times, self.valid, mode, dtype=torch.float64)
            day = periodic_features(self.times+86400000, self.valid, mode, dtype=torch.float64)
            torch.testing.assert_close(original, day, rtol=0, atol=1e-12)
            shifted = periodic_features(self.times+7200000, self.valid, mode, dtype=torch.float64)
            if mode == 'absolute_phase':
                self.assertGreater((shifted[self.valid]-original[self.valid]).abs().max().item(), .1)
            else:
                self.assertTrue(torch.equal(original, shifted))
        six = periodic_features(self.times+21600000, self.valid, 'absolute_phase', dtype=torch.float64)
        base = periodic_features(self.times, self.valid, 'absolute_phase', dtype=torch.float64)
        torch.testing.assert_close(six[..., :2], base[..., :2], rtol=0, atol=1e-12)

    def test_precision_before_remainder(self):
        times = torch.tensor([[1660000000000., 1660000000001.]], dtype=torch.float64)
        valid = torch.ones_like(times, dtype=torch.bool)
        phi = periodic_features(times, valid, 'absolute_phase', dtype=torch.float64)
        self.assertGreater((phi[0, 0]-phi[0, 1]).abs().max().item(), 1e-8)
        lost = periodic_features(times.float(), valid, 'absolute_phase', dtype=torch.float64)
        self.assertTrue(torch.equal(lost[0, 0], lost[0, 1]))

    def test_invalid_valid_timestamp_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            bad = self.times.clone()
            bad[0, 1] = value
            for mode in ('absolute_phase', 'relative_phase'):
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    periodic_features(bad, self.valid, mode)

    def test_large_timestamp_and_period_boundary(self):
        times = torch.tensor([[0., 9e15, 9e15+1000]], dtype=torch.float64)
        valid = torch.ones_like(times, dtype=torch.bool)
        for mode in ('absolute_phase', 'relative_phase'):
            self.assertTrue(torch.isfinite(periodic_features(times, valid, mode)).all())
        boundary = torch.tensor([[21600000-1e-3, 21600000+1e-3]], dtype=torch.float64)
        phi = periodic_features(boundary, torch.ones_like(boundary, dtype=torch.bool),
                                'absolute_phase', dtype=torch.float64)
        self.assertLess((phi[0, 0, :2]-phi[0, 1, :2]).abs().max().item(), 1e-9)

    def test_roundtrip_weights_only(self):
        source = PeriodicPhase(32, 'absolute_phase')
        with torch.no_grad():
            source.W.copy_(torch.arange(128).reshape(32, 4)/100)
        stream = io.BytesIO()
        torch.save(source.state_dict(), stream)
        stream.seek(0)
        target = PeriodicPhase(32, 'absolute_phase')
        target.load_state_dict(torch.load(stream, weights_only=True))
        self.assertTrue(torch.equal(source(self.times, self.valid), target(self.times, self.valid)))

    def test_no_target_or_cross_user_input(self):
        module = PeriodicPhase(3, 'relative_phase')
        with torch.no_grad():
            module.W.fill_(.2)
        got = module(self.times, self.valid)
        intervened = self.times.clone()
        intervened[1] += torch.tensor([0., 10000., 1000000., 99999999.])
        self.assertTrue(torch.equal(got[0], module(intervened, self.valid)[0]))
        intervened = self.times.clone()
        intervened[:, 2:] += 1234567
        self.assertTrue(torch.equal(got[:, :2], module(intervened, self.valid)[:, :2]))


if __name__ == '__main__':
    unittest.main()
