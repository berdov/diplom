"""CPU tests exercise the production selector and production reader."""
import math
import unittest

import torch

from experiments.mamba3_time_memory.memory import (
    ANCHORS, REFERENCE_MS, read_memory, read_memory_reference, select)


def nonuniform():
    return torch.tensor([[0, 1, 2, 3, 10, 11, 20, 21]], dtype=torch.float64) * REFERENCE_MS


def fixture(length=8, batch=1):
    times = torch.arange(length, dtype=torch.float64)[None].repeat(batch, 1) * REFERENCE_MS
    return times, torch.ones(batch, length, dtype=torch.bool)


class SelectorTests(unittest.TestCase):
    def test_strict_causality_unique_counts_every_prefix(self):
        times, valid = fixture(50, 3)
        valid[1, 3:] = False
        valid[2, 1:] = False
        for mode in ('index_memory', 'time_memory'):
            result = select(times, valid, mode)
            for b in range(3):
                for t in range(50):
                    indices = result['indices'][b, t][result['mask'][b, t]].tolist()
                    expected = min(4, int(valid[b, :t].sum())) if valid[b, t] else 0
                    self.assertEqual(len(indices), expected)
                    self.assertEqual(indices, sorted(set(indices)))
                    self.assertTrue(all(j < t and valid[b, j] for j in indices))

    def test_lengths_one_through_four(self):
        for length in range(1, 5):
            times, valid = fixture(length)
            for mode in ('index_memory', 'time_memory'):
                result = select(times, valid, mode)
                self.assertEqual(result['mask'].sum(-1).tolist(), [list(range(length))])
                self.assertTrue(torch.isfinite(result['ages']).all())

    def test_same_timestamp_latest_ties_zero_ages(self):
        times = torch.zeros(1, 8, dtype=torch.float64)
        result = select(times, torch.ones_like(times, dtype=torch.bool), 'time_memory')
        self.assertEqual(result['anchor_indices'][0, -1].tolist(), [6, 5, 4, 3])
        self.assertEqual(result['indices'][0, -1].tolist(), [3, 4, 5, 6])
        self.assertTrue((result['ages'] == 0).all())
        self.assertTrue(result['anchor_out_of_range'][0, -1].all())
        self.assertTrue(torch.equal(result['indices'], select(times, torch.ones_like(times, dtype=torch.bool), 'time_memory')['indices']))

    def test_uniform_clocks_equal(self):
        times, valid = fixture(50, 2)
        index, time = (select(times, valid, mode) for mode in ('index_memory', 'time_memory'))
        for key in index:
            self.assertTrue(torch.equal(index[key], time[key]), key)

    def test_hand_calculated_nonuniform(self):
        times = nonuniform()
        valid = torch.ones_like(times, dtype=torch.bool)
        index, time = (select(times, valid, mode) for mode in ('index_memory', 'time_memory'))
        self.assertEqual(index['indices'][0, -1].tolist(), [0, 1, 3, 6])
        self.assertEqual(time['anchor_indices'][0, -1].tolist(), [6, 5, 3, 0])
        self.assertEqual(time['indices'][0, -1].tolist(), [0, 3, 5, 6])
        self.assertEqual(time['ages'][0, -1].tolist(), [21, 18, 10, 1])
        self.assertEqual(time['event_lags'][0, -1].tolist(), [7, 4, 2, 1])
        self.assertEqual(time['anchor_out_of_range'][0, -1].tolist(), [False, True, False, True])

    def test_translation_preserves_addressing_and_ages(self):
        times = nonuniform()
        valid = torch.ones_like(times, dtype=torch.bool)
        for mode in ('index_memory', 'time_memory'):
            before, after = select(times, valid, mode), select(times + 1_600_000_000_000, valid, mode)
            for key in before:
                self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_suffix_huge_future_timestamp_no_prefix_effect(self):
        times = nonuniform()
        valid = torch.ones_like(times, dtype=torch.bool)
        changed = times.clone()
        changed[:, 5:] = torch.tensor([1e100, 2e100, 3e100], dtype=torch.float64)
        for mode in ('index_memory', 'time_memory'):
            before, after = select(times, valid, mode), select(changed, valid, mode)
            for key in before:
                self.assertTrue(torch.equal(before[key][:, :5], after[key][:, :5]), key)

    def test_padding_arbitrary_timestamps(self):
        times, valid = fixture(8)
        valid[:, 3:] = False
        before = select(times, valid, 'time_memory')
        times[:, 3:] = torch.tensor([float('nan'), float('inf'), -float('inf'), -9e250, 9e250])
        after = select(times, valid, 'time_memory')
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertFalse(after['mask'][:, 3:].any())

    def test_negative_gap_uses_existing_clamp_clock(self):
        times = torch.tensor([[3., 1., 4., 4.]], dtype=torch.float64) * REFERENCE_MS
        result = select(times, torch.ones_like(times, dtype=torch.bool), 'time_memory')
        self.assertEqual(result['clock_ms'].tolist(), [[0., 0., 3 * REFERENCE_MS, 3 * REFERENCE_MS]])
        self.assertEqual(result['ages'][0, -1, :3].tolist(), [3., 3., 0.])

    def test_repeated_item_ids_are_distinct_event_positions(self):
        item_ids = torch.tensor([[9, 9, 9, 9, 9, 9]])
        times, _ = fixture(6)
        for mode in ('index_memory', 'time_memory'):
            result = select(times, item_ids != 0, mode)
            self.assertEqual(len(set(result['indices'][0, -1].tolist())), 4)

    def test_batch_permutation_and_user_isolation(self):
        times = torch.cat((nonuniform(), nonuniform() * 5, nonuniform() + 1000))
        valid = torch.ones_like(times, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 1])
        for mode in ('index_memory', 'time_memory'):
            before = select(times, valid, mode)
            after = select(times[permutation], valid[permutation], mode)
            for key in before:
                self.assertTrue(torch.equal(before[key][permutation], after[key]), key)
            changed = times.clone()
            changed[1] *= 100
            isolated = select(changed, valid, mode)
            self.assertTrue(torch.equal(before['indices'][[0, 2]], isolated['indices'][[0, 2]]))

    def test_large_small_finite_and_invalid_inputs(self):
        valid = torch.ones(1, 4, dtype=torch.bool)
        for scale in (1e-200, 1e300):
            times = torch.arange(4, dtype=torch.float64)[None] * scale
            result = select(times, valid, 'time_memory')
            self.assertTrue(torch.isfinite(result['ages']).all())
        for times in (torch.tensor([[0., float('nan'), 2., 3.]], dtype=torch.float64),
                      torch.tensor([[-1e308, 1e308, 1e308, 1e308]], dtype=torch.float64),
                      torch.tensor([[0., 1e308, 0., 1e308]], dtype=torch.float64)):
            with self.assertRaises(ValueError):
                select(times, valid, 'time_memory')
        with self.assertRaises(ValueError):
            select(torch.zeros(1, 4), valid, 'wrong')
        with self.assertRaises(ValueError):
            select(torch.zeros(1, 4), valid.long(), 'time_memory')


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.times, self.valid = fixture(6)
        self.selection = select(self.times, self.valid, 'index_memory')
        self.hidden = (torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) / 20 + .1)

    def test_hand_computed_projection_free_reader(self):
        times, valid = fixture(3)
        selection = select(times, valid, 'index_memory')
        hidden = torch.tensor([[[1., 0.], [0., 1.], [1., 2.]]])
        beta = torch.tensor(.25)
        output, details = read_memory(hidden, selection, beta)
        first = math.exp(1 / math.sqrt(2))
        second = math.exp(2 / math.sqrt(2))
        weights = torch.tensor([first / (first + second), second / (first + second)])
        self.assertTrue(torch.allclose(details['weights'][0, 2, :2], weights, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(details['readout'][0, 2], weights, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(output[0, 2], hidden[0, 2] + math.tanh(.25) * weights, atol=1e-7, rtol=1e-6))

    def test_zero_gate_exact_output_and_common_gradient(self):
        hidden = self.hidden.clone().requires_grad_()
        baseline = self.hidden.clone().requires_grad_()
        beta = torch.zeros((), requires_grad=True)
        output, details = read_memory(hidden, self.selection, beta)
        self.assertTrue(torch.equal(output, baseline))
        output.square().sum().backward()
        baseline.square().sum().backward()
        self.assertTrue(torch.equal(hidden.grad, baseline.grad))
        self.assertGreater(abs(beta.grad.item()), 0.)
        self.assertTrue(torch.isfinite(details['weights']).all())

    def test_reference_outputs_and_fixed_index_gradcheck(self):
        hidden = self.hidden.double().requires_grad_()
        beta = torch.tensor(.23, dtype=torch.float64, requires_grad=True)
        reference, _ = read_memory_reference(hidden, self.selection, beta)
        actual, _ = read_memory(hidden.float(), self.selection, beta.float())
        self.assertTrue(torch.allclose(actual.double(), reference, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.autograd.gradcheck(
            lambda h, b: read_memory_reference(h, self.selection, b)[0],
            (hidden, beta), eps=1e-6, atol=1e-6, rtol=1e-5))

    def test_nonzero_gate_query_and_selected_value_gradients(self):
        hidden = self.hidden.clone().requires_grad_()
        beta = torch.tensor(.4, requires_grad=True)
        _, details = read_memory(hidden, self.selection, beta)
        details['readout'][0, -1].square().sum().backward()
        self.assertGreater(hidden.grad[0, -1].abs().sum().item(), 0.)
        for index in self.selection['indices'][0, -1].tolist():
            self.assertGreater(hidden.grad[0, index].abs().sum().item(), 0.)

    def test_all_empty_backward(self):
        times, valid = fixture(1, 3)
        selection = select(times, valid, 'time_memory')
        hidden = torch.ones(3, 1, 4, requires_grad=True)
        beta = torch.tensor(.4, requires_grad=True)
        output, details = read_memory(hidden, selection, beta)
        self.assertTrue(torch.equal(output, hidden))
        self.assertTrue((details['readout'] == 0).all())
        self.assertTrue((details['weights'] == 0).all())
        output.sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertEqual(beta.grad.item(), 0.)

    def test_masks_sentinel_and_padding_neutrality(self):
        times, valid = fixture(6)
        valid[:, 2:] = False
        selection = select(times, valid, 'time_memory')
        hidden = self.hidden.clone()
        hidden[:, -1] = 1e10
        output, details = read_memory(hidden, selection, torch.tensor(.5))
        self.assertTrue(torch.equal(output[:, 0], hidden[:, 0]))
        self.assertTrue(torch.equal(output[:, 2:], hidden[:, 2:]))
        self.assertTrue((details['values'][~selection['mask']] == 0).all())
        self.assertTrue((details['weights'][~selection['mask']] == 0).all())
        self.assertEqual(details['weights'].sum(-1).tolist(), [[0., 1., 0., 0., 0., 0.]])

    def test_identical_addresses_outputs_derivatives_and_fp32_scores(self):
        outputs, gradients = [], []
        for mode in ('index_memory', 'time_memory'):
            hidden = self.hidden.clone().requires_grad_()
            beta = torch.tensor(-.3, requires_grad=True)
            output, details = read_memory(hidden, select(self.times, self.valid, mode), beta)
            output.square().sum().backward()
            outputs.append(output.detach())
            gradients.append((hidden.grad, beta.grad))
            self.assertEqual(details['weights'].dtype, torch.float32)
            self.assertEqual(details['readout'].dtype, torch.float32)
        self.assertTrue(torch.equal(*outputs))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*gradients)))

    def test_fixed_hidden_age_intervention_only_changes_time_addressing(self):
        uniform, valid = fixture(8)
        times = nonuniform()
        hidden = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4) / 20
        beta = torch.tensor(.5)
        for mode in ('index_memory', 'time_memory'):
            first = read_memory(hidden, select(uniform, valid, mode), beta)[0]
            second = read_memory(hidden, select(times, valid, mode), beta)[0]
            self.assertEqual(torch.equal(first, second), mode == 'index_memory')

    def test_signed_gate_can_become_negative(self):
        beta = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([beta], lr=.001)
        output, _ = read_memory(self.hidden, self.selection, beta)
        output.sum().backward()
        self.assertGreater(beta.grad.item(), 0.)
        optimizer.step()
        self.assertLess(beta.item(), 0.)

    def test_suffix_intervention_and_no_persistent_state(self):
        beta = torch.tensor(.3)
        first, _ = read_memory(self.hidden, self.selection, beta)
        changed = self.hidden.clone()
        changed[:, 4:] *= 100
        second, _ = read_memory(changed, self.selection, beta)
        self.assertTrue(torch.equal(first[:, :4], second[:, :4]))
        read_memory(self.hidden + 30, self.selection, beta)
        third, _ = read_memory(self.hidden, self.selection, beta)
        self.assertTrue(torch.equal(first, third))

    def test_invalid_addresses_rejected(self):
        for index in (-2, 5, 999):
            selection = {key: value.clone() for key, value in self.selection.items()}
            selection['indices'][0, 1, 0] = index
            with self.assertRaises(ValueError):
                read_memory(self.hidden, selection, torch.tensor(.3))


if __name__ == '__main__':
    unittest.main()
