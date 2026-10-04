"""Scientific counters, frozen coverage decisions, and RNG-neutral callbacks."""
import json
import math
import random
import unittest
from types import SimpleNamespace

import torch

from experiments.mamba3_absolute_phase.coverage import audit_histories, fixed_indices
from experiments.mamba3_absolute_phase.diagnostics import (
    PhaseDiagnostics, parameter_snapshot, ABSOLUTE_HOURS, RELATIVE_HOURS,
)


def history_fixture():
    origin = 1651363200000
    hours = torch.tensor([[0, 0, 1, 6, 24], [24, 27, 30, 48, 49],
                          [48, 0, 0, 0, 0]], dtype=torch.float64)
    lengths = torch.tensor([5, 5, 1])
    valid = torch.arange(5)[None] < lengths[:, None]
    times = origin + hours * 3600000
    times[~valid] = float('nan')
    items = torch.where(valid, torch.arange(1, 6)[None], 0)
    return items, lengths, times


def observer_fixture():
    active = torch.tensor([[False, True, True], [False, False, False]])
    base = torch.tensor([[[[0., .5], [1., -.5]]] * 3] * 2)
    delta = torch.tensor([[[0., 0.], [.1, -.2], [.2, .3]], [[0., 0.]] * 3])
    corrected = base + delta[:, :, None, :]
    return dict(phase_active=active, raw_angles_before=base,
                raw_angle_correction=delta, raw_angles_corrected=corrected,
                phase_features=torch.zeros(2, 3, 4), dp=torch.ones(2, 2, 3),
                angles=corrected)


class CoverageTests(unittest.TestCase):
    def test_fixed_indices_deterministic(self):
        self.assertEqual(fixed_indices(12, 5), [0, 2, 5, 8, 11])
        self.assertEqual(len(fixed_indices(1062567)), 10000)

    def test_coverage_counts_duplicate_times_and_phase_dates(self):
        items, lengths, times = history_fixture()
        result = audit_histories(items, lengths, times, [0, 12, 20])
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['active_event_occurrences'], 8)
        self.assertEqual(result['valid_event_occurrences'], 11)
        self.assertEqual(result['padding_occurrences'], 4)
        self.assertEqual(result['equal_adjacent_timestamp_count'], 1)
        self.assertEqual(result['equal_adjacent_timestamp_fraction'], 1 / 8)
        self.assertEqual(result['chronological_history_span_ms']['max'], 25 * 3600000)
        self.assertEqual(result['periods'][1]['complete_period_lengths_in_range'], 2)
        self.assertEqual(result['periods'][1]['by_utc_date'][0]['utc_date'], '2022-05-01')
        self.assertGreater(result['periods'][1]['unique_exact_phases_on_multiple_dates'], 0)
        self.assertEqual(sum(result['periods'][0]['active_bin_counts']), 8)
        json.dumps(result, allow_nan=False)

    def test_constant_active_phase_and_short_range_blocked(self):
        items, lengths, times = history_fixture()
        valid = items != 0
        times[valid] = 1651363200000
        result = audit_histories(items, lengths, times, [0, 1, 2])
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertIn('constant', result['blocking_reason'])
        self.assertIn('24-hour', result['blocking_reason'])

    def test_nonfinite_or_fractional_valid_milliseconds_fail_semantics(self):
        items, lengths, times = history_fixture()
        for value in (float('nan'), float('inf'), 1651363200000.5, 2.0 ** 54):
            changed = times.clone()
            changed[0, 0] = value
            result = audit_histories(items, lengths, changed, [0, 1, 2])
            self.assertEqual(result['status'], 'BLOCKED')
            json.dumps(result, allow_nan=False)

    def test_negative_gap_is_reported_with_inherited_clamp(self):
        items, lengths, times = history_fixture()
        times[0, 2] = times[0, 1] - 1000
        result = audit_histories(items, lengths, times, [0, 1, 2])
        self.assertEqual(result['negative_adjacent_gap_count'], 1)
        self.assertEqual(result['status'], 'PASS')
        self.assertGreater(result['clamped_cumulative_history_span_ms']['mean'],
                           result['chronological_history_span_ms']['mean'])

    def test_no_phase_bin_threshold(self):
        items = torch.ones(2, 3, dtype=torch.int64)
        lengths = torch.tensor([3, 3])
        origin = 1651363200000
        # Informative sub-bin phase differences must not fail an arbitrary bin quota.
        times = torch.tensor([[origin, origin + 1, origin + 2],
                              [origin + 86400000, origin + 86400001, origin + 86400002]], dtype=torch.float64)
        result = audit_histories(items, lengths, times, [0, 1])
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['periods'][0]['occupied_active_bins'], 1)

    def test_hashes_bind_inputs_and_do_not_consume_rng(self):
        items, lengths, times = history_fixture()
        before_py, before_torch = random.getstate(), torch.get_rng_state().clone()
        first = audit_histories(items, lengths, times, [0, 1, 2])
        changed = times.clone()
        changed[0, 1] += 1
        second = audit_histories(items, lengths, changed, [0, 1, 2])
        self.assertNotEqual(first['selected_records_sha256'], second['selected_records_sha256'])
        self.assertEqual(before_py, random.getstate())
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))


class DiagnosticsTests(unittest.TestCase):
    def test_fixed_grids_zero_weight_and_grad_missing(self):
        phase = SimpleNamespace(W=torch.nn.Parameter(torch.zeros(2, 4)))
        result = parameter_snapshot(phase, 'absolute_phase', {'status': 'PASS'})
        self.assertEqual(result['absolute_24h_grid']['hours'], list(ABSOLUTE_HOURS))
        self.assertEqual(result['relative_grid']['hours'], list(RELATIVE_HOURS))
        self.assertEqual(result['gradients']['status'], 'NOT_RECORDED')
        self.assertEqual(result['absolute_24h_grid']['raw_angle_correction_stats']['abs_max'], 0)
        self.assertEqual(result['W_l2_norm'], 0)
        self.assertEqual(result['train_coverage_summary']['status'], 'PASS')
        json.dumps(result, allow_nan=False)

    def test_frequency_pairs_and_native_increment_formula(self):
        phase = SimpleNamespace(W=torch.nn.Parameter(torch.tensor([[.1, .2, .3, .4]])))
        phase.W.grad = torch.tensor([[1., 0., 0., 0.]])
        result = parameter_snapshot(phase, 'relative_phase')
        first = result['absolute_24h_grid']
        expected = math.tanh((.2 + .4) / math.sqrt(2))
        self.assertAlmostEqual(first['raw_angle_correction'][0][0], expected, places=7)
        self.assertAlmostEqual(first['frequency_6h_linear_contribution'][0][0], .2 / math.sqrt(2), places=7)
        fixture = first['content_dt_fixtures'][2]
        self.assertAlmostEqual(fixture['native_increment_difference'][0][0],
                               math.pi * math.tanh(expected) * .5, places=6)
        self.assertTrue(result['gradients']['finite'])
        self.assertEqual(result['gradients']['l2_norm'], 1.)
        # Features repeat at 24h/48h; raw corrections should repeat as well.
        relative = result['relative_grid']['raw_angle_correction']
        self.assertEqual(relative[0], relative[5])
        self.assertEqual(relative[0], relative[6])

    def test_observer_exact_last_query_scope_and_bf16_label(self):
        payload = observer_fixture()
        collector = PhaseDiagnostics('absolute_phase')
        collector.observe(0, payload)
        collector.observe(1, payload)
        result = collector.result()
        layer = result['layers']['0']
        self.assertEqual(layer['histories'], 2)
        self.assertEqual(layer['histories_without_active_events'], 1)
        self.assertEqual(layer['selected_active_queries'], 1)
        self.assertEqual(layer['observed_active_positions'], 2)
        self.assertEqual(layer['stats']['raw_angle_correction']['count'], 2)
        self.assertEqual(layer['stats']['native_increment_difference_fp32']['count'], 4)
        self.assertEqual(layer['examples'][0]['query_position'], 2)
        self.assertIn('COUNTERFACTUAL', result['bf16_scope'])
        self.assertEqual(result['additional_dataset_forwards'], 0)
        json.dumps(result, allow_nan=False)

    def test_observer_empty_not_measured_and_bad_correction_rejected(self):
        collector = PhaseDiagnostics('relative_phase')
        self.assertEqual(collector.result()['status'], 'NOT_RECORDED')
        payload = observer_fixture()
        payload['raw_angles_corrected'] = payload['raw_angles_corrected'] + .1
        with self.assertRaises(ValueError):
            collector.observe(0, payload)
        self.assertEqual(collector.result()['status'], 'NOT_RECORDED')

    def test_snapshot_and_observer_are_rng_neutral(self):
        phase = SimpleNamespace(W=torch.nn.Parameter(torch.ones(2, 4)))
        before_py, before_torch = random.getstate(), torch.get_rng_state().clone()
        parameter_snapshot(phase, 'relative_phase')
        collector = PhaseDiagnostics('relative_phase')
        collector.observe(0, observer_fixture())
        collector.result()
        self.assertEqual(before_py, random.getstate())
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))

    def test_baseline_no_adapter_and_nonfinite_gradient_rejected(self):
        self.assertEqual(parameter_snapshot(None, 'baseline_dual')['status'], 'NOT_APPLICABLE')
        phase = SimpleNamespace(W=torch.nn.Parameter(torch.zeros(2, 4)))
        phase.W.grad = torch.full_like(phase.W, float('inf'))
        with self.assertRaises(ValueError):
            parameter_snapshot(phase, 'absolute_phase')
        with self.assertRaises(ValueError):
            parameter_snapshot(phase, 'baseline_dual')


if __name__ == '__main__':
    unittest.main()
