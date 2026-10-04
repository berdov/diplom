"""Auditor-only regressions; standard library, no scientific execution."""
import copy
import hashlib
import json
import math
import struct
import sys
import unittest
from pathlib import Path

from . import audit_saved as a


def stats(values):
    n = len(values)
    total, squared = math.fsum(values), math.fsum(x*x for x in values)
    zeros = sum(x == 0 for x in values)
    return dict(count=n, sum=total, squared_sum=squared, mean=total/n if n else None,
        rms=math.sqrt(squared/n) if n else None, min=min(values) if n else None, max=max(values) if n else None,
        abs_max=max(map(abs, values)) if n else None, zero_count=zeros, zero_fraction=zeros/n if n else None)


def packed(values):
    return dict(shape=[len(values)], dtype='torch.float32', values=values,
                sha256=hashlib.sha256(b''.join(struct.pack('<f', x) for x in values)).hexdigest())


class AuditorTests(unittest.TestCase):
    def test_import_does_not_load_tensor_libraries(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_packed_byte_hash_and_shape_detect_changes(self):
        row = packed([1., 0., -1.])
        self.assertEqual(a.packed_check(row), 1)
        for change in ({'sha256': '0'*64}, {'values': [1., 0., 1.]}, {'shape': [4]}, {'values': [1., math.nan, -1.]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                a.packed_check(dict(row, **change))

    def test_bf16_round_to_nearest_even(self):
        self.assertEqual(a.bf16(1+1/256), 1.)
        self.assertEqual(a.bf16(1+3/256), 1+1/64)
        self.assertEqual(a.bf16(-1-1/256), -1.)
        self.assertEqual(a.bf16(0.), 0.)

    def test_each_child_preserves_original_allocation_mask_and_identity(self):
        expected = {'gate/child_environment.json': 'gate', 'smoke/child_environment.json': 'smoke',
                    'relative_seed2027/process/child_environment.json': 'relative_phase',
                    'relative_seed2028/process/child_environment.json': 'relative_phase'}
        for mask in ({'present': True, 'value': 'GPU-owned-uuid'}, {'present': False, 'value': None}):
            children = {path: dict(stage=stage, parent_cuda_visibility=copy.deepcopy(mask), child_cuda_visibility=copy.deepcopy(mask))
                        for path, stage in expected.items()}
            self.assertEqual(a.child_environment_check(mask, children, expected)['children_verified'], 4)
            for side in ('parent_cuda_visibility', 'child_cuda_visibility'):
                changed = copy.deepcopy(children)
                changed['relative_seed2028/process/child_environment.json'][side] = {'present': True, 'value': '0'}
                with self.subTest(mask=mask, changed=side), self.assertRaises(ValueError):
                    a.child_environment_check(mask, changed, expected)
            changed = copy.deepcopy(children); changed['gate/child_environment.json']['stage'] = 'smoke'
            with self.assertRaises(ValueError):
                a.child_environment_check(mask, changed, expected)
            changed = copy.deepcopy(children); changed.pop('relative_seed2028/process/child_environment.json')
            with self.assertRaises(ValueError):
                a.child_environment_check(mask, changed, expected)
            changed = copy.deepcopy(children)
            changed['relative_seed2026/process/child_environment.json'] = changed.pop('relative_seed2028/process/child_environment.json')
            with self.assertRaises(ValueError):
                a.child_environment_check(mask, changed, expected)
        with self.assertRaises(ValueError):
            a.child_environment_check({'present': False, 'value': '0'}, {}, {})

    def test_scalar_stats_signed_empty_and_mutations(self):
        values = [-1., 0., .5, .5]
        a.scalar_stats(stats(values), 'fixture', a.Arithmetic(), len(values), values)
        a.scalar_stats(stats([]), 'empty', a.Arithmetic(), 0, [])
        for change in ({'count': 5}, {'sum': 0.1}, {'squared_sum': -1.}, {'zero_fraction': 0.}, {'max': .4}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                a.scalar_stats(dict(stats(values), **change), 'fixture', a.Arithmetic(), len(values), values)

    def test_local_increment_distinguishes_counterfactual_rounding(self):
        native, effective, counter = a.local_increment(1., a.f32(1+.001), .25)
        self.assertNotEqual(native, 0.)
        self.assertEqual(effective, 0.)
        self.assertEqual(counter, 0.)
        self.assertEqual(a.local_increment(0., 0., 1.), (0., 0., 0.))

    def test_zero_analytic_grid_and_fabricated_fixture_aggregate(self):
        hours = [0, 1, 3, 6, 12, 24, 48]
        n = len(hours)
        features = []
        for h in hours:
            row = []
            for period in a.PERIODS:
                angle = (h*3600000 % period)*(2*math.pi/period)
                row += [a.f32(math.sin(angle)/math.sqrt(2)), a.f32(math.cos(angle)/math.sqrt(2))]
            features.append(row)
        grid = dict(hours=hours, clock_ms=[h*3600000 for h in hours], features=features,
            feature_dtype='float32 after float64 remainder/sin/cos', abs_delta_near_saturation_count=0,
            abs_delta_near_saturation_fraction=0., content_dt_fixtures=[])
        for key in ('frequency_6h_linear_contribution', 'frequency_24h_linear_contribution',
                    'linear_preactivation', 'raw_angle_correction'):
            grid[key] = [[0.]*32 for _ in hours]
        for key in ('frequency_6h_stats', 'frequency_24h_stats', 'raw_angle_correction_stats'):
            grid[key] = stats([0.]*(32*n))
        for content, dt in [(-2., .05), (-.5, .25), (0., .5), (.5, 1.), (2., 2.)]:
            fixture = dict(content_raw_angle=content, dt_phase=dt, sampled_angle_indices=[0,1,2,3],
                samples_scope='all grid clocks, first four angle coordinates; stats cover every angle',
                counterfactual_nonzero_fp32_corrections_lost_in_bf16=0)
            for key in ('native_increment_difference', 'counterfactual_bf16_effective_raw_correction',
                        'counterfactual_bf16_increment_difference'):
                fixture[key] = [[0.]*4 for _ in hours]
            for key in ('native_increment_difference_stats', 'counterfactual_bf16_increment_difference_stats',
                        'counterfactual_bf16_rounding_increment_error_stats'):
                fixture[key] = stats([0.]*(32*n))
            grid['content_dt_fixtures'].append(fixture)
        a.analytic_grid(grid, [[0.]*4 for _ in range(32)], hours, 'zero-grid', a.Arithmetic())
        with self.assertRaises(ValueError):
            a.approximate_stats(stats([1.]*128), [0.]*128, 1e-6, 'fabricated', a.Arithmetic())

    def test_signed_reduction_cancellation_uses_magnitudes(self):
        values = [1., -1., 1e-16]*256
        row = stats(values)
        row['sum'] = 0.
        row['mean'] = 0.
        a.scalar_stats(row, 'cancelled', a.Arithmetic(), len(values), values)

    def test_diagnostic_bounds_are_finite_and_bounded(self):
        ar = a.Arithmetic()
        ar.tanh(a.f32(math.tanh(.1)), .1, 'tanh')
        for value in (math.nan, math.inf, 1.):
            with self.subTest(value=value), self.assertRaises((ValueError, OverflowError)):
                ar.increment(value, 0., .25, 'bad')
        with self.assertRaises(ValueError):
            ar.dot(1., [0., 0., 0., 0.], 'bad dot')

    def test_paired_sample_std_and_matched_first27(self):
        records = {}
        for seed in (2027, 2028):
            for i, mode in enumerate(a.MODES):
                records[mode, seed] = dict(best_valid_metrics={m: .01*(seed-2026)+.001*i for m in a.METRICS},
                    first27_complete=not (mode == 'baseline_dual' and seed == 2028),
                    first27_best_ndcg10=.009*(seed-2026)+.001*i)
        result = a.aggregates(records, (2027, 2028), 'new')
        self.assertAlmostEqual(result['per_variant']['baseline_dual']['ndcg@10']['sample_std'], .01/math.sqrt(2))
        self.assertEqual(result['contrasts'][0]['signs'], {'+': 2, '-': 0, '0': 0})
        self.assertEqual(result['first27_subsets'][1]['seeds'], [2027])
        self.assertEqual(result['first27_subsets'][1]['paired']['sample_std'], None)
        records['baseline_dual', 2027]['best_valid_metrics'] = {m: 0. for m in a.METRICS}
        records['baseline_dual', 2028]['best_valid_metrics'] = {m: 0. for m in a.METRICS}
        self.assertIsNone(a.aggregates(records, (2027, 2028), 'zero')['contrasts'][1]['relative_difference_of_means_percent'])

    def test_saved_CPU_coverage_and_tampered_scope(self):
        path = Path(__file__).resolve().parents[1]/'evidence/job4374917/submission/train_coverage.json'
        row = json.loads(path.read_text())
        digest = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        a.coverage_check(row, a.Arithmetic(), digest)
        for change in ({'target_fields_read': True}, {'valid_event_occurrences': 1}, {'active_event_occurrences': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                a.coverage_check(dict(row, **change), a.Arithmetic(), digest)
        changed = copy.deepcopy(row); changed['periods'][0]['active_bin_counts'][0] += 1
        with self.assertRaises(ValueError):
            a.coverage_check(changed, a.Arithmetic(), digest)

    def test_both_log_copies_parser_preserves_all_metrics(self):
        metrics = '    '.join(k+' : 0.062' for k in sorted(a.METRICS, key=lambda x: (x != 'hit@5', x)))
        text = '\x1b[32mepoch 0 training [time: 1.2s, train loss: 2.0000]\x1b[0m\n'
        text += 'epoch 0 evaluating [time: 0.2s, valid_score: 0.062]\n'+metrics+'\n'
        text += 'Finished training, best eval result in epoch 0\n'
        evaluations, parsed, training, selected = a.parse_logs(text)
        self.assertEqual(set(parsed[0]), a.METRICS)
        self.assertEqual(evaluations[0][0], training[0][0])
        self.assertEqual(selected, ['0'])


if __name__ == '__main__':
    unittest.main()
