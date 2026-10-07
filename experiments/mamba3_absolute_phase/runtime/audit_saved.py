"""Independent saved-byte audit: standard library only, no models or weights.

Run in a fresh process with ``python -B -m
experiments.mamba3_absolute_phase.runtime.audit_saved evidence_directory``.
The only created file is independent_audit.json; existing evidence is immutable.
Scientific metrics, replay, identities and hashes use exact comparisons. Separate
rounding envelopes below concern detached saved diagnostic arithmetic only.
"""
import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXECUTION = '11913ff8fe99aa28f124d667e2ae3aad14465098'
SOURCE = '4c404906e61121f967307e8c3e97a63e9a476711202a4a5baa005ef9899574c6'
PERIODS = (21600000, 86400000)
MODES = ('baseline_dual', 'relative_phase', 'absolute_phase')
CONTRASTS = (('absolute_phase', 'relative_phase'), ('absolute_phase', 'baseline_dual'), ('relative_phase', 'baseline_dual'))
LINEAGE_PATHS = tuple('experiments/mamba3_absolute_phase/'+path for path in
                     ('config.py', 'provenance.py', 'tests/test_failure_records.py', 'tests/test_protocol.py'))
METRICS = {f'{kind}@{k}' for kind in ('hit', 'ndcg', 'recall') for k in (5, 10, 20, 50)}
PAIRING = ('initial_backbone_sha256', 'initial_common_calibrator_hashes',
           'rng_components', 'protocol', 'manifest_sha256', 'train_time_stats_sha256',
           'verified_history_stats', 'precision', 'optimizer_settings',
           'first_train_batch_sha256')
REMOTE = Path('/home/daryumin/iberdov/diplom')
QUANTILES = ('p00', 'p25', 'p50', 'p75', 'p90', 'p99', 'p100')


def require(condition, label):
    if not condition:
        raise ValueError(label)


def integer(value, label, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum and
            (maximum is None or value <= maximum), label)
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return value


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return True


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return struct.unpack('<f', struct.pack('<I', (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000))[0]


def flatten(value):
    if isinstance(value, list):
        return [x for part in value for x in flatten(part)]
    return [value]


def tensor(value, shape, label, fp32=False):
    if shape:
        require(isinstance(value, list) and len(value) == shape[0], label + '.shape')
        for i, row in enumerate(value):
            tensor(row, shape[1:], label + '.' + str(i), fp32)
    else:
        number(value, label)
        require(not fp32 or f32(value) == value, label + '.fp32')
    return flatten(value)


class Arithmetic:
    """Bounded diagnostics checks, never an alternative GPU admission policy."""
    def __init__(self):
        self.differences = []
        self.total_differences = 0
        self.by_reason = {}

    def record(self, value):
        self.total_differences += 1
        reason = value['reason']
        row = self.by_reason.setdefault(reason, dict(count=0, maximum_absolute_difference=0.))
        row['count'] += 1
        row['maximum_absolute_difference'] = max(row['maximum_absolute_difference'], abs(value['saved']-value['derived']))
        if len(self.differences) < 200:
            self.differences.append(value)

    def bounded(self, actual, expected, bound, label, reason):
        number(actual, label); number(expected, label)
        require(abs(actual - expected) <= bound, label + ': diagnostic arithmetic mismatch')
        if actual != expected:
            self.record(dict(field=label, saved=actual, derived=expected,
                absolute_difference=abs(actual-expected), absolute_bound=bound, reason=reason))

    def close(self, actual, expected, label):
        self.bounded(actual, expected, max(1e-15, 1e-12 * max(abs(actual), abs(expected))),
                     label, 'float64 diagnostic reduction order')

    def dot(self, actual, terms, label):
        # Four FP32 products and reduction, including possible FMA contraction.
        self.bounded(actual, math.fsum(terms), 8 * 2**-24 * math.fsum(abs(x) for x in terms) + 2**-149,
                     label, 'four-term FP32 product/reduction error bound')

    def tanh(self, actual, before, label):
        expected = f32(math.tanh(before))
        bits = lambda x: struct.unpack('<I', struct.pack('<f', abs(x)))[0]
        require(actual == 0 or expected == 0 or (actual > 0) == (expected > 0), label + '.sign')
        require(abs(bits(actual)-bits(expected)) <= 2, label + '.tanh_fp32_2ULP')
        if actual != expected:
            self.record(dict(field=label, saved=actual, derived=expected,
                fp32_ulps=abs(bits(actual)-bits(expected)), reason='CPU/CUDA FP32 tanh versus rounded libm'))

    def increment(self, actual, expected, dt, label):
        # Diagnostic local increments only. A pair of tanh implementations,
        # FP32 products and subtraction can differ, especially near cancellation.
        self.bounded(actual, expected, 32 * 2**-24 * math.pi * abs(dt) + 2**-145,
                     label, 'FP32 local tanh/product/subtraction propagation; not recurrence acceptance')

    def result(self):
        return dict(count=self.total_differences, first_200_differences=self.differences, by_reason=self.by_reason,
            reduction_relative_tolerance=1e-12, reduction_absolute_tolerance=1e-15,
            tanh_fp32_max_ulps=2, dot_sum_abs_multiplier=8*2**-24,
            increment_abs_multiplier_pi_abs_dt=32*2**-24,
            scope='Detached diagnostic arithmetic only; all metrics, replay, hashes and frozen GPU acceptance remain exact')


def scalar_stats(row, label, arithmetic, count=None, values=None):
    require(set(row) == {'count', 'sum', 'squared_sum', 'mean', 'rms', 'min', 'max',
                         'abs_max', 'zero_count', 'zero_fraction'}, label + '.keys')
    n = integer(row['count'], label + '.count')
    if count is not None:
        require(n == count, label + '.population')
    zero = integer(row['zero_count'], label + '.zeros', maximum=n)
    total = number(row['sum'], label); squares = number(row['squared_sum'], label)
    require(squares >= 0, label + '.squares')
    if not n:
        require(total == squares == zero == 0 and all(row[k] is None for k in
                ('mean', 'rms', 'min', 'max', 'abs_max', 'zero_fraction')), label + '.empty')
        return
    low, high = number(row['min'], label), number(row['max'], label)
    require(low <= high and row['mean'] == total/n and row['rms'] == math.sqrt(squares/n)
            and row['abs_max'] == max(abs(low), abs(high)) and row['zero_fraction'] == zero/n,
            label + '.algebra')
    require(low-1e-12*max(1, abs(low)) <= row['mean'] <= high+1e-12*max(1, abs(high)), label + '.mean_range')
    require(abs(row['mean']) <= row['rms']+1e-12*max(1, row['rms']), label + '.Cauchy_bound')
    if values is not None:
        require(n == len(values) and min(values) == low and max(values) == high and
                sum(x == 0 for x in values) == zero, label + '.full_saved_values')
        arithmetic.bounded(total, math.fsum(values), max(1e-15, n*2**-52*math.fsum(map(abs, values))),
                           label + '.sum', 'signed FP64 reduction bound based on sum of magnitudes')
        arithmetic.close(squares, math.fsum(x*x for x in values), label + '.squared_sum')


def local_increment(before, after, dt):
    native = f32(f32(f32(math.pi)*f32(math.tanh(after)))*dt) - f32(f32(f32(math.pi)*f32(math.tanh(before)))*dt)
    effective = f32(bf16(after)-bf16(before))
    counter = f32(f32(f32(math.pi)*f32(f32(math.tanh(bf16(after)))-f32(math.tanh(bf16(before)))))*dt)
    return f32(native), effective, counter


def approximate_stats(row, values, element_bound, label, arithmetic):
    """Propagate diagnostic FP32 libm uncertainty through full saved grids."""
    n = len(values)
    reduction = n*2**-52*math.fsum(map(abs, values))
    arithmetic.bounded(row['sum'], math.fsum(values), n*element_bound+reduction,
                       label+'.derived_sum', 'propagated per-element FP32 diagnostic increment bound')
    square_bound = math.fsum(2*abs(x)*element_bound+element_bound**2 for x in values)
    square_bound += n*2**-52*math.fsum(x*x for x in values)
    arithmetic.bounded(row['squared_sum'], math.fsum(x*x for x in values), square_bound,
                       label+'.derived_squares', 'propagated per-element FP32 diagnostic increment bound')
    for key, expected in [('min', min(values)), ('max', max(values))]:
        arithmetic.bounded(row[key], expected, element_bound, label+'.derived_'+key,
                           'propagated per-element FP32 diagnostic increment bound')
    require(row['zero_count'] <= sum(abs(x) <= element_bound for x in values), label+'.possible_zero_count')


def analytic_grid(grid, weight, hours, label, arithmetic):
    n = len(hours)
    require(grid['hours'] == hours and grid['clock_ms'] == [h*3600000 for h in hours] and
            grid['feature_dtype'] == 'float32 after float64 remainder/sin/cos', label + '.clock')
    phi = grid['features']; tensor(phi, [n, 4], label + '.features', True)
    for i, h in enumerate(hours):
        expected = []
        for period in PERIODS:
            angle = ((h*3600000) % period) * (2*math.pi/period)
            expected += [f32(math.sin(angle)/math.sqrt(2)), f32(math.cos(angle)/math.sqrt(2))]
        for j, value in enumerate(phi[i]):
            arithmetic.bounded(value, expected[j], 2**-24, label + f'.feature.{i}.{j}', 'FP64 libm feature rounded to FP32')
    for key in ('frequency_6h_linear_contribution', 'frequency_24h_linear_contribution',
                'linear_preactivation', 'raw_angle_correction'):
        tensor(grid[key], [n, 32], label + '.' + key, True)
    for i, features in enumerate(phi):
        for j, w in enumerate(weight):
            for key, indices in (('frequency_6h_linear_contribution', range(2)),
                                 ('frequency_24h_linear_contribution', range(2, 4)),
                                 ('linear_preactivation', range(4))):
                arithmetic.dot(grid[key][i][j], [features[k]*w[k] for k in indices], label + f'.{key}.{i}.{j}')
            arithmetic.tanh(grid['raw_angle_correction'][i][j], grid['linear_preactivation'][i][j], label + f'.delta.{i}.{j}')
    for field, key in (('frequency_6h_stats', 'frequency_6h_linear_contribution'),
                       ('frequency_24h_stats', 'frequency_24h_linear_contribution'),
                       ('raw_angle_correction_stats', 'raw_angle_correction')):
        scalar_stats(grid[field], label + '.' + field, arithmetic, n*32, flatten(grid[key]))
    deltas = flatten(grid['raw_angle_correction'])
    saturation = sum(abs(x) >= f32(.99) for x in deltas)
    require(grid['abs_delta_near_saturation_count'] == saturation, label + '.saturation')
    arithmetic.close(grid['abs_delta_near_saturation_fraction'], saturation/len(deltas), label + '.saturation_fraction')
    fixtures = grid['content_dt_fixtures']
    require([(x['content_raw_angle'], x['dt_phase']) for x in fixtures] ==
            [(-2., .05), (-.5, .25), (0., .5), (.5, 1.), (2., 2.)], label + '.fixed_fixtures')
    for index, fixture in enumerate(fixtures):
        prefix = label + '.fixture' + str(index)
        before, dt = f32(fixture['content_raw_angle']), f32(fixture['dt_phase'])
        require(fixture['sampled_angle_indices'] == [0, 1, 2, 3] and fixture['samples_scope'] ==
                'all grid clocks, first four angle coordinates; stats cover every angle', prefix + '.sample_scope')
        for key in ('native_increment_difference', 'counterfactual_bf16_effective_raw_correction',
                    'counterfactual_bf16_increment_difference'):
            tensor(fixture[key], [n, 4], prefix + '.' + key, True)
        lost = 0
        derived = {k: [] for k in ('native_increment_difference_stats', 'counterfactual_bf16_increment_difference_stats',
                                   'counterfactual_bf16_rounding_increment_error_stats')}
        for i, row in enumerate(grid['raw_angle_correction']):
            for j, delta in enumerate(row):
                native, effective, counter = local_increment(before, f32(before+delta), dt)
                lost += delta != 0 and effective == 0
                derived['native_increment_difference_stats'].append(native)
                derived['counterfactual_bf16_increment_difference_stats'].append(counter)
                derived['counterfactual_bf16_rounding_increment_error_stats'].append(f32(counter-native))
                if j < 4:
                    require(fixture['counterfactual_bf16_effective_raw_correction'][i][j] == effective, prefix + '.bf16_cast')
                    arithmetic.increment(fixture['native_increment_difference'][i][j], native, dt, prefix + f'.native.{i}.{j}')
                    arithmetic.increment(fixture['counterfactual_bf16_increment_difference'][i][j], counter, dt, prefix + f'.counterfactual.{i}.{j}')
        require(fixture['counterfactual_nonzero_fp32_corrections_lost_in_bf16'] == lost, prefix + '.lost_count')
        for key in ('native_increment_difference_stats', 'counterfactual_bf16_increment_difference_stats',
                    'counterfactual_bf16_rounding_increment_error_stats'):
            scalar_stats(fixture[key], prefix + '.' + key, arithmetic, n*32)
            bound = (96 if 'rounding' in key else 32)*2**-24*math.pi*abs(dt)+2**-145
            approximate_stats(fixture[key], derived[key], bound, prefix+'.'+key, arithmetic)
        for key in ('native_increment_difference', 'counterfactual_bf16_increment_difference'):
            stats = fixture[key + '_stats']
            require(all(stats['min'] <= x <= stats['max'] for x in flatten(fixture[key])), prefix + '.sample_extrema')


def phase_diagnostics(d, mode, label, arithmetic, coverage_summary):
    require(finite(d) and set(d) == {'observed', 'parameters'}, label + '.phase_structure')
    p, observed = d['parameters'], d['observed']
    require(p['status'] == 'MEASURED' and p['phase_mode'] == mode and p['extra_parameters'] == 128 and
            p['W_shape'] == [32, 4] and p['W_dtype'] == 'torch.float32' and
            p['sharing'] == 'one W shared across both layers and both heads' and
            p['train_coverage_summary'] == coverage_summary, label + '.W_identity')
    weights = tensor(p['W'], [32, 4], label + '.W', True)
    arithmetic.close(p['W_l2_norm'], math.sqrt(math.fsum(x*x for x in weights)), label + '.W_norm')
    require(p['W_abs_max'] == max(map(abs, weights)), label + '.W_max')
    for key, indices in [('6h', range(2)), ('24h', range(2, 4))]:
        arithmetic.close(p['frequency_pair_l2_norms'][key], math.sqrt(math.fsum(row[j]**2 for row in p['W'] for j in indices)), label + '.W_' + key)
    grad = p['gradients']
    require(grad['scope'] == 'last available TRAIN backward at VALID boundary; not all optimizer steps' and
            grad['status'] == 'MEASURED' and grad['finite'] is True and number(grad['l2_norm'], label) >= 0,
            label + '.gradient_snapshot')
    integer(grad['nonzero_count'], label + '.gradient_nonzero', maximum=128)
    require('counterfactual only' in p['raw_angle_dtype'] and 'before shared tanh' in p['frequency_contribution_definition'], label + '.diagnostic_interpretation')
    analytic_grid(p['absolute_24h_grid'], p['W'], list(range(24)), label + '.absolute_grid', arithmetic)
    analytic_grid(p['relative_grid'], p['W'], [0, 1, 3, 6, 12, 24, 48], label + '.relative_grid', arithmetic)
    require(observed['status'] == 'MEASURED' and observed['phase_mode'] == mode and
            observed['training_rng_consumed'] is False and observed['additional_dataset_forwards'] == 0 and
            observed['accumulated_theta'] == observed['full_history_hidden_states'] == 'NOT_RECORDED' and
            observed['saturation_abs_delta'] == .99 and set(observed['layers']) == {'0', '1'} and
            observed['scope'] == 'existing VALID forwards only; last active observed event per input history, separately for each layer' and
            observed['bf16_scope'].startswith('COUNTERFACTUAL raw-angle BF16 cast only; native MIMO angles stay FP32.'), label + '.observed_scope')
    stat_keys = {'raw_angle_correction', 'raw_angles_before', 'raw_angles_corrected', 'native_increment_before_fp32',
        'native_increment_after_fp32', 'native_increment_difference_fp32', 'counterfactual_effective_raw_correction_bf16',
        'counterfactual_native_increment_difference_bf16', 'counterfactual_bf16_correction_rounding_error',
        'counterfactual_bf16_increment_rounding_error'}
    common = []
    for index, layer in observed['layers'].items():
        here = label + '.layer' + index
        integer(layer['forward_calls'], here + '.forward_calls', 1)
        require(layer['histories'] == 23951, here + '.VALID_population')
        q = integer(layer['selected_active_queries'], here + '.queries', maximum=23951)
        require(integer(layer['histories_without_active_events'], here) + q == 23951 and
                q <= integer(layer['observed_active_positions'], here) <= 23951*49, here + '.scope_counts')
        require(set(layer['stats']) == stat_keys, here + '.stat_keys')
        for key, row in layer['stats'].items():
            scalar_stats(row, here + '.' + key, arithmetic, q*32 if key == 'raw_angle_correction' else q*64)
        near = integer(layer['abs_delta_near_saturation_count'], here, maximum=q*32)
        require(layer['abs_delta_near_saturation_fraction'] == (near/(q*32) if q else None), here + '.saturation_fraction')
        expanded = 2*(q*32-layer['stats']['raw_angle_correction']['zero_count'])
        lost = integer(layer['counterfactual_nonzero_fp32_corrections_lost_in_bf16'], here, maximum=expanded)
        require(layer['nonzero_expanded_corrections'] == expanded and layer['counterfactual_bf16_lost_nonzero_fraction'] ==
                (lost/expanded if expanded else None), here + '.counterfactual_denominator')
        examples = layer['examples']; require(len(examples) == min(q, 8), here + '.sample_count')
        sample_common = []
        for i, row in enumerate(examples):
            at = here + '.example' + str(i)
            integer(row['query_position'], at, 1, 49)
            for key, shape in [('features', [4]), ('raw_angle_correction', [32]), ('raw_angles_before', [2,32]),
                               ('raw_angles_corrected', [2,32]), ('dt_phase', [2]),
                               ('counterfactual_effective_raw_correction_bf16', [2,32]),
                               ('counterfactual_native_increment_difference_bf16', [2,32])]:
                tensor(row[key], shape, at + '.' + key, True)
            require(all(x > 0 for x in row['dt_phase']), at + '.positive_DT')
            for k in (0, 2):
                arithmetic.bounded(row['features'][k]**2+row['features'][k+1]**2, .5, 2**-22,
                                   at+'.basis_norm'+str(k), 'FP32 sine/cosine feature pair norm')
            for j, delta in enumerate(row['raw_angle_correction']):
                # The GPU matmul and tanh were not saved between operations.
                dot = math.fsum(row['features'][k]*p['W'][j][k] for k in range(4))
                bound = 8*2**-24*math.fsum(abs(row['features'][k]*p['W'][j][k]) for k in range(4)) + 2**-22
                arithmetic.bounded(delta, math.tanh(dot), bound, at + '.delta' + str(j), 'four-term FP32 matmul and tanh')
                for head in range(2):
                    before, after = row['raw_angles_before'][head][j], row['raw_angles_corrected'][head][j]
                    require(after == f32(before+delta), at + '.raw_addition')
                    _, effective, counter = local_increment(before, after, row['dt_phase'][head])
                    require(row['counterfactual_effective_raw_correction_bf16'][head][j] == effective, at + '.bf16_cast')
                    arithmetic.increment(row['counterfactual_native_increment_difference_bf16'][head][j], counter,
                                         row['dt_phase'][head], at + '.counterfactual')
            for key in ('raw_angle_correction', 'raw_angles_before', 'raw_angles_corrected',
                        'counterfactual_effective_raw_correction_bf16', 'counterfactual_native_increment_difference_bf16'):
                stats = layer['stats'][key]
                require(all(stats['min'] <= x <= stats['max'] for x in flatten(row[key])), at + '.sample_extrema.' + key)
            sample_common.append({k: row[k] for k in ('query_position', 'features', 'raw_angle_correction')})
        common.append((layer['forward_calls'], layer['histories_without_active_events'], layer['observed_active_positions'], q,
                       layer['stats']['raw_angle_correction'], sample_common))
    require(common[0] == common[1], label + '.shared_layer_features_and_correction')


def parse_logs(text):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    evaluations = re.findall(r'epoch (\d+) evaluating \[time: ([0-9.]+)s, valid_score: ([0-9.]+)\]', text)
    metrics = [dict((k, float(v)) for k, v in re.findall(r'((?:hit|ndcg|recall)@\d+) : ([0-9.]+)', line))
               for line in text.splitlines() if line.startswith('hit@5 :')]
    training = re.findall(r'epoch (\d+) training \[time: ([0-9.]+)s, train loss: ([0-9.]+)\]', text)
    finishes = re.findall(r'Finished training, best eval result in epoch (\d+)\b', text)
    return evaluations, metrics, training, finishes


def runtime_identity(record, historical, manifest, label, gpu):
    actual = record['runtime']
    keys = ('torch', 'recbole', 'mamba_ssm', 'triton', 'numpy', 'tilelang',
            'apache_tvm_ffi', 'cuda', 'pinned_commit', 'upstream_manifest_sha256')
    require(all(actual.get(k) == historical[k] for k in keys), label + '.runtime_dependencies')
    require((isinstance(actual.get('gpu'), str) and 'A100' in actual['gpu']) if gpu else actual.get('gpu') is None,
            label + '.runtime_device')
    mappings = [actual['imported_sources']]
    if 'imported_sources' in record:
        mappings.append(record['imported_sources'])
    for mapping in mappings:
        require(isinstance(mapping, dict) and bool(mapping), label + '.imported_sources')
        for name, path in mapping.items():
            require(name.startswith('experiments.') and path in manifest['files'] and
                    path in (name.replace('.', '/') + '.py', name.replace('.', '/') + '/__init__.py'),
                    label + '.imported_source.' + name)


def child_environment_check(allocation, children, expected):
    """Bind every saved GPU child to the original allocation visibility.

    Preserve an absent scheduler mask as absent; never replace it with an
    assumed device index. Per-seed path identity distinguishes repeated modes.
    """
    require(isinstance(allocation, dict) and set(allocation) == {'present', 'value'} and
            type(allocation['present']) is bool and
            (isinstance(allocation['value'], str) if allocation['present'] else allocation['value'] is None),
            'Allocation CUDA visibility schema')
    require(set(children) == set(expected), 'Exactly planned child environment records')
    for path, stage in expected.items():
        row = children[path]
        require(isinstance(row, dict) and set(row) == {'stage', 'parent_cuda_visibility', 'child_cuda_visibility'} and
                row['stage'] == stage, 'Child environment stage/path identity ' + path)
        require(row['parent_cuda_visibility'] == allocation and row['child_cuda_visibility'] == allocation,
                'Original allocation CUDA mask preserved in parent and child ' + path)
    return dict(allocation_cuda_visibility=allocation, children_verified=len(children),
                child_paths=list(expected), original_mask_preserved=True,
                scope='Saved environment at child launch; absent masks remain absent, no assumed physical GPU index')


def coverage_check(row, arithmetic, digest):
    require(finite(row), 'TRAIN coverage finite')
    scope = dict(stage='TRAIN_PHASE_COVERAGE', split='TRAIN', sampled_examples=10000, train_population=1062567,
        periods_ms=list(PERIODS), phase_bins=24, max_history=50, max_examples=10000, batch_size=256,
        scientific_fit_started=False, scientific_fits_started=0, model_instances_created=0, model_forward_count=0,
        train_loaders_created=0, valid_loaders_created=0, test_loaders_created=0, target_fields_read=False,
        test_evaluation_count=0, cuda_visible_devices='', cuda_initialized_before=False,
        cuda_initialized_after=False, cuda_available=False, training_rng_consumed=False, examples=10000)
    require(all(row.get(k) == v for k, v in scope.items()), 'TRAIN-only coverage scope')
    indices = [i*(1062567-1)//9999 for i in range(10000)]
    require(row['selected_train_indices'] == indices and row['selected_indices_sha256'] == digest(indices) and
        row['index_rule'] == 'floor(i*(N-1)/(n-1)) for i=0..n-1; n=min(N,10000); [0] when n=1', 'TRAIN deterministic sample')
    hashes = row['selected_input_hashes']
    require(set(hashes) == {'item_history', 'history_length', 'precise_history_timestamps'} and
            all(re.fullmatch('[0-9a-f]{64}', v) for v in hashes.values()) and
            row['selected_records_sha256'] == digest(hashes), 'TRAIN input hash binding')
    semantics = row['timestamp_semantics']
    require(all(semantics[k] is True for k in ('finite', 'integral_milliseconds', 'within_exact_integer_range')) and
            semantics['dtype'] == 'float64' and semantics['unit'] == 'milliseconds since Unix epoch' and
            semantics['origin'] == 'Unix epoch, fixed globally' and semantics['user_timezone'] == 'NOT_RECORDED' and
            semantics['local_time_interpretation'] is False, 'TRAIN timestamp semantics')
    valid = integer(row['valid_event_occurrences'], 'TRAIN valid', 10000, 500000)
    active = integer(row['active_event_occurrences'], 'TRAIN active', 1, 490000)
    require(valid == active+10000 and row['first_event_occurrences'] == 10000 and
            row['padding_occurrences']+valid == 500000, 'TRAIN occurrence accounting')
    low, high = row['timestamp_min_ms'], row['timestamp_max_ms']
    require(0 < low <= high <= 2**53-1 and int(low) == low and int(high) == high and
            row['range_ms'] == high-low >= 86400000, 'TRAIN exact Unix-ms range')
    require(row['utc_range'] == [datetime.fromtimestamp(x/1000, timezone.utc).isoformat() for x in (low, high)], 'TRAIN UTC range')
    distinct = integer(row['distinct_timestamps'], 'TRAIN distinct', 1, valid)
    require(row['repeated_timestamp_occurrence_fraction'] == 1-distinct/valid, 'TRAIN duplicate fraction')
    require(0 <= row['occurrences_with_nonunique_timestamp_fraction'] <= 1, 'TRAIN repeated occurrence bounds')
    equal = integer(row['equal_adjacent_timestamp_count'], 'TRAIN equal gaps', maximum=active)
    require(row['equal_adjacent_timestamp_fraction'] == equal/active, 'TRAIN equal gap fraction')
    integer(row['negative_adjacent_gap_count'], 'TRAIN negative gaps', maximum=active)
    if row['negative_adjacent_gap_count'] == 0:
        require(row['chronological_history_span_ms'] == row['clamped_cumulative_history_span_ms'], 'TRAIN nonnegative gap span equality')
    for key in ('chronological_history_span_ms', 'clamped_cumulative_history_span_ms', 'history_length'):
        stats = row[key]; require(stats['count'] == 10000 and stats['min'] <= stats['mean'] <= stats['max'], 'TRAIN distribution ' + key)
        values = [number(stats['quantiles'][q], key) for q in QUANTILES]
        require(values == sorted(values) and values[0] == stats['min'] and values[-1] == stats['max'], 'TRAIN quantiles ' + key)
        if key == 'history_length':
            require(1 <= stats['min'] <= stats['max'] <= 50, 'TRAIN history range')
            arithmetic.close(stats['mean'], valid/10000, 'TRAIN mean history length')
    require([p['period_ms'] for p in row['periods']] == list(PERIODS), 'TRAIN periods')
    for p in row['periods']:
        period = p['period_ms']; label = 'TRAIN period ' + str(period)
        require(p['span_in_periods'] == (high-low)/period and p['complete_period_lengths_in_range'] == int((high-low)//period)
                and p['phase_bins'] == 24 and p['bin_edges_fraction'] == [i/24 for i in range(25)], label + '.range')
        for key, total in [('valid_bin_counts', valid), ('active_bin_counts', active)]:
            require(len(p[key]) == 24 and sum(integer(x, label) for x in p[key]) == total, label + '.bins')
        require(all(a <= b for a, b in zip(p['active_bin_counts'], p['valid_bin_counts'])) and
                p['occupied_valid_bins'] == sum(x > 0 for x in p['valid_bin_counts']) and
                p['occupied_active_bins'] == sum(x > 0 for x in p['active_bin_counts']), label + '.occupied')
        unique = integer(p['unique_exact_phases'], label, 2, min(valid, period))
        integer(p['unique_active_exact_phases'], label, 2, min(unique, active))
        repeats = integer(p['unique_exact_phases_on_multiple_dates'], label, maximum=unique)
        arithmetic.close(p['exact_phase_repeat_fraction'], repeats/unique, label + '.repeat_fraction')
        dates = p['by_utc_date']; days = [d['unix_day'] for d in dates]
        require(len(days) >= 2 and days == sorted(set(days)) and days[0] == int(low//86400000) and
                days[-1] == int(high//86400000), label + '.dates')
        for d in dates:
            require(d['utc_date'] == datetime.fromtimestamp(d['unix_day']*86400, timezone.utc).date().isoformat() and
                    len(d['bin_counts']) == 24 and sum(integer(x, label) for x in d['bin_counts']) == d['occurrences'], label + '.date_bins')
        require([sum(d['bin_counts'][j] for d in dates) for j in range(24)] == p['valid_bin_counts'] and
                p['dates_per_phase_bin'] == [sum(d['bin_counts'][j] > 0 for d in dates) for j in range(24)] and
                p['bins_seen_on_multiple_dates'] == sum(x > 1 for x in p['dates_per_phase_bin']), label + '.date_aggregation')
    expected = {k: row[k] for k in ('status', 'scope', 'sampled_examples', 'valid_event_occurrences', 'active_event_occurrences',
        'timestamp_min_ms', 'timestamp_max_ms', 'utc_range', 'range_ms', 'equal_adjacent_timestamp_fraction',
        'negative_adjacent_gap_count', 'chronological_history_span_ms', 'clamped_cumulative_history_span_ms',
        'selected_records_sha256', 'decision_rule', 'blocking_reason')}
    expected['periods'] = [{k: p[k] for k in ('period_ms', 'span_in_periods', 'complete_period_lengths_in_range',
        'occupied_valid_bins', 'occupied_active_bins', 'unique_active_exact_phases', 'unique_exact_phases_on_multiple_dates',
        'bins_seen_on_multiple_dates')} for p in row['periods']]
    require(row['status'] == 'PASS' and row['blocking_reason'] is None and row['summary'] == expected, 'TRAIN compact summary')


def packed_check(value, label='gate', counter=None):
    if counter is None:
        counter = [0]
    if isinstance(value, dict):
        if all(k in value for k in ('shape', 'dtype', 'sha256', 'values')):
            values = tensor(value['values'], value['shape'], label)
            dtype = value['dtype']
            codes = {'torch.float64': 'd', 'torch.float32': 'f', 'torch.float16': 'e',
                     'torch.int64': 'q', 'torch.int32': 'i', 'torch.bool': '?'}
            if dtype == 'torch.bfloat16':
                require(all(bf16(x) == x for x in values), label + '.bf16_representable')
                blob = b''.join(struct.pack('<H', struct.unpack('<I', struct.pack('<f', x))[0] >> 16) for x in values)
            else:
                require(dtype in codes, label + '.known_dtype')
                blob = b''.join(struct.pack('<'+codes[dtype], x) for x in values)
            require(hashlib.sha256(blob).hexdigest() == value['sha256'], label + '.numeric_bytes_SHA')
            counter[0] += 1
        else:
            for key, row in value.items():
                packed_check(row, label + '.' + key, counter)
    elif isinstance(value, list):
        for i, row in enumerate(value):
            packed_check(row, label + '.' + str(i), counter)
    return counter[0]


def measured_difference(left, right, saved, label, arithmetic, informative=False, native=False):
    require(left['shape'] == right['shape'], label + '.shape')
    a, b = flatten(left['values']), flatten(right['values'])
    errors = [x-y for x, y in zip(a, b)]
    norms = dict(l2_error=math.sqrt(math.fsum(x*x for x in errors)),
                 reference_norm=math.sqrt(math.fsum(x*x for x in b)))
    mixed = all(abs(x-y) <= saved['atol']+saved['rtol']*abs(y) for x, y in zip(a, b))
    relative = norms['l2_error']/norms['reference_norm'] if norms['reference_norm'] else None
    capped = 'relative_l2_cap' not in saved or (relative <= saved['relative_l2_cap'] if relative is not None else norms['l2_error'] == 0)
    if native:
        require(saved['passed'] is (mixed and capped) and saved['mixed_pass'] is mixed and saved['norm_cap_pass'] is capped and
                saved['finite'] is True and saved['actual_dtype'] == left['dtype'] and saved['reference_dtype'] == right['dtype'], label + '.acceptance')
        arithmetic.close(saved['actual_norm'], math.sqrt(math.fsum(x*x for x in a)), label + '.actual_norm')
        if relative is None:
            require(saved['relative_l2'] is None, label + '.zero_reference')
        else:
            arithmetic.close(saved['relative_l2'], relative, label + '.relative_l2')
    else:
        require(saved['atol'] == 1e-6 and saved['rtol'] == 1e-5 and saved['shape'] == left['shape'] and
                saved['dtype'] == left['dtype'] == right['dtype'], label + '.frozen_invariance_policy')
        require(saved['within_invariance_tolerance' if informative else 'passed'] is mixed and mixed is not informative and
                saved['bitwise_equal'] is (left['dtype'] == right['dtype'] and left['sha256'] == right['sha256']), label + '.acceptance')
    require(saved['max_abs'] == max(map(abs, errors), default=0), label + '.max_abs')
    arithmetic.close(saved['mean_abs'], math.fsum(map(abs, errors))/len(errors) if errors else 0, label + '.mean_abs')
    for key, expected in norms.items():
        arithmetic.close(saved[key], expected, label + '.' + key)


def gate_numerics(gate, arithmetic, policy):
    packed = packed_check(gate)
    cases = {r['case_id']: r for r in gate['cases']}
    for mode in MODES:
        case = cases[mode+'_shift']; fixtures = case['numeric_fixture']; checks = case['checks']
        require(set(fixtures) == {'original', 'plus_2h', 'plus_24h'}, mode + '.shift_labels')
        original = fixtures['original']
        for name, offset in [('original', 0), ('plus_2h', 7200000), ('plus_24h', 86400000)]:
            row = fixtures[name]
            require(flatten(row['timestamps']['values']) == [x+offset for x in flatten(original['timestamps']['values'])], mode + '.timestamp_shift')
            require(row['raw_angles']['dtype'] == row['phase_increments_fp32']['dtype'] == row['DT_phase']['dtype'] == 'torch.float32', mode + '.native_precision')
            require(row['DT_phase'] == original['DT_phase'], mode + '.unchanged_DT')
            require(all(x == 0 for x in row['correction']['values'][0][0]), mode + '.first_neutral')
            angles, increments, dt = row['raw_angles']['values'], row['phase_increments_fp32']['values'], row['DT_phase']['values']
            require(row['raw_angles']['shape'] == row['phase_increments_fp32']['shape'] == [1, 8, 2, 32] and
                    row['DT_phase']['shape'] == [1, 2, 8], mode + '.native_layout')
            for event in range(8):
                for head in range(2):
                    for angle in range(32):
                        raw_angle = angles[0][event][head][angle]; delta_t = dt[0][head][event]
                        expected = f32(f32(f32(math.pi)*f32(math.tanh(raw_angle)))*delta_t)
                        arithmetic.increment(increments[0][event][head][angle], expected, delta_t,
                                             mode+'.'+name+f'.native_increment.{event}.{head}.{angle}')
        for suffix, name in [('2h', 'plus_2h'), ('24h', 'plus_24h')]:
            for key, field in [('features', 'features'), ('correction', 'correction'), ('output', 'output')]:
                if mode == 'baseline_dual' and key == 'features' and suffix == '2h':
                    continue
                informative = mode == 'absolute_phase' and suffix == '2h'
                leaf = checks[key+'_'+suffix]
                measured_difference(fixtures[name][field], original[field], leaf['measurement'] if informative else leaf,
                                    mode+'.'+key+'_'+suffix, arithmetic, informative)
            measured_difference(case['zero_W_outputs'][name], case['zero_W_outputs']['original'], checks['zero_W_output_'+suffix],
                                mode+'.zero_W_'+suffix, arithmetic)
        theta2, theta = flatten(fixtures['plus_2h']['theta']['values']), flatten(original['theta']['values'])
        require(checks['native_phase_finite']['max_abs_2h'] == max(abs(f32(x-y)) for x, y in zip(theta2, theta)), mode + '.native_theta_delta')
        if mode != 'baseline_dual':
            require(number(checks['W_gradient_informative']['l2'], mode) > 0, mode + '.informative_W')
    reference = cases['native_recurrence_reference']
    for key, profile in [('output', 'output'), ('W_gradient', 'vjp')]:
        saved = reference['checks'][key]
        require(all(saved[k] == v for k, v in policy[profile].items()), 'Frozen native ' + profile + ' policy')
        measured_difference(reference[key], reference['reference_'+key], saved, 'native_reference.'+key, arithmetic, native=True)
    return dict(packed_tensors_rehashed=packed, shift_cases_recomputed=3, native_reference_comparisons_recomputed=2,
                native_raw_angle_dtype='torch.float32', counterfactual_bf16_is_not_actual_rotation=True)


def audit_run(r, variant, seed, files, saved, terminal, arithmetic, c, read, report, coverage):
    require(r['status'] == 'PASS' and r['stage'] == 'COMPLETED', variant + '.fit_status')
    require(not any(k in r for k in ('error', 'traceback', 'validation_error')), variant + '.fit_errors')
    report.validate_record(r, variant, seed)
    require(hashlib.sha256(json.dumps(r['config'], sort_keys=True).encode()).hexdigest() == r['config_sha256'], variant + '.config_sha')
    require(r['epoch_indexing'] == 'zero-based' and r['selection_split'] == 'VALID' and
            r['evaluation_mode'] == 'full-ranking' and r['history_length'] == 50 and
            r['kernel_length_for_max_history'] == 56, variant + '.protocol')
    require(r['effective_config_parity']['status'] == 'PASS' and
            r['effective_config_parity']['allowed_differences'] == ['checkpoint_dir', 'phase_mode', 'seed'] and
            r['effective_config_parity']['seed'] == seed and r['effective_config_parity']['cpu_reference_device'] == 'cuda', variant + '.effective_config')
    paths = c.paths(variant, seed)
    runtime = files / paths['runtime'].relative_to(c.HERE)
    meta = read(runtime / 'checkpoints/best_metadata.json')
    checkpoint = saved['checkpoints'][str(paths['checkpoint'].relative_to(c.HERE))]
    require(checkpoint == terminal['checkpoints'][r['run_id']], variant + '.terminal_checkpoint')
    require(checkpoint['sha256'] == r['checkpoint_sha256'] == meta['checkpoint_sha256'] and
            re.fullmatch('[0-9a-f]{64}', checkpoint['sha256']), variant + '.checkpoint_sha')
    expected_path = str(REMOTE / paths['checkpoint'].relative_to(c.ROOT))
    require(r['checkpoint_path'] == checkpoint['path'] == expected_path and
            r['checkpoint_metadata_path'] == str(REMOTE / paths['metadata'].relative_to(c.ROOT)), variant + '.checkpoint_path')
    integer(checkpoint['bytes'], variant + '.checkpoint_bytes', 1)
    for key in ('run_id', 'mode', 'phase_mode', 'seed', 'execution_commit', 'config_sha256', 'source_hash', 'core_hash'):
        require(meta[key] == r[key], variant + '.checkpoint_owner.' + key)
    require(meta['epoch_indexing'] == 'zero-based' and meta['epoch'] == r['best_epoch'] and
            meta['metrics'] == r['best_valid_metrics'] and meta['phase'] == r['best_diagnostics'].get('phase'), variant + '.checkpoint_selection')
    logs = list((runtime / 'log').rglob('*.log'))
    require(len(logs) == 1, variant + '.metric_log_count')
    parsed = parse_logs(logs[0].read_text())
    require(parse_logs((runtime / 'process/stderr.log').read_text()) == parsed, variant + '.process_recbole_logs')
    evaluations, metrics, training, finishes = parsed
    require(len(evaluations) == len(metrics) == len(training) == r['actual_epochs'], variant + '.log_epochs')
    scores = []
    best, stale, stop = -math.inf, 0, None
    for i, ((epoch, valid_time, score), metric, (train_epoch, train_time, loss), row) in enumerate(zip(evaluations, metrics, training, r['history'])):
        label = variant + '.epoch' + str(i)
        require(int(epoch) == int(train_epoch) == row['epoch'] == i and float(score) == row['valid_ndcg10'] and
                set(metric) == METRICS and metric == row['valid_metrics'], label + '.metrics')
        require(all(0 <= v <= 1 for v in metric.values()), label + '.metric_range')
        require(float(loss) == round(row['train_loss'], 4), label + '.loss')
        require(float(valid_time) >= 0 and float(train_time) >= 0, label + '.log_time')
        for phase in ('train', 'valid'):
            require(number(row[phase + '_seconds'], label + '.time') > 0, label + '.positive_time')
            allocated = integer(row[phase + '_peak_allocated_bytes'], label + '.allocated', 1)
            reserved = integer(row[phase + '_peak_reserved_bytes'], label + '.reserved', allocated)
        scores.append(row['valid_ndcg10'])
        if scores[-1] >= best:
            best, stale = scores[-1], 0
        else:
            stale += 1
        if stale > 10:
            require(stop is None, label + '.epochs_after_stop'); stop = i
        diag = row['diagnostics']
        require((diag['architecture'], diag['mimo_rank'], diag['temporal_heads'], diag['layers'], diag['calibrator_count']) ==
                ('MIMO', 4, 2, 2, 2), label + '.backbone_diagnostics')
        if variant == 'baseline_dual':
            require('phase' not in diag and 'phase_gradient_checks' not in row, label + '.baseline_diagnostics')
        else:
            gradient = row['phase_gradient_checks']
            require(gradient['backward_calls'] == 519 and gradient['finite'] is True and
                    gradient['hook_removed'] is True, label + '.all_TRAIN_backward_checks')
            integer(gradient['nonzero_calls'], label + '.nonzero_backward_calls', maximum=519)
            phase_diagnostics(diag['phase'], variant, label, arithmetic, coverage['summary'])
    require(stop == len(scores) - 1 if len(scores) < 300 else stop in (None, 299), variant + '.early_stop')
    selected = max(i for i, value in enumerate(scores) if value == max(scores))
    require(r['best_epoch'] == selected and r['best_valid_score'] == max(scores), variant + '.last_tie')
    require(finishes == ([str(selected)] if stop is not None else []), variant + '.finished_selection_log')
    require(r['first27_complete'] is (len(scores) >= 27) and
            r['first27_best_ndcg10'] == (max(scores[:27]) if len(scores) >= 27 else None), variant + '.first27')
    for key in ('train_seconds', 'valid_seconds'):
        require(r[key] == sum(x[key] for x in r['history']), variant + '.' + key)
    for kind in ('allocated', 'reserved'):
        require(r['peak_gpu_' + kind + '_bytes'] == max(max(x['train_peak_' + kind + '_bytes'], x['valid_peak_' + kind + '_bytes']) for x in r['history']), variant + '.peak_' + kind)
    row = dict(variant=variant, seed=seed, run_id=r['run_id'], parameters=r['parameter_count'], epochs=r['actual_epochs'],
               metric_cells=12 * r['actual_epochs'], checkpoint_sha256=r['checkpoint_sha256'], checkpoint_bytes=checkpoint['bytes'],
               best_epoch=selected, ndcg10=r['best_valid_metrics']['ndcg@10'], hr10=r['best_valid_metrics']['hit@10'],
               first27_complete=r['first27_complete'], first27_best_ndcg10=r['first27_best_ndcg10'],
               train_seconds=r['train_seconds'], valid_seconds=r['valid_seconds'],
               peak_allocated_bytes=r['peak_gpu_allocated_bytes'], peak_reserved_bytes=r['peak_gpu_reserved_bytes'])
    if variant != 'baseline_dual':
        diag = [x['diagnostics']['phase'] for x in r['history']]
        row['phase_boundaries'] = dict(
            best_W_l2_norm=diag[selected]['parameters']['W_l2_norm'], final_W_l2_norm=diag[-1]['parameters']['W_l2_norm'],
            maximum_W_l2_norm=max(x['parameters']['W_l2_norm'] for x in diag),
            best_frequency_pair_l2_norms=diag[selected]['parameters']['frequency_pair_l2_norms'],
            final_frequency_pair_l2_norms=diag[-1]['parameters']['frequency_pair_l2_norms'],
            best_VALID_layers={k: {field: v[field] for field in ('selected_active_queries', 'abs_delta_near_saturation_fraction',
                'counterfactual_bf16_lost_nonzero_fraction')} for k, v in diag[selected]['observed']['layers'].items()},
            interpretation='Descriptive VALID-boundary diagnostics; not feature importance or actual downstream BF16 rotation')
    return row


def moments(values):
    require(bool(values) and all(math.isfinite(x) for x in values), 'Nonempty finite aggregate')
    mean = math.fsum(values)/len(values)
    return dict(n=len(values), mean=mean,
                sample_std=math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1)) if len(values) > 1 else None)


def aggregates(records, seeds, label):
    result = dict(label=label, seeds=list(seeds), per_variant={}, contrasts=[], first27_subsets=[])
    for mode in MODES:
        selected = [records[mode, seed] for seed in seeds]
        result['per_variant'][mode] = {metric: moments([r['best_valid_metrics'][metric] for r in selected])
                                      for metric in sorted(METRICS)}
    for left, right in CONTRASTS:
        a = [records[left, seed]['best_valid_metrics']['ndcg@10'] for seed in seeds]
        b = [records[right, seed]['best_valid_metrics']['ndcg@10'] for seed in seeds]
        deltas = [x-y for x, y in zip(a, b)]
        denominator = moments(b)['mean']
        row = dict(comparison=left+' - '+right, seeds=list(seeds), deltas=deltas, **moments(deltas),
                   signs={'+': sum(x > 0 for x in deltas), '-': sum(x < 0 for x in deltas), '0': sum(x == 0 for x in deltas)},
                   relative_difference_of_means_percent=100*(moments(a)['mean']-denominator)/denominator if denominator else None)
        result['contrasts'].append(row)
        subset = [s for s in seeds if records[left, s]['first27_complete'] and records[right, s]['first27_complete']]
        first = dict(comparison=left+' - '+right, seeds=subset, omitted_seeds=[s for s in seeds if s not in subset],
                     label='Matched complete epochs 0–26 only; no imputation of short fits')
        if subset:
            av = [records[left, s]['first27_best_ndcg10'] for s in subset]
            bv = [records[right, s]['first27_best_ndcg10'] for s in subset]
            differences = [x-y for x, y in zip(av, bv)]
            denominator = moments(bv)['mean']
            first.update(left=moments(av), right=moments(bv), paired=moments(differences), deltas=differences,
                signs={'+': sum(x > 0 for x in differences), '-': sum(x < 0 for x in differences), '0': sum(x == 0 for x in differences)},
                relative_difference_of_means_percent=100*(moments(av)['mean']-denominator)/denominator if denominator else None)
        result['first27_subsets'].append(first)
    return result


def independent_summary(summary, records, c, report):
    require(len(summary['rows']) == len(c.tasks()) and summary['seeds'] == list(c.SEEDS) and
            summary['primary_contrast'] == 'absolute_phase - relative_phase' and summary['blocking_reason'] is None,
            'Summary scope')
    for row, task in zip(summary['rows'], c.tasks()):
        r = records[task['variant'], task['seed']]
        expected = dict(**task, status='PASS', validation_error=None, parameters=c.COUNTS[task['variant']],
                        scientific_fit_started=True, actual_epochs=r['actual_epochs'])
        expected.update({k: r[k] for k in ('best_valid_metrics', 'best_epoch', 'first27_complete', 'first27_best_ndcg10',
            'train_seconds', 'valid_seconds', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes', 'checkpoint_sha256', 'best_diagnostics')})
        require(row == expected, 'Independent summary row ' + task['run_id'])
    expected = []
    for seed in c.SEEDS:
        for left, right in CONTRASTS:
            a, b = records[left, seed], records[right, seed]
            x, y = a['best_valid_metrics']['ndcg@10'], b['best_valid_metrics']['ndcg@10']
            expected.append(dict(seed=seed, comparison=left+' - '+right, status='COMPLETE', delta=x-y,
                relative_percent=100*(x-y)/y if y else None,
                first27_delta=a['first27_best_ndcg10']-b['first27_best_ndcg10'] if a['first27_complete'] and b['first27_complete'] else None))
    require(summary['contrasts'] == expected, 'Independent summary contrasts')
    require(summary['scientific_fits_expected'] == summary['scientific_fits_started'] ==
            summary['scientific_fits_completed'] == len(c.tasks()) and summary['unknown_scientific_starts'] == 0,
            'Independent summary counters')
    replays = [report.replay_check(records['baseline_dual', seed]) for seed in c.SEEDS]
    require(all(r['status'] == 'PASS' for r in replays) and summary['fresh_control_replays'] == replays,
            'Fresh historical control exact checkpoint/science replay')
    return replays


def source_manifest_name(phase, attempt):
    require(phase in ('pilot', 'confirmation') and attempt in ('001', '002'), 'Manifest phase/attempt')
    return 'source_manifest'+('_confirmation' if phase == 'confirmation' else '')+('_002' if attempt == '002' else '')+'.json'


def confirmation_lineage(decision, manifest, execution, plan_sha, load, manifest_name='source_manifest_confirmation.json'):
    """Independent byte-only verification of the four approved fixture paths.

    ``load`` resolves a repository-relative preserved dependency and returns
    its parsed value and actual SHA256. No current-source globals are mutated.
    """
    if decision['source_hash'] == manifest['source_hash']:
        require(not any(k in decision for k in ('confirmation_source_hash', 'confirmation_execution_commit',
                    'source_lineage_path', 'source_lineage_sha256')), 'Ambiguous same-source confirmation lineage')
        return None
    require(decision.get('confirmation_source_hash') == manifest['source_hash'] and
            decision.get('confirmation_execution_commit') == execution and re.fullmatch('[0-9a-f]{40}', execution),
            'Confirmation source/execution lineage target')
    prefix = 'experiments/mamba3_absolute_phase/'
    require(decision['source_lineage_path'] == prefix+'runtime/confirmation_source_lineage.json', 'Canonical source lineage')
    lineage, lineage_sha = load(decision['source_lineage_path'])
    require(lineage_sha == decision['source_lineage_sha256'], 'Confirmation lineage bytes')
    expected = dict(schema='absolute_phase_test_scope_lineage_v1', status='PASS', study_id='mamba3_absolute_phase_001',
        reason='confirmation_test_fixture_scope', pilot_execution_commit=decision['execution_commit'],
        confirmation_execution_commit=execution, pilot_source_hash=decision['source_hash'],
        confirmation_source_hash=manifest['source_hash'], plan_sha256=plan_sha)
    require(all(lineage.get(k) == v for k, v in expected.items()) and decision['plan_sha256'] == plan_sha,
            'Explicit unchanged-plan fixture lineage')
    require(lineage['pilot_manifest_path'] == prefix+'source_manifest.json' and
            manifest_name in ('source_manifest_confirmation.json', 'source_manifest_confirmation_002.json') and
            lineage['confirmation_manifest_path'] == prefix+manifest_name,
            'Original pilot and separate confirmation manifest paths')
    old, old_sha = load(lineage['pilot_manifest_path'])
    new, new_sha = load(lineage['confirmation_manifest_path'])
    require(old_sha == lineage['pilot_manifest_sha256'] and new_sha == lineage['confirmation_manifest_sha256'] and
            new == manifest and old['source_hash'] == SOURCE and decision['execution_commit'] == EXECUTION,
            'Original pilot and current confirmation manifest bytes')
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(old['source_hash'] == decision['source_hash'] == digest(old['files']) and
            new['source_hash'] == digest(new['files']) and set(old['files']) == set(new['files']) and len(old['files']) == 447,
            'Same 447 frozen source paths')
    require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for row in (old, new) for value in row['files'].values()) and
            all(old.get(k) == new.get(k) for k in set(old)|set(new) if k not in ('files', 'source_hash')),
            'Frozen manifest metadata and file hash formats')
    require(old['files'][prefix+'study_plan.json'] == new['files'][prefix+'study_plan.json'] == plan_sha,
            'Study plan belongs to both unchanged source closures')
    changes = [dict(path=name, before_sha256=old['files'][name], after_sha256=new['files'][name])
               for name in sorted(old['files']) if old['files'][name] != new['files'][name]]
    require([x['path'] for x in changes] == list(LINEAGE_PATHS) and lineage['changes'] == changes,
            'Only the four reviewed fixture/config/provenance paths changed')
    unchanged = {name: value for name, value in old['files'].items() if name not in LINEAGE_PATHS}
    require(lineage['unchanged_file_count'] == len(unchanged) == 443 and
            lineage['unchanged_files_sha256'] == digest(unchanged), 'All 443 remaining source bytes unchanged')
    failure, failure_sha = load(lineage['failure_evidence_path'])
    require(Path(lineage['failure_evidence_path']).parent == Path(prefix+'evidence/confirmation_cpu_scope_failure'),
            'Canonical preserved CPU failure scope')
    require(failure_sha == lineage['failure_evidence_sha256'] and failure['status'] == 'FAIL' and
            failure['study_phase'] == 'confirmation' and failure['execution_commit'] == EXECUTION and
            failure['source_hash'] == SOURCE and failure['scientific_fits'] == 0 and
            failure['cpu_tests']['run'] == 100 and failure['cpu_tests']['failures'] == 29 and
            failure['cpu_tests']['errors'] == 2 and failure['cpu_tests']['skipped'] == 0 and
            failure['TEST'] == 'NOT_RUN' and failure['test_evaluation_count'] == failure['mimo_model_forward_calls'] == 0 and
            all(type(failure[k]) is int for k in ('scientific_fits', 'test_evaluation_count', 'mimo_model_forward_calls')) and
            all(type(failure['cpu_tests'][k]) is int for k in ('run', 'failures', 'errors', 'skipped')),
            'Preserved original confirmation CPU fixture failure')
    require(lineage['review_path'] == prefix+'runtime/confirmation_transition_review.json', 'Canonical transition review')
    review, review_sha = load(lineage['review_path'])
    review_expected = {k: expected[k] for k in ('status', 'study_id', 'pilot_execution_commit', 'pilot_source_hash',
                        'confirmation_execution_commit', 'confirmation_source_hash', 'plan_sha256')}
    review_expected['changed_paths'] = list(LINEAGE_PATHS)
    require(review_sha == lineage['review_sha256'] and all(review.get(k) == v for k, v in review_expected.items()),
            'Preserved transition review identity and exact changed paths')
    return dict(source_lineage_sha256=lineage_sha, pilot_source_hash=old['source_hash'],
        confirmation_source_hash=new['source_hash'], pilot_manifest_sha256=old_sha,
        confirmation_manifest_sha256=new_sha, unchanged_file_count=len(unchanged),
        changed_paths=[x['path'] for x in changes], failure_evidence_sha256=failure_sha, review_sha256=review_sha)


def confirmation_pilot(files, base, c, read, sha, report, manifest):
    decision_path = files/'runtime/confirmation_decision.json'
    decision = read(decision_path)
    require(decision['status'] == 'AUTHORIZED_BY_FROZEN_RULE' and
            decision['plan_sha256'] == base['plan_sha256'] and decision['pairing_verified'] is True, 'Frozen confirmation decision')
    def preserved(path):
        relative = Path(path)
        prefix = Path('experiments/mamba3_absolute_phase')
        require(not relative.is_absolute() and '..' not in relative.parts and relative.is_relative_to(prefix), 'Safe lineage dependency')
        result = files/relative.relative_to(prefix)
        require(result.resolve().is_relative_to(files.resolve()) and result.is_file() and not result.is_symlink(), 'Preserved lineage dependency')
        return read(result), sha(result)
    lineage = confirmation_lineage(decision, manifest, base['execution_commit'], base['plan_sha256'], preserved, c.MANIFEST.name)
    def local(path):
        relative = Path(path)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Safe linked pilot artifact')
        result = c.ROOT/relative
        require(result.resolve().is_relative_to(c.ROOT.resolve()) and result.is_file() and not result.is_symlink(), 'Linked pilot bytes')
        return result
    audit_path = local(decision['audit_path']); audit = read(audit_path)
    require(sha(audit_path) == decision['audit_sha256'], 'Linked pilot audit SHA')
    expected = dict(status='PASS', study_id=c.STUDY, study_phase='pilot', execution_commit=decision['execution_commit'],
        source_hash=decision['source_hash'], plan_sha256=base['plan_sha256'], scientific_fits_started=3,
        scientific_fits_completed=3, unknown_scientific_starts=0, TEST='NOT_RUN', test_evaluation_count=0, pairing_verified=True)
    require(all(audit.get(k) == v for k, v in expected.items()), 'Linked complete independent pilot audit')
    records, scores = {}, {}
    require(set(decision['pilot_results']) == set(MODES), 'Exactly three decision pilot references')
    for mode in MODES:
        entry = decision['pilot_results'][mode]; path = local(entry['path']); r = read(path)
        require(sha(path) == entry['sha256'] == audit['pilot_result_sha256'][mode] and r['status'] == 'PASS' and
                r['seed'] == 2026 and r['phase_mode'] == mode and all(r[k] == expected[k] for k in
                ('study_id', 'study_phase', 'execution_commit', 'source_hash', 'plan_sha256', 'TEST', 'test_evaluation_count')), 'Pilot raw binding')
        if lineage is not None:
            require(r['source_manifest_sha256'] == lineage['pilot_manifest_sha256'], 'Raw pilot original source manifest SHA')
        report.validate_record(r, mode, 2026)
        records[mode, 2026] = r; scores[mode] = r['best_valid_metrics']['ndcg@10']
    require(scores == decision['pilot_scores'] and scores['absolute_phase'] >= max(scores['baseline_dual'], scores['relative_phase']),
            'Frozen conditional metric rule')
    require(all(records[v, 2026][k] == records['baseline_dual', 2026][k] for v in MODES[1:] for k in PAIRING), 'Linked pilot actual pairing')
    return sha(decision_path), records, lineage


def audit(folder, execution=EXECUTION):
    folder = Path(folder).resolve()
    saved = json.loads((folder/'preservation_manifest.json').read_text())
    phase, attempt, job = saved['study_phase'], saved['execution_attempt'], saved['job_id']
    require(phase in ('pilot', 'confirmation') and attempt in ('001', '002') and re.fullmatch('[0-9]+', job), 'Preserved phase/attempt/job')
    os.environ.setdefault('ABS_PHASE_STAGE', phase); os.environ.setdefault('ABS_PHASE_ATTEMPT', attempt)
    from experiments.mamba3_absolute_phase import config as c, provenance as p, report
    from experiments.mamba3_mimo_time.records import read, sha, create, now, digest, accepted_cases
    require(c.STAGE == phase and c.EXECUTION_ATTEMPT == attempt, 'Fresh-process phase/attempt configuration')
    require('torch' not in sys.modules, 'Audit must run without Torch')
    require(re.fullmatch('[0-9a-f]{40}', execution) and saved['execution_commit'] == execution, 'Exact execution identity')
    manifest = p.verify(); plan = c.plan(); files = folder/'files'
    require(saved['source_hash'] == manifest['source_hash'], 'Preserved source identity')
    if execution == EXECUTION:
        require(manifest['source_hash'] == SOURCE and len(manifest['files']) == 447, 'Initial frozen source')
    require(saved['checkpoint_loading'] is False and saved['weights_copied'] is False, 'Preservation no-weight scope')
    preserved = {}
    for row in saved['files']:
        relative = Path(row['path']); path = files/relative
        require(not relative.is_absolute() and '..' not in relative.parts and path.resolve().is_relative_to(files.resolve()) and
                not path.is_symlink() and path.is_file() and row['path'] not in preserved, 'Safe unique preserved path')
        require(row['cluster_path'] == str(REMOTE/'experiments/mamba3_absolute_phase'/relative) and
                path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Preserved bytes ' + str(relative))
        preserved[row['path']] = row
    require(set(preserved) == {str(x.relative_to(files)) for x in files.rglob('*') if x.is_file()}, 'Exact preserved inventory')
    for name, expected in manifest['files'].items():
        blob = subprocess.check_output(['git', 'show', execution+':'+name], cwd=c.ROOT)
        require(hashlib.sha256(blob).hexdigest() == expected, 'Published execution blob ' + name)
    for name in (c.MANIFEST.name, 'study_plan.json', 'DESIGN.md', 'NEW_PLAN.md'):
        require(sha(files/name) == sha(c.HERE/name), 'Frozen document ' + name)
    scheduler = read(folder/'scheduler_terminal.json'); raw = scheduler['sacct_raw'].splitlines()
    parsed = [dict(zip(raw[0].split('|'), line.split('|'))) for line in raw[1:] if line]
    require(parsed == scheduler['steps'] and [x for x in parsed if x['JobIDRaw'] == job] == [scheduler['job']], 'Saved scheduler raw binding')
    require(parsed and len({x['JobIDRaw'] for x in parsed}) == len(parsed) and all(
        x['JobIDRaw'] in (job, job+'.batch', job+'.extern') and x['State'] == 'COMPLETED' and x['ExitCode'] == '0:0' for x in parsed), 'Successful terminal allocation')
    logs, runs = files/'slurm_logs'/phase/('attempt_'+attempt), files/'runs'/phase/('attempt_'+attempt)
    login, reservation, submission = [read(logs/name) for name in ('login_verification.json', 'reservation.json', 'submission.json')]
    base = p.bindings(execution, manifest)
    p.validate_ownership(login, reservation, sha(logs/'login_verification.json'), base, job, submission)
    require(submission['status'] == 'SUBMITTED' and submission['job_id'] == job, 'Exact submitted owner')
    require(reservation['scientific_fits_before_submit'] == 0, 'Fresh allocation without prior scientific fits')
    budget = reservation['global_budget']
    integer(budget['previous_submissions'], 'Global previous submissions', maximum=2)
    integer(budget['requested_seconds_before'], 'Global previous allocation', maximum=64800)
    require(budget['requested_seconds_after'] == budget['requested_seconds_before']+reservation['requested_seconds'] <= 64800,
            'Global 18-hour allocation cap')
    require(submission['jobs_submitted'] == budget['previous_submissions']+1 <= 3, 'Global submission cap')
    if attempt == '002':
        review_path = files/('runtime/retry_review_'+phase+'.json'); review = read(review_path)
        require(sha(review_path) == reservation['retry_review_sha256'] and review['status'] == 'APPROVED_PRE_FIT_TECHNICAL_RETRY' and
                review['scientific_fits_started'] == review['unknown_scientific_starts'] == 0 and
                all(review[k] is True for k in ('old_job_terminal', 'method_unchanged', 'tolerances_unchanged', 'regression_passed')),
                'Authorized pre-fit-only technical retry')
        # The actual review must name preserved terminal evidence; never infer
        # zero fits from the mere absence of result files.
        parent_path = c.ROOT/review['preservation_manifest_path']
        require(sha(parent_path) == review['preservation_manifest_sha256'], 'Retry parent terminal binding')
        parent = read(parent_path)
        require(parent['job_id'] == review['old_job_id'] != job and parent['study_phase'] == phase and
                parent['execution_attempt'] == '001' and not parent['checkpoints'], 'Retry parent identity/scope')
        for row in parent['files']:
            path = parent_path.parent/'files'/row['path']
            require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Retry parent immutable bytes')
        terminal_parent = read(parent_path.parent/'scheduler_terminal.json')['job']
        require(terminal_parent == review['scheduler'] and terminal_parent['JobIDRaw'] == review['old_job_id'] and terminal_parent['State'].split()[0].rstrip('+') in
                ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED', 'SPECIAL_EXIT', 'COMPLETED'),
                'Retry parent is terminal')
        parent_files = parent_path.parent/'files'
        inventory = {x['path'] for x in parent['files']}
        for task in c.tasks():
            result = f"runs/{phase}/attempt_001/{task['run_id']}.json"
            directory = f"slurm_logs/{phase}/attempt_001/{task['run_id']}/"
            owners = [name for name in inventory if name.startswith(directory) and
                      (name.endswith(('run.lock', 'progress.json')) or '/checkpoints/' in name)]
            if result not in inventory:
                require(not owners, 'Unknown prior scientific start blocks retry')
                continue
            r = read(parent_files/result)
            require(report.scientific_start(r) is False and not any('/checkpoints/' in name for name in owners) and
                    not (r.get('status') == 'NOT_RUN' and owners), 'Retry parent scientific fits zero')
            progress = directory+'progress.json'
            if progress in inventory:
                require(report.scientific_start(read(parent_files/progress)) is False, 'Retry parent progress must agree')
    pilot_records = {}
    lineage = None
    decision_sha = None
    if phase == 'confirmation':
        decision_sha, pilot_records, lineage = confirmation_pilot(files, base, c, read, sha, report, manifest)
    require(login.get('confirmation_decision_sha256') == reservation.get('confirmation_decision_sha256') == decision_sha,
            'Conditional authorization SHA chain')
    arithmetic = Arithmetic(); old = read(c.PILOT)
    coverage = read(logs/'train_coverage.json'); coverage_sha = sha(logs/'train_coverage.json')
    require(all(coverage.get(k) == v for k, v in base.items()) and coverage_sha == login['coverage_sha256'] == reservation['coverage_sha256'], 'Frozen TRAIN coverage chain')
    coverage_check(coverage, arithmetic, digest)
    prior_folder = c.ROOT/'experiments/mamba3_time_memory/evidence/job4373393'
    prior_inventory = read(prior_folder/'preservation_manifest.json')
    prior_relative = 'slurm_logs/attempt_002/train_coverage.json'
    prior_entry = [x for x in prior_inventory['files'] if x['path'] == prior_relative]
    prior_path = prior_folder/'files'/prior_relative
    require(len(prior_entry) == 1 and sha(prior_path) == prior_entry[0]['sha256'] and
            prior_path.stat().st_size == prior_entry[0]['bytes'], 'Previous fixed TRAIN coverage bytes')
    prior_coverage = read(prior_path)
    require(coverage['selected_train_indices'] == prior_coverage['selected_train_indices'] and
            coverage['selected_input_hashes'] == prior_coverage['selected_input_hashes'], 'Exact TRAIN sample agrees with published memory study')
    require(coverage['protocol'] == old['protocol'] and coverage['manifest_sha256'] == old['manifest_sha256'] and
            coverage['frozen_train_time_stats_sha256'] == old['train_time_stats_sha256'], 'TRAIN data identity')
    cfg = old['effective_config']
    require(coverage['input_field_names_read'] == [cfg['ITEM_ID_FIELD']+cfg['LIST_SUFFIX'], cfg['TIME_FIELD']+cfg['LIST_SUFFIX'], cfg['ITEM_LIST_LENGTH_FIELD']], 'Coverage input fields only')
    runtime_identity(login, old['runtime'], manifest, 'Login', False)
    base.update(job_id=job, reservation_token=reservation['token'], reservation_sha256=sha(logs/'reservation.json'),
        login_verification_sha256=sha(logs/'login_verification.json'), coverage_sha256=coverage_sha, confirmation_decision_sha256=decision_sha)
    def owner(record, label, status=True):
        require(all(record.get(k) == v for k, v in base.items()), label + '.owner')
        if status:
            require(record.get('status') == 'PASS', label + '.status')
    cpu_rows = []
    for prefix in ('cpu_preflight_', 'no_git_preflight_'):
        name = prefix+execution+'.json'; cpu = read(logs/name); tests = cpu['cpu_tests']
        require(cpu['status'] == 'PASS' and cpu['execution_commit'] == execution and cpu['source_hash'] == manifest['source_hash'] and
                cpu['execution_attempt'] == attempt and cpu['study_phase'] == phase and cpu['cuda_initialized'] is False and cpu['cuda_mask'] == '', 'CPU provenance')
        require(tests['run'] == plan['cpu_test_count'] and not any(tests[k] for k in ('failures', 'errors', 'skipped')), 'Frozen CPU count')
        observed_tests = re.findall(r'^(test_\w+) \(([^)]+)\) \.\.\. ok$', tests['log'], re.M)
        require(sorted(cls+'.'+name for name, cls in observed_tests) == sorted(plan['cpu_test_ids']), 'Frozen CPU test identities')
        require(cpu['parameter_counts'] == c.COUNTS and cpu['scientific_fits'] == cpu['test_evaluation_count'] == cpu['mimo_model_forward_calls'] == 0 and
                cpu['TEST'] == 'NOT_RUN' and cpu['gradcheck'] == 'PASS', 'CPU scope')
        runtime_identity(cpu, old['runtime'], manifest, name, False)
        cpu_rows.append(dict(file=name, tests=tests['run'], status='PASS'))
    inherited = read(runs/'inherited_kernel.json'); owner(inherited, 'Inherited')
    require(inherited['inherited'] == p.inherited(), 'Inherited unchanged admission')
    runtime_identity(inherited, old['runtime'], manifest, 'Inherited', True)
    gate = read(runs/'targeted_gate.json'); owner(gate, 'Gate'); runtime_identity(gate, old['runtime'], manifest, 'Gate', True)
    require(gate['required_cases'] == plan['required_cases'] and accepted_cases(gate['cases'], plan['required_cases']), 'Exact targeted cases/leaves')
    checks = sum(len(x['checks']) for x in gate['cases'])
    require(len(gate['cases']) == 17 and checks == 532 and gate['scientific_fits'] == 0, 'GPU gate scope')
    for case in gate['cases']:
        require(not any(case.get(k) for k in ('missing_keys', 'unexpected_keys', 'failed_keys', 'traceback')), 'Final GPU case errors')
    numeric_gate = gate_numerics(gate, arithmetic, read(c.POLICY))
    smoke = read(runs/'smoke.json'); owner(smoke, 'Smoke'); p.validate_smoke(smoke)
    runtime_identity(smoke, old['runtime'], manifest, 'Smoke', True)
    require(smoke['targeted_gate_sha256'] == sha(runs/'targeted_gate.json') and smoke['scientific_fits'] == 0 and smoke['fixture_seed'] == 314159, 'Smoke fixture/gate')
    for row in smoke['rows']:
        require(row['peak_reserved_bytes'] >= row['peak_allocated_bytes'], 'Smoke memory ordering')
        for i, step in enumerate(row['steps']):
            if row['phase_mode'] == 'baseline_dual':
                require(step['W_before_l2'] is None, 'Smoke baseline no W')
            else:
                require(step['W_before_l2'] == (row['steps'][i-1]['W_after_l2'] if i else 0), 'Smoke W continuity')
    pipeline = read(logs/'pipeline_status.json'); terminal = read(runs/'terminal_metadata.json')
    total = len(c.tasks())
    for label, record in [('Pipeline', pipeline), ('Terminal', terminal)]:
        owner(record, label)
        require((record['scientific_fits_started'], record['scientific_fits_completed'], record['unknown_scientific_starts']) == (total, total, 0)
                and record['summary_status'] == 'PASS' and not any(k in record for k in ('error', 'traceback', 'report_traceback')), label + '.complete_scope')
    stages = [('gate', None, None, 'PASS'), ('smoke', None, None, 'PASS')]+[(x['variant'], x['seed'], x['run_id'], 'PASS') for x in c.tasks()]
    require([(x['stage'], x['seed'], x['run_id'], x['status']) for x in pipeline['stages']] == stages, 'Ordered owned stages')
    expected_environments = {str((logs/stage/'child_environment.json').relative_to(files)): stage for stage in ('gate', 'smoke')}
    expected_environments.update({str((c.paths(t['variant'], t['seed'])['runtime']/'process/child_environment.json').relative_to(c.HERE)):
                                  t['variant'] for t in c.tasks()})
    saved_environments = {str(path.relative_to(files)): read(path) for path in logs.rglob('child_environment.json')}
    cuda_environment = child_environment_check(pipeline['allocation_cuda_visibility'], saved_environments, expected_environments)
    previous = datetime.fromisoformat(pipeline['started_at'])
    for stage in pipeline['stages']:
        start, end = (datetime.fromisoformat(stage[k]) for k in ('started_at', 'finished_at'))
        require(previous <= start <= end, 'Sequential stage timing'); previous = end
    require(previous <= datetime.fromisoformat(pipeline['finished_at']), 'Pipeline finish ordering')
    require(terminal['pipeline_sha256'] == sha(logs/'pipeline_status.json') and terminal['checkpoint_loading'] is False and
            {k:v for k,v in pipeline.items() if k != 'stages'} == {k:v for k,v in terminal.items() if k not in ('pipeline_sha256', 'checkpoint_loading', 'checkpoints')},
            'Terminal exact pipeline binding')
    require(set(terminal['checkpoints']) == {x['run_id'] for x in c.tasks()} and set(saved['checkpoints']) ==
            {str(c.paths(x['variant'], x['seed'])['checkpoint'].relative_to(c.HERE)) for x in c.tasks()}, 'Exactly planned checkpoints')
    owner(read(logs/'pipeline.lock'), 'Pipeline lock', False)
    require({x.name for x in runs.glob('mamba3_absolute_phase_*_seed*.json')} ==
            {c.paths(x['variant'], x['seed'])['result'].name for x in c.tasks()}, 'Exactly planned scientific results')
    records, rows, hashes = {}, [], {}
    zero_hash = hashlib.sha256(json.dumps(['phase_adapter.W', 'torch.float32', [32, 4]]).encode()+bytes(512)).hexdigest()
    for task in c.tasks():
        mode, seed = task['variant'], task['seed']; paths = c.paths(mode, seed)
        result_path = files/paths['result'].relative_to(c.HERE); r = read(result_path); owner(r, task['run_id'])
        require(finite(r), task['run_id']+'.finite_record')
        runtime_identity(r, old['runtime'], manifest, task['run_id'], True)
        require(r['runtime']['gpu'] == inherited['runtime']['gpu'] == gate['runtime']['gpu'] == smoke['runtime']['gpu'], 'Allocation GPU identity')
        lock = read(files/paths['lock'].relative_to(c.HERE)); owner(lock, task['run_id']+'.lock', False)
        require(all(lock[k] == r[k] for k in ('run_id', 'phase_mode', 'seed', 'targeted_gate_sha256', 'smoke_sha256')) and
                lock['scientific_fit_started'] is False and lock['status'] == 'RUNNING', 'Fresh immutable fit lock')
        require(r['targeted_gate_sha256'] == sha(runs/'targeted_gate.json') and r['smoke_sha256'] == sha(runs/'smoke.json'), 'Fit admission SHA')
        progress = read(files/paths['runtime'].relative_to(c.HERE)/'progress.json')
        require(all(progress[k] == r.get(k) for k in ('execution_attempt', 'study_phase', 'study_id', 'execution_commit', 'source_hash',
                'job_id', 'run_id', 'phase_mode', 'seed', 'status', 'stage', 'scientific_fit_started', 'actual_epochs', 'started_at', 'finished_at', 'error'))
                and progress['last_epoch'] == r['actual_epochs']-1, 'Final fit progress binding')
        if mode != 'baseline_dual':
            require(r['initial_phase_parameters']['phase_adapter.W']['sha256'] == zero_hash, 'Exact zero FP32 W tensor hash')
        row = audit_run(r, mode, seed, files, saved, terminal, arithmetic, c, read, report, coverage)
        rows.append(row); records[mode, seed] = r; hashes[r['run_id']] = sha(result_path)
    for seed in c.SEEDS:
        require(set(records['baseline_dual', seed]['rng_components']) == {'python', 'numpy', 'cpu', 'cuda', 'aggregate', 'loader_generator'}, 'Full paired RNG components')
        for mode in MODES[1:]:
            require(all(records[mode, seed][k] == records['baseline_dual', seed][k] for k in PAIRING), 'Within-seed actual pairing')
    summary = read(runs/(phase+'_summary.json')); owner(summary, 'Summary')
    replays = independent_summary(summary, records, c, report)
    errors, warnings = [], []
    for path in logs.rglob('*'):
        if path.suffix not in ('.log', '.out', '.err'):
            continue
        for line in path.read_text().splitlines():
            if re.search(r'Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:|\b(?:nan|inf)\b', line, re.I):
                errors.append(dict(path=str(path.relative_to(files)), line=line))
            if 'Warning:' in line:
                warnings.append(dict(path=str(path.relative_to(files)), line=line))
    require(not errors, 'Runtime errors: ' + repr(errors[:4]))
    groups = [aggregates(records, c.SEEDS, 'pilot descriptive only' if phase == 'pilot' else 'PRIMARY new four seeds')]
    if phase == 'confirmation':
        groups.append(aggregates({**pilot_records, **records}, (2026, 2027, 2028, 2029, 2030), 'EXPLORATORY all five including selection pilot'))
    require('torch' not in sys.modules, 'Audit imported Torch')
    value = dict(status='PASS', study_id=c.STUDY, study_phase=phase, job_id=job, execution_attempt=attempt,
        execution_commit=execution, source_hash=manifest['source_hash'], plan_sha256=base['plan_sha256'],
        source_blobs=len(manifest['files']), audited_at=now(), preserved_files=len(preserved),
        preserved_bytes=sum(x['bytes'] for x in preserved.values()), scheduler=scheduler['job'], cpu=cpu_rows,
        gpu_cases=17, gpu_checks=checks, numeric_gate=numeric_gate, smoke_steps_per_variant=3,
        cuda_environment=cuda_environment,
        scientific_fits_started=total, scientific_fits_completed=total, unknown_scientific_starts=0,
        complete_triples_verified=len(c.SEEDS), pairing_verified=True, paired_fields=list(PAIRING),
        TEST='NOT_RUN', test_evaluation_count=0, rows=rows, epochs=sum(x['epochs'] for x in rows),
        metric_cells=sum(x['metric_cells'] for x in rows), contrasts=summary['contrasts'], aggregates=groups,
        fresh_control_replays=replays, coverage_sha256=coverage_sha, coverage_summary=coverage['summary'],
        runtime_errors=errors, warnings=warnings, diagnostic_arithmetic=arithmetic.result(),
        checkpoint_loading=False, model_forward_count=0, torch_imported=False,
        runtime_identity=dict(dependencies_match_historical=True, imported_paths_in_frozen_manifest=True,
            gpu=inherited['runtime']['gpu'], historical_gpu_label_not_required=True),
        result_sha256=hashes, summary_sha256=sha(runs/(phase+'_summary.json')),
        preservation_manifest_sha256=sha(folder/'preservation_manifest.json'), scheduler_sha256=sha(folder/'scheduler_terminal.json'),
        confirmation_decision_sha256=decision_sha,
        limitations=[
            'Checkpoint SHA and byte counts were streamed during preservation and bound to selected-epoch metadata; no weights copied, loaded or locally rehashed.',
            'TRAIN raw histories are not preserved: exact indices, input hash binding, coverage counts and algebra checked; timestamp multiplicities and span quantiles cannot be independently reconstructed.',
            'VALID aggregates cover the last active query per history separately for each layer, not every event. Eight saved examples cannot reconstruct all-query sums or extrema.',
            'W and fixed analytic grid values are preserved and checked; native fixture samples cover first four angles. All-angle fixture aggregates are independently bounded from saved corrections and fixed content/DT, accounting for FP32 libm/reduction differences.',
            'Gradient norms are preserved summaries, not raw gradients; all TRAIN backward hook counts and finite flags checked, no backward replay.',
            'Native MIMO raw angles stay FP32. BF16 diagnostic fields are hypothetical raw-angle casts, not measurements of actual downstream Q/K rotation quantization.',
            'Native local increment formulas and saved shift tensors checked; accumulated theta and recurrence are not regenerated. GPU reference acceptance uses the unchanged frozen policy.',
            'RecBole log timers and saved wrapper timers have different envelopes; both log copies agree and saved sums/memory maxima are checked.',
            'No-Git CPU proof is evidence from a separate invocation; its CPU JSON does not itself record the deny-git environment.',
            'A conditional confirmation uses new four seeds as primary; all-five aggregates include the selection pilot and are explicitly exploratory.'
        ])
    if phase == 'pilot':
        value['pilot_result_sha256'] = {mode: hashes[c.paths(mode, 2026)['run_id']] for mode in MODES}
    else:
        value['confirmation_lineage'] = lineage
    output = folder/'independent_audit.json'
    if output.exists():
        prior = read(output)
        require({k:v for k,v in prior.items() if k != 'audited_at'} == {k:v for k,v in value.items() if k != 'audited_at'},
                'Existing audit differs; refusing to replace evidence')
        return prior
    create(output, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--execution', default=EXECUTION, help='Exact expected published execution SHA')
    args = parser.parse_args()
    result = audit(args.folder, args.execution)
    print(json.dumps({k:result[k] for k in ('status', 'study_phase', 'job_id', 'epochs', 'metric_cells',
        'scientific_fits_started', 'scientific_fits_completed', 'pairing_verified', 'rows')}, indent=2))
