"""Audit preserved terminal bytes with the standard library, never model code.

Run from the repository root with ``python -B -m
experiments.mamba3_time_memory.runtime.audit_saved <evidence/job...>``.
Only independent_audit.json is created; existing evidence is never replaced.
The eight published examples are not the complete 256-query reservoir and
cannot reconstruct its quantiles or all-query sums. See limitations in output.
"""
import argparse
import hashlib
import heapq
import json
import math
import os
import re
import struct
import subprocess
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

EXECUTION = '55d812bf55b1dffbbab6a7b0da86e616a6227b8b'
SOURCE = '4e26434f8bf084d6acd562e6859c24699013510d8f2ffcaaf2653435d3698e8d'
ANCHORS = (1, 4, 16, 32)
REFERENCE = 838393
MODES = ('no_memory', 'index_memory', 'time_memory')
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


class Arithmetic:
    """Diagnostic-only reductions; scientific metric comparisons stay exact."""
    def __init__(self):
        self.differences = []

    def close(self, actual, expected, label):
        number(actual, label); number(expected, label)
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), label)
        if actual != expected:
            self.differences.append(dict(field=label, saved=actual, derived=expected,
                                         absolute_difference=abs(actual - expected)))

    def gate(self, beta, lam, label):
        number(beta, label + '.beta'); number(lam, label + '.lambda')
        f32 = lambda x: struct.unpack('!f', struct.pack('!f', x))[0]
        require(beta == f32(beta) and lam == f32(lam) and -1 <= lam <= 1,
                label + '.fp32_signed_gate')
        expected = f32(math.tanh(beta))
        # CPU libm and CUDA fp32 tanh are separate implementations. This is a
        # saved diagnostic relation, not an admission or training tolerance.
        bits = lambda x: struct.unpack('!I', struct.pack('!f', abs(x)))[0]
        ulps = abs(bits(lam) - bits(expected))
        require((lam == 0 or expected == 0 or (lam > 0) == (expected > 0))
                and ulps <= 2, label + '.tanh_beta')
        if lam != expected:
            self.differences.append(dict(field=label + '.tanh_beta', saved=lam,
                                         derived=expected, fp32_ulps=ulps))

    def weight_sum(self, actual, nonempty, label):
        number(actual, label)
        # At most four nonnegative fp32 softmax weights per query. Their
        # saved float64 sum need not be the exact integer number of queries.
        bound = nonempty * 4 * 2**-23
        require(abs(actual - nonempty) <= bound, label)
        if actual != nonempty:
            self.differences.append(dict(field=label, saved=actual, derived=nonempty,
                                         absolute_difference=abs(actual - nonempty),
                                         fp32_softmax_absolute_bound=bound))

    def result(self):
        return dict(relative_tolerance=1e-12, absolute_tolerance=1e-15,
                    tanh_fp32_max_ulps=2, softmax_sum_bound_per_query=4 * 2**-23,
                    count=len(self.differences),
                    differences=self.differences,
                    scope='Saved diagnostic algebra only; no metric, replay, SHA or GPU admission tolerance')


def stat(row, label, count=None, sample_count=None, lower=0, upper=None):
    """Check saved scalar summaries, without claiming to reconstruct tensors."""
    n = integer(row['count'], label + '.count')
    if count is not None:
        require(n == count, label + '.expected_count')
    sampled = integer(row['reservoir_value_count'], label + '.reservoir_count', maximum=n)
    if sample_count is not None:
        require(sampled == sample_count, label + '.expected_reservoir_count')
    total = number(row['sum'], label + '.sum')
    if not n:
        require(total == 0 and all(row[k] is None for k in ('mean', 'min', 'max')),
                label + '.empty')
    else:
        low, high = number(row['min'], label), number(row['max'], label)
        require(lower <= low <= high and (upper is None or high <= upper), label + '.bounds')
        require(row['mean'] == total / n, label + '.mean')
        require(low - 1e-12 * max(1, abs(low)) <= row['mean'] <= high + 1e-12 * max(1, abs(high)),
                label + '.mean_range')
    q = row['approximate_quantiles']
    if not sampled:
        require(q is None, label + '.empty_quantiles')
    else:
        require(isinstance(q, dict) and set(q) == set(QUANTILES), label + '.quantile_keys')
        values = [number(q[k], label + '.' + k) for k in QUANTILES]
        require(values == sorted(values) and row['min'] <= values[0] <= values[-1] <= row['max'],
                label + '.quantile_order_range')


def sample_in_stat(value, row, label):
    number(value, label)
    require(row['count'] > 0 and row['min'] <= value <= row['max'], label + '.sample_range')


@lru_cache(maxsize=4)
def reservoir_ordinals(examples):
    def key(i):
        return hashlib.sha256(('mamba3-time-memory-query-v1:' + str(i)).encode()).digest(), i
    return heapq.nsmallest(min(8, examples), range(examples), key=key)


def index_addresses(query):
    available = list(range(query))
    chosen, outside = [], []
    for anchor in ANCHORS:
        if not available:
            outside.append(False)
            continue
        ages = [query - j for j in available]
        outside.append(anchor < min(ages) or anchor > max(ages))
        j = min(available, key=lambda j: (abs(math.log1p(query - j) - math.log1p(anchor)), -j))
        chosen.append(j); available.remove(j)
    return sorted(chosen), outside


def address_diagnostics(d, label, arithmetic, expected_examples, coverage_indices=None):
    require(finite(d), label + '.finite')
    n = integer(d['examples'], label + '.examples', 1)
    require(n == expected_examples, label + '.population')
    expected_scope = ('TRAIN input histories only, CPU metadata-only audit' if coverage_indices is not None
                      else 'existing VALID forwards; no additional dataset/model pass')
    require(d['scope'] == expected_scope and d['query_scope'] == 'last valid query of each input example only',
            label + '.scope')
    reservoir = d['reservoir']
    retained = min(256, n)
    require(reservoir == dict(method='lowest SHA256 priorities of fixed stream ordinals',
                             capacity=256, retained=retained, training_rng_consumed=False,
                             approximate_quantiles='computed from retained queries; not all-query exact quantiles'),
            label + '.reservoir_definition')
    hist = d['history_length_histogram']
    require(hist and all(str(int(k)) == k and 1 <= int(k) <= 50 for k in hist), label + '.history_keys')
    for k, v in hist.items():
        integer(v, label + '.history_count.' + k, 1)
    require(sum(hist.values()) == n and d['nonempty_histories'] == n, label + '.histories')
    slots = {str(k): sum(v for h, v in hist.items() if min(4, int(h) - 1) == k) for k in range(5)}
    require(d['valid_slot_count_histogram'] == slots, label + '.slot_histogram')
    total_slots = sum(int(k) * v for k, v in slots.items())
    for count, fraction, expected in (
            ('at_least_four_past_events_count', 'at_least_four_past_events_fraction', slots['4']),
            ('empty_memory_count', 'empty_memory_fraction', slots['0'])):
        require(d[count] == expected and d[fraction] == expected / n, label + '.' + count)
    same = integer(d['selected_sets_equal_count'], label + '.same_sets', maximum=n)
    require(d['selected_sets_equal_fraction'] == same / n, label + '.same_fraction')
    require(len(d['time_anchor_reach']) == 4, label + '.anchor_reach_count')
    previous = n
    for anchor, row in zip(ANCHORS, d['time_anchor_reach']):
        count = integer(row['windows_reaching'], label + '.reach_count', maximum=previous)
        require(row['anchor'] == anchor and row['fraction'] == count / n, label + '.reach_fraction')
        previous = count
    for name in ('history_length', 'temporal_span_ms', 'temporal_span_R0', 'selected_set_jaccard'):
        stat(d[name], label + '.' + name, n, retained,
             upper=50 if name == 'history_length' else 1 if name == 'selected_set_jaccard' else None)
    h = d['history_length']
    require(h['sum'] == sum(int(k) * v for k, v in hist.items()) and
            h['min'] == min(map(int, hist)) and h['max'] == max(map(int, hist)), label + '.history_stats')
    for key in ('sum', 'mean', 'min', 'max'):
        arithmetic.close(d['temporal_span_ms'][key] / REFERENCE, d['temporal_span_R0'][key],
                         label + '.span_units.' + key)
    sampled_slot_counts = []
    for mode in MODES[1:]:
        row = d[mode]
        for key in ('selected_ages_R0', 'selected_ages_ms', 'selected_event_lags'):
            stat(row[key], label + '.' + mode + '.' + key, total_slots,
                 lower=1 if key == 'selected_event_lags' else 0,
                 upper=49 if key == 'selected_event_lags' else None)
        sampled_counts = [row[key]['reservoir_value_count'] for key in
                          ('selected_ages_R0', 'selected_ages_ms', 'selected_event_lags')]
        require(len(set(sampled_counts)) == 1 and sampled_counts[0] <= retained * 4,
                label + '.selected_reservoir_counts')
        sampled_slot_counts.append(sampled_counts[0])
        for key in ('sum', 'mean', 'min', 'max'):
            if row['selected_ages_R0'][key] is not None:
                arithmetic.close(row['selected_ages_R0'][key] * REFERENCE, row['selected_ages_ms'][key],
                                 label + '.' + mode + '.age_units.' + key)
        outside = row['anchor_out_of_range']
        require(outside['definition'] == 'anchor outside min/max ages among remaining candidates at that greedy step'
                and len(outside['by_anchor']) == 4, label + '.outside_definition')
        for k, (anchor, entry) in enumerate(zip(ANCHORS, outside['by_anchor'])):
            eligible = sum(v for size, v in slots.items() if int(size) > k)
            count = integer(entry['count'], label + '.outside_count', maximum=eligible)
            require(entry['anchor'] == anchor and entry['eligible_count'] == eligible and
                    entry['fraction'] == (count / eligible if eligible else None), label + '.outside_anchor')
        count = sum(x['count'] for x in outside['by_anchor'])
        require(outside['eligible_anchor_count'] == total_slots and outside['count'] == count and
                outside['fraction'] == (count / total_slots if total_slots else None), label + '.outside_total')
    require(len(set(sampled_slot_counts)) == 1, label + '.paired_reservoir_count')
    samples = d['examples_reservoir']
    require([x['stream_ordinal'] for x in samples] == reservoir_ordinals(n), label + '.reservoir_members')
    for i, row in enumerate(samples):
        here = label + '.sample' + str(i)
        ordinal = row['stream_ordinal']
        require(row['input_index'] == (coverage_indices[ordinal] if coverage_indices is not None else ordinal),
                here + '.input_index')
        length = integer(row['history_length'], here + '.length', 1, 50)
        query = length - 1
        k = min(4, query)
        require(row['query_position'] == query and row['valid_slots'] == k, here + '.query_slots')
        require(row['temporal_span_R0'] == row['temporal_span_ms'] / REFERENCE, here + '.span')
        for key in ('history_length', 'temporal_span_ms', 'temporal_span_R0', 'selected_set_jaccard'):
            sample_in_stat(row[key], d[key], here + '.' + key)
        shared_ages = {}
        for mode in MODES[1:]:
            values = row[mode]
            indices = values['indices']
            require(len(indices) == k and indices == sorted(set(indices)) and
                    all(type(j) is int and 0 <= j < query for j in indices), here + '.' + mode + '.causal_indices')
            require(values['selected_event_lags'] == [query - j for j in indices], here + '.event_lags')
            require(values['anchor_valid'] == [j < k for j in range(4)] and
                    len(values['anchor_out_of_range']) == 4 and
                    all(type(v) is bool for v in values['anchor_out_of_range']) and
                    not any(values['anchor_out_of_range'][k:]), here + '.anchor_masks')
            ages = values['selected_ages_R0']
            require(len(ages) == len(values['selected_ages_ms']) == k and ages == sorted(ages, reverse=True),
                    here + '.age_order')
            for j, age, ms in zip(indices, ages, values['selected_ages_ms']):
                require(0 <= age <= row['temporal_span_R0'] and ms == age * REFERENCE, here + '.physical_age')
                if j in shared_ages:
                    require(shared_ages[j] == age, here + '.shared_age')
                shared_ages[j] = age
            for key in ('selected_event_lags', 'selected_ages_R0', 'selected_ages_ms'):
                for value in values[key]:
                    sample_in_stat(value, d[mode][key], here + '.' + mode + '.' + key)
            if mode == 'index_memory':
                expected, outside = index_addresses(query)
                require(indices == expected and values['anchor_out_of_range'] == outside, here + '.ordinal_selector')
        a, b = (set(row[mode]['indices']) for mode in MODES[1:])
        require(row['selected_sets_equal'] is (a == b) and
                row['selected_set_jaccard'] == (len(a & b) / len(a | b) if a or b else 1.), here + '.overlap')
    require(sum(x['valid_slots'] for x in samples) <= sampled_slot_counts[0], label + '.published_sample_subset')
    return dict(examples=n, slots=total_slots, retained=retained, retained_slots=sampled_slot_counts[0])


def memory_diagnostics(d, mode, label, arithmetic, eval_batch_size):
    require(d['status'] == 'MEASURED' and d['memory_mode'] == mode and
            d['no_persistent_representations'] is True and
            d['norm_ratio_definition'] == 'norm(readout) / norm(augmented output); zero denominator excluded' and
            d['interpretation'] == 'retrieval weights are not causal event importance', label + '.definitions')
    counts = address_diagnostics(d, label, arithmetic, 23951)
    arithmetic.gate(d['beta'], d['lambda'], label)
    require(d['negative_gate'] is (d['lambda'] < 0), label + '.negative_gate')
    zero = integer(d['zero_output_norm_count'], label + '.zero_norm', maximum=counts['examples'])
    shapes = d['reader_bank_shapes']
    require(shapes and shapes == [list(x) for x in sorted(set(tuple(x) for x in shapes))], label + '.unique_shapes')
    for shape in shapes:
        require(len(shape) == 4 and all(type(x) is int for x in shape) and
                1 <= shape[0] <= eval_batch_size and shape[1:] == [50, 4, 64], label + '.bounded_bank')
    calls = integer(d['forward_calls'], label + '.calls', len(shapes), counts['examples'])
    require(min(x[0] for x in shapes) * calls <= counts['examples'] <= max(x[0] for x in shapes) * calls,
            label + '.forward_population')
    require(d['max_input_batch'] == max(x[0] for x in shapes) and d['max_history'] == 50 and
            d['max_reader_bank_elements'] == max(math.prod(x) for x in shapes), label + '.bank_maxima')
    stat(d['reader_weights'], label + '.weights', counts['slots'], counts['retained_slots'], upper=1)
    arithmetic.weight_sum(d['reader_weights']['sum'], counts['examples'] - d['empty_memory_count'], label + '.weight_sum')
    for key in ('query_norm', 'memory_norm', 'output_norm'):
        stat(d[key], label + '.' + key, counts['examples'], counts['retained'])
    for key in ('memory_to_output_norm_ratio', 'residual_to_output_norm_ratio'):
        stat(d[key], label + '.' + key, counts['examples'] - zero)
        require(d[key]['reservoir_value_count'] <= counts['retained'], label + '.ratio_reservoir')
    a, b = d['memory_to_output_norm_ratio'], d['residual_to_output_norm_ratio']
    require(a['reservoir_value_count'] == b['reservoir_value_count'], label + '.ratio_counts')
    for key in ('sum', 'mean', 'min', 'max'):
        if a[key] is None:
            require(b[key] is None, label + '.ratio_empty')
        else:
            arithmetic.close(b[key], abs(d['lambda']) * a[key], label + '.ratio.' + key)
    for i, row in enumerate(d['examples_reservoir']):
        here = label + '.sample' + str(i)
        weights = row['reader_weights']
        require(len(weights) == row['valid_slots'] and all(0 <= v <= 1 for v in weights), here + '.weights')
        # FP32 softmax normalization, separate from diagnostic float64 sums.
        arithmetic.weight_sum(sum(weights), 1 if weights else 0, here + '.softmax_sum')
        for weight in weights:
            sample_in_stat(weight, d['reader_weights'], here + '.weight')
        for key in ('query_norm', 'memory_norm', 'output_norm'):
            sample_in_stat(row[key], d[key], here + '.' + key)
        if not weights:
            require(row['memory_norm'] == 0 and row['output_norm'] == row['query_norm'], here + '.empty_memory')
        if row['output_norm']:
            arithmetic.close(row['memory_to_output_norm_ratio'], row['memory_norm'] / row['output_norm'], here + '.norm_ratio')
            arithmetic.close(row['residual_to_output_norm_ratio'], abs(d['lambda']) * row['memory_to_output_norm_ratio'], here + '.residual_ratio')
            for key in ('memory_to_output_norm_ratio', 'residual_to_output_norm_ratio'):
                sample_in_stat(row[key], d[key], here + '.' + key)
        else:
            require(row['memory_to_output_norm_ratio'] is None and row['residual_to_output_norm_ratio'] is None,
                    here + '.zero_denominator')


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


def audit_run(r, variant, files, saved, terminal, arithmetic, c, read, report):
    require(r['status'] == 'PASS' and r['stage'] == 'COMPLETED', variant + '.fit_status')
    require(not any(k in r for k in ('error', 'traceback', 'validation_error')), variant + '.fit_errors')
    report.validate_record(r, variant)
    require(hashlib.sha256(json.dumps(r['config'], sort_keys=True).encode()).hexdigest() == r['config_sha256'], variant + '.config_sha')
    require(r['epoch_indexing'] == 'zero-based' and r['selection_split'] == 'VALID' and
            r['evaluation_mode'] == 'full-ranking' and r['history_length'] == 50 and
            r['kernel_length_for_max_history'] == 56, variant + '.protocol')
    require(r['effective_config_parity']['status'] == 'PASS' and
            r['effective_config_parity']['allowed_differences'] == ['checkpoint_dir', 'memory_mode'], variant + '.effective_config')
    paths = c.paths(variant)
    runtime = files / paths['runtime'].relative_to(c.HERE)
    meta = read(runtime / 'checkpoints/best_metadata.json')
    checkpoint = saved['checkpoints'][str(paths['checkpoint'].relative_to(c.HERE))]
    require(checkpoint == terminal['checkpoints'][variant], variant + '.terminal_checkpoint')
    require(checkpoint['sha256'] == r['checkpoint_sha256'] == meta['checkpoint_sha256'] and
            re.fullmatch('[0-9a-f]{64}', checkpoint['sha256']), variant + '.checkpoint_sha')
    expected_path = str(REMOTE / paths['checkpoint'].relative_to(c.ROOT))
    require(r['checkpoint_path'] == checkpoint['path'] == expected_path and
            r['checkpoint_metadata_path'] == str(REMOTE / paths['metadata'].relative_to(c.ROOT)), variant + '.checkpoint_path')
    integer(checkpoint['bytes'], variant + '.checkpoint_bytes', 1)
    for key in ('run_id', 'mode', 'memory_mode', 'seed', 'execution_commit', 'config_sha256', 'source_hash', 'core_hash'):
        require(meta[key] == r[key], variant + '.checkpoint_owner.' + key)
    require(meta['epoch_indexing'] == 'zero-based' and meta['epoch'] == r['best_epoch'] and
            meta['metrics'] == r['best_valid_metrics'] and meta['memory'] == r['best_diagnostics'].get('memory'), variant + '.checkpoint_selection')
    logs = list((runtime / 'log').rglob('*.log'))
    require(len(logs) == 1, variant + '.metric_log_count')
    parsed = parse_logs(logs[0].read_text())
    require(parse_logs((runtime / 'process/stderr.log').read_text()) == parsed, variant + '.process_recbole_logs')
    evaluations, metrics, training, finishes = parsed
    require(len(evaluations) == len(metrics) == len(training) == r['actual_epochs'], variant + '.log_epochs')
    scores = []
    best, stale, stop = -math.inf, 0, None
    address_reference = None
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
        if variant == 'no_memory':
            require('memory' not in diag, label + '.no_memory_diagnostics')
        else:
            memory_diagnostics(diag['memory'], variant, label, arithmetic, r['effective_config']['eval_batch_size'])
            address = {k: v for k, v in diag['memory'].items() if k in ADDRESS_FIELDS}
            samples = [{k: v for k, v in x.items() if k in SAMPLE_ADDRESS_FIELDS} for x in diag['memory']['examples_reservoir']]
            address['examples_reservoir'] = samples
            if address_reference is None:
                address_reference = address
            require(address == address_reference, label + '.epoch_address_stability')
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
    row = dict(variant=variant, run_id=r['run_id'], parameters=r['parameter_count'], epochs=r['actual_epochs'],
               metric_cells=12 * r['actual_epochs'], checkpoint_sha256=r['checkpoint_sha256'], checkpoint_bytes=checkpoint['bytes'],
               best_epoch=selected, ndcg10=r['best_valid_metrics']['ndcg@10'], hr10=r['best_valid_metrics']['hit@10'],
               first27_complete=r['first27_complete'], first27_best_ndcg10=r['first27_best_ndcg10'],
               train_seconds=r['train_seconds'], valid_seconds=r['valid_seconds'],
               peak_allocated_bytes=r['peak_gpu_allocated_bytes'], peak_reserved_bytes=r['peak_gpu_reserved_bytes'])
    return row, address_reference


ADDRESS_FIELDS = {'scope', 'query_scope', 'examples', 'nonempty_histories', 'reservoir',
                  'history_length_histogram', 'valid_slot_count_histogram', 'at_least_four_past_events_count',
                  'at_least_four_past_events_fraction', 'empty_memory_count', 'empty_memory_fraction',
                  'selected_sets_equal_count', 'selected_sets_equal_fraction', 'time_anchor_reach',
                  'history_length', 'temporal_span_ms', 'temporal_span_R0', 'selected_set_jaccard',
                  'index_memory', 'time_memory'}
SAMPLE_ADDRESS_FIELDS = {'stream_ordinal', 'input_index', 'query_position', 'history_length', 'valid_slots',
                         'temporal_span_ms', 'temporal_span_R0', 'index_memory', 'time_memory',
                         'selected_sets_equal', 'selected_set_jaccard'}


def audit(folder, execution=EXECUTION):
    # Config imports are pure stdlib. Select the preserved attempt before the
    # first import; never mutate existing module globals or scientific files.
    folder = Path(folder).resolve()
    saved = json.loads((folder / 'preservation_manifest.json').read_text())
    attempt, job = saved['execution_attempt'], saved['job_id']
    require(attempt in ('001', '002') and re.fullmatch('[0-9]+', job), 'Preserved attempt/job')
    os.environ.setdefault('TIME_MEMORY_ATTEMPT', attempt)
    from experiments.mamba3_time_memory import config as c, provenance as p, report
    from experiments.mamba3_mimo_time.records import read, sha, create, now, digest, accepted_cases
    require(c.EXECUTION_ATTEMPT == attempt, 'Set TIME_MEMORY_ATTEMPT to the preserved attempt before importing config')
    require('torch' not in sys.modules, 'Audit must run in a fresh process without Torch')
    require(re.fullmatch('[0-9a-f]{40}', execution) and saved['execution_commit'] == execution, 'Expected execution identity')
    manifest = p.verify(); files = folder / 'files'
    require(saved['source_hash'] == manifest['source_hash'], 'Preserved source identity')
    if execution == EXECUTION:
        require(manifest['source_hash'] == SOURCE and len(manifest['files']) == 410, 'Initial frozen source')
    require(saved['checkpoint_loading'] is False and saved['weights_copied'] is False, 'Preservation scope')
    preserved = {}
    for row in saved['files']:
        rel = Path(row['path']); path = files / rel
        require(not rel.is_absolute() and '..' not in rel.parts and path.resolve().is_relative_to(files.resolve()) and
                not path.is_symlink() and path.is_file(), 'Safe preserved path ' + str(rel))
        require(row['path'] not in preserved, 'Duplicate preserved path')
        require(row['cluster_path'] == str(REMOTE / 'experiments/mamba3_time_memory' / rel), 'Preserved canonical path')
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Preserved bytes ' + str(rel))
        preserved[row['path']] = row
    require(set(preserved) == {str(x.relative_to(files)) for x in files.rglob('*') if x.is_file()}, 'Unmanifested preserved file')
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256(subprocess.check_output(['git', 'show', execution + ':' + name], cwd=c.ROOT)).hexdigest()
        require(actual == expected, 'Published execution blob ' + name)
    require(sha(files / c.MANIFEST.name) == sha(c.MANIFEST) and
            sha(files / 'study_plan.json') == sha(c.HERE / 'study_plan.json') and
            sha(files / 'DESIGN.md') == sha(c.HERE / 'DESIGN.md'), 'Frozen documents')
    require(sha(c.PILOT) == c.plan()['control_reference_sha256'], 'Historical control bytes')
    scheduler = read(folder / 'scheduler_terminal.json'); raw = scheduler['sacct_raw'].splitlines()
    parsed = [dict(zip(raw[0].split('|'), line.split('|'))) for line in raw[1:] if line]
    require(parsed == scheduler['steps'] and [x for x in parsed if x['JobIDRaw'] == job] == [scheduler['job']], 'Scheduler raw binding')
    require(parsed and all(x['JobIDRaw'] in (job, job + '.batch', job + '.extern') and
                          x['State'] == 'COMPLETED' and x['ExitCode'] == '0:0' for x in parsed), 'Terminal scheduler')
    require(len({x['JobIDRaw'] for x in parsed}) == len(parsed), 'Duplicate scheduler steps')
    logs, runs = files / ('slurm_logs/attempt_' + attempt), files / ('runs/attempt_' + attempt)
    login, reservation, submission = [read(logs / name) for name in ('login_verification.json', 'reservation.json', 'submission.json')]
    base = p.bindings(execution, manifest)
    p.validate_ownership(login, reservation, sha(logs / 'login_verification.json'), base, job, submission)
    coverage = read(logs / 'train_coverage.json'); coverage_sha = sha(logs / 'train_coverage.json')
    require(all(coverage.get(k) == v for k, v in base.items()) and coverage['status'] == 'PASS', 'TRAIN coverage owner')
    require(coverage_sha == login['coverage_sha256'] == reservation['coverage_sha256'], 'Frozen TRAIN coverage SHA chain')
    scope = dict(stage='TRAIN_ADDRESS_COVERAGE', split='TRAIN', sampled_examples=10000, train_population=1062567,
                 anchors=list(ANCHORS), reference_ms=REFERENCE, max_history=50, max_examples=10000, batch_size=256,
                 scientific_fit_started=False, scientific_fits_started=0, model_instances_created=0, model_forward_count=0,
                 train_loaders_created=0, valid_loaders_created=0, test_loaders_created=0, target_fields_read=False,
                 test_evaluation_count=0, cuda_visible_devices='', cuda_initialized_before=False,
                 cuda_initialized_after=False, cuda_available=False)
    require(all(coverage.get(k) == v for k, v in scope.items()), 'TRAIN-only coverage scope')
    indices = [i * (1062567 - 1) // (10000 - 1) for i in range(10000)]
    require(coverage['selected_train_indices'] == indices and coverage['selected_indices_sha256'] == digest(indices) and
            coverage['index_rule'] == 'floor(i*(N-1)/(n-1)) for i=0..n-1; n=min(N,10000); [0] when n=1', 'TRAIN sample indices')
    old = read(c.PILOT)
    runtime_identity(login, old['runtime'], manifest, 'Login', False)
    require(coverage['protocol'] == old['protocol'] and coverage['manifest_sha256'] == old['manifest_sha256'] and
            coverage['frozen_train_time_stats_sha256'] == old['train_time_stats_sha256'], 'TRAIN data identity')
    cfg = old['effective_config']
    fields = [cfg['ITEM_ID_FIELD'] + cfg['LIST_SUFFIX'], cfg['TIME_FIELD'] + cfg['LIST_SUFFIX'], cfg['ITEM_LIST_LENGTH_FIELD']]
    require(coverage['input_field_names_read'] == fields, 'Coverage input-only field names')
    hashes = coverage['selected_input_hashes']
    require(set(hashes) == {'item_history', 'history_length', 'precise_history_timestamps'} and
            all(re.fullmatch('[0-9a-f]{64}', v) for v in hashes.values()) and
            coverage['selected_records_sha256'] == digest(hashes), 'Coverage input hash algebra')
    arithmetic = Arithmetic()
    address_diagnostics(coverage, 'TRAIN_coverage', arithmetic, 10000, indices)
    require(coverage['selected_sets_equal_count'] < 10000, 'Coverage informative admission')
    base.update(job_id=job, reservation_token=reservation['token'], reservation_sha256=sha(logs / 'reservation.json'),
                login_verification_sha256=sha(logs / 'login_verification.json'), coverage_sha256=coverage_sha)

    def owner(record, label, status=True):
        require(all(record.get(k) == v for k, v in base.items()), label + '.owner')
        if status:
            require(record.get('status') == 'PASS', label + '.status')

    cpu_rows = []
    for prefix in ('cpu_preflight_', 'no_git_preflight_'):
        name = prefix + execution + '.json'; cpu = read(logs / name); tests = cpu['cpu_tests']
        require(cpu['status'] == 'PASS' and cpu['execution_commit'] == execution and cpu['source_hash'] == manifest['source_hash'] and
                cpu['execution_attempt'] == attempt and cpu['cuda_initialized'] is False and cpu['cuda_mask'] == '', 'CPU provenance')
        require(tests['run'] == 67 and not any(tests[k] for k in ('failures', 'errors', 'skipped')), 'CPU67 tests')
        require(cpu['parameter_counts'] == c.COUNTS and cpu['scientific_fits'] == 0 and cpu['test_evaluation_count'] == 0 and
                cpu['TEST'] == 'NOT_RUN' and cpu['mimo_model_forward_calls'] == 0 and cpu['gradcheck'] == 'PASS', 'CPU scope')
        runtime_identity(cpu, old['runtime'], manifest, name, False)
        cpu_rows.append(dict(file=name, tests=67, status='PASS'))
    inherited = read(runs / 'inherited_kernel.json'); owner(inherited, 'Inherited')
    require(inherited['inherited'] == p.inherited(), 'Inherited exact admission')
    runtime_identity(inherited, old['runtime'], manifest, 'Inherited', True)
    gate = read(runs / 'targeted_gate.json'); owner(gate, 'Gate')
    runtime_identity(gate, old['runtime'], manifest, 'Gate', True)
    specs = c.plan()['required_cases']
    require(gate['required_cases'] == specs and accepted_cases(gate['cases'], specs), 'Targeted required cases')
    checks = sum(len(x['checks']) for x in gate['cases'])
    require(len(gate['cases']) == 11 and checks == 429 and gate['scientific_fits'] == 0, 'Gate scope')
    for case in gate['cases']:
        require(not any(case.get(k) for k in ('missing_keys', 'unexpected_keys', 'failed_keys', 'traceback')), 'Gate final case errors')
    smoke = read(runs / 'smoke.json'); owner(smoke, 'Smoke'); p.validate_smoke(smoke)
    runtime_identity(smoke, old['runtime'], manifest, 'Smoke', True)
    require(smoke['targeted_gate_sha256'] == sha(runs / 'targeted_gate.json') and smoke['scientific_fits'] == 0, 'Smoke gate/scope')
    for row in smoke['rows']:
        mode = row['memory_mode']
        allocated = integer(row['peak_allocated_bytes'], 'Smoke allocated', 1)
        integer(row['peak_reserved_bytes'], 'Smoke reserved', allocated)
        for i, step in enumerate(row['steps']):
            require(step['hook_removed'] is True, 'Smoke hook cleanup')
            if mode != 'no_memory':
                for boundary in ('before', 'after'):
                    arithmetic.gate(step['beta_' + boundary], step['lambda_' + boundary], 'smoke.' + mode + '.' + str(i) + '.' + boundary)
                if i == 0:
                    require(step['beta_before'] == step['lambda_before'] == 0, 'Smoke zero initialization')
                else:
                    require(step['beta_before'] == row['steps'][i - 1]['beta_after'], 'Smoke beta continuity')
    pipeline = read(logs / 'pipeline_status.json'); terminal = read(runs / 'terminal_metadata.json')
    for label, record in [('Pipeline', pipeline), ('Terminal', terminal)]:
        owner(record, label)
        require((record['scientific_fits_started'], record['scientific_fits_completed'], record['unknown_scientific_starts']) == (3, 3, 0), label + '.counters')
        require(record['summary_status'] == 'PASS' and not any(k in record for k in ('error', 'traceback', 'report_traceback')), label + '.errors')
    require([(x['stage'], x['status']) for x in pipeline['stages']] == [(x, 'PASS') for x in ('gate', 'smoke', *MODES)], 'Ordered five stages')
    previous = datetime.fromisoformat(pipeline['started_at'])
    for stage in pipeline['stages']:
        start, end = (datetime.fromisoformat(stage[k]) for k in ('started_at', 'finished_at'))
        require(previous <= start <= end, 'Stage execution ordering'); previous = end
    require(previous <= datetime.fromisoformat(pipeline['finished_at']), 'Pipeline finish ordering')
    require(terminal['pipeline_sha256'] == sha(logs / 'pipeline_status.json') and terminal['checkpoint_loading'] is False,
            'Terminal pipeline SHA/scope')
    require({k: v for k, v in pipeline.items() if k != 'stages'} ==
            {k: v for k, v in terminal.items() if k not in ('pipeline_sha256', 'checkpoint_loading', 'checkpoints')}, 'Terminal pipeline fields')
    require(set(terminal['checkpoints']) == set(MODES) and set(saved['checkpoints']) ==
            {str(c.paths(v)['checkpoint'].relative_to(c.HERE)) for v in MODES}, 'Exactly three checkpoints')
    owner(read(logs / 'pipeline.lock'), 'Pipeline lock', False)
    records, rows, addresses = {}, [], {}
    expected_results = {c.paths(v)['result'].name for v in MODES}
    require({x.name for x in runs.glob('mamba3_time_memory_*_seed*.json')} == expected_results, 'Exactly planned scientific results')
    for variant in MODES:
        paths = c.paths(variant); r = read(files / paths['result'].relative_to(c.HERE)); owner(r, variant)
        runtime_identity(r, old['runtime'], manifest, variant, True)
        require(r['runtime']['gpu'] == inherited['runtime']['gpu'] == gate['runtime']['gpu'] == smoke['runtime']['gpu'],
                variant + '.allocation_gpu_identity')
        lock = read(files / paths['lock'].relative_to(c.HERE)); owner(lock, variant + '.lock', False)
        for key in ('run_id', 'memory_mode', 'seed', 'targeted_gate_sha256', 'smoke_sha256'):
            require(lock[key] == r[key], variant + '.lock.' + key)
        require(lock['scientific_fit_started'] is False and lock['status'] == 'RUNNING', variant + '.fresh_lock')
        require(r['targeted_gate_sha256'] == sha(runs / 'targeted_gate.json') and r['smoke_sha256'] == sha(runs / 'smoke.json'), variant + '.admission')
        row, address = audit_run(r, variant, files, saved, terminal, arithmetic, c, read, report)
        rows.append(row); records[variant] = r; addresses[variant] = address
    for variant in MODES[1:]:
        require(all(records[variant][k] == records['no_memory'][k] for k in PAIRING), variant + '.paired_fields')
    require(set(records['no_memory']['rng_components']) == {'python', 'numpy', 'cpu', 'cuda', 'aggregate', 'loader_generator'}, 'RNG components')
    require(addresses['index_memory'] == addresses['time_memory'], 'Within-triple exact VALID address diagnostics')
    replay = report.replay_check(records['no_memory'])
    summary = read(runs / 'pilot_summary.json'); owner(summary, 'Summary')
    expected = report.summarize(records)
    require(all(summary[k] == v for k, v in expected.items()) and summary['fresh_control_replay'] == replay and
            summary['blocking_reason'] is None, 'Recomputed summary/replay')
    for variant, row in zip(MODES, summary['rows']):
        if variant == 'no_memory':
            require('gate_boundaries' not in row, 'No-memory summary gate absent')
            continue
        history = [x['diagnostics']['memory'] for x in records[variant]['history']]
        for field in ('beta', 'lambda'):
            negative = sum(x[field] < 0 for x in history)
            require(row['gate_boundaries'][field] == dict(
                best=history[records[variant]['best_epoch']][field], final=history[-1][field],
                max_abs=max(abs(x[field]) for x in history), negative_count=negative,
                observed_epochs=len(history), negative_fraction=negative / len(history)),
                variant + '.independent_gate_boundaries.' + field)
    # Recompute contrasts independently of the reporting implementation.
    for row, (left, right) in zip(summary['contrasts'], (('time_memory', 'index_memory'), ('time_memory', 'no_memory'), ('index_memory', 'no_memory'))):
        a, b = records[left], records[right]; x, y = a['best_valid_metrics']['ndcg@10'], b['best_valid_metrics']['ndcg@10']
        first = a['first27_best_ndcg10'] - b['first27_best_ndcg10'] if a['first27_complete'] and b['first27_complete'] else None
        require(row == dict(comparison=left + ' - ' + right, status='COMPLETE', delta=x - y,
                            relative_percent=100 * (x - y) / y if y else None, first27_delta=first), 'Independent contrast')
    require(len(summary['contrasts']) == 3, 'Three planned contrasts')
    errors, warnings = [], []
    for path in logs.rglob('*'):
        if path.suffix not in ('.log', '.out', '.err'):
            continue
        for line in path.read_text().splitlines():
            if re.search(r'Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:|\b(?:nan|inf)\b', line, re.I):
                errors.append(dict(path=str(path.relative_to(files)), line=line))
            if 'Warning:' in line:
                warnings.append(dict(path=str(path.relative_to(files)), line=line))
    require(not errors, 'Runtime errors ' + repr(errors[:4]))
    require('torch' not in sys.modules, 'Audit imported Torch')
    value = dict(status='PASS', job_id=job, execution_attempt=attempt, execution_commit=execution,
                 source_hash=manifest['source_hash'], source_blobs=len(manifest['files']), audited_at=now(),
                 preserved_files=len(preserved), preserved_bytes=sum(x['bytes'] for x in preserved.values()),
                 scheduler=scheduler['job'], cpu=cpu_rows, gpu_cases=11, gpu_checks=checks, smoke_steps_per_variant=3,
                 scientific_fits_started=3, scientific_fits_completed=3, complete_triples_verified=1,
                 TEST='NOT_RUN', test_evaluation_count=0, rows=rows, epochs=sum(x['epochs'] for x in rows),
                 metric_cells=sum(x['metric_cells'] for x in rows), contrasts=summary['contrasts'],
                 fresh_control_replay=replay, paired_fields=list(PAIRING), coverage_sha256=coverage_sha,
                 coverage_selected_sets_equal_fraction=coverage['selected_sets_equal_fraction'],
                 runtime_errors=errors, warnings=warnings, diagnostic_arithmetic=arithmetic.result(),
                 checkpoint_loading=False, model_forward_count=0, torch_imported=False,
                 runtime_identity=dict(dependencies_match_historical=True, imported_paths_in_frozen_manifest=True,
                                       gpu=inherited['runtime']['gpu'],
                                       historical_gpu_name_not_required='A100 model/capacity may differ from historical node; exact GPU label required within this allocation; CPU/login gpu is null'),
                 limitations=[
                     'Checkpoint SHA/bytes were streamed during preservation and bound to terminal/selected-epoch metadata; weights are not locally loaded or rehashed.',
                     'All-query saved counts, sums, means, histograms and units checked algebraically; raw query tensors are not preserved.',
                     'Eight published examples checked against deterministic reservoir membership, causal indices, masks, physical ages, ordinal selection and overlap. Full 256-query quantiles cannot be recomputed from eight examples.',
                     'Elapsed-time selector cannot be replayed from samples without their complete timestamp histories; GPU gate provides saved selector/intervention evidence.',
                     'Norms and reader weights are saved values: checked bounds/normalization/ratios, not reconstructed from representations or checkpoint tensors.',
                     'RecBole log timers and saved wrapper timers have different envelopes; both logs agree, saved epoch sums and memory maxima checked, no false equality of distinct timers.',
                     'No-Git CPU status is preserved evidence from its separate invocation; the CPU JSON itself contains no deny-git flag.'
                 ],
                 result_sha256={v: sha(files / c.paths(v)['result'].relative_to(c.HERE)) for v in MODES},
                 summary_sha256=sha(runs / 'pilot_summary.json'),
                 preservation_manifest_sha256=sha(folder / 'preservation_manifest.json'),
                 scheduler_sha256=sha(folder / 'scheduler_terminal.json'))
    output = folder / 'independent_audit.json'
    if output.exists():
        prior = read(output)
        require({k: v for k, v in prior.items() if k != 'audited_at'} ==
                {k: v for k, v in value.items() if k != 'audited_at'}, 'Existing audit changed; do not replace evidence')
        return prior
    create(output, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--execution', default=EXECUTION, help='Exact expected published execution commit')
    args = parser.parse_args()
    result = audit(args.folder, args.execution)
    print(json.dumps({k: v for k, v in result.items() if k not in ('warnings', 'paired_fields', 'diagnostic_arithmetic')}, indent=2))
