"""Frozen TRAIN history metadata; no loaders, recommender or model execution."""
import json
import os
import traceback
from datetime import datetime, timezone

import torch

from experiments.mamba3_mimo_time.records import create, update, now, digest
from experiments.mamba3_time_memory.coverage import (
    MAX_EXAMPLES, BATCH_SIZE, INDEX_RULE, fixed_indices, tensor_fingerprint,
)
from experiments.mamba3_timeaware.time_inputs import history_gaps

PERIODS_MS = (21600000, 86400000)
PHASE_BINS = 24
DAY_MS = 86400000


def _distribution(values):
    values = values.detach().double().cpu().reshape(-1)
    if not values.numel():
        return dict(count=0, min=None, max=None, mean=None, quantiles=None)
    return dict(count=values.numel(), min=values.min().item(), max=values.max().item(),
                mean=values.mean().item(), quantiles=dict(zip(
                    ('p00', 'p25', 'p50', 'p75', 'p90', 'p99', 'p100'),
                    torch.quantile(values, torch.tensor([0, .25, .5, .75, .9, .99, 1],
                                                       dtype=torch.float64)).tolist())))


def _phase_summary(times, active_times, period):
    # Integer milliseconds have already been checked before this operation.
    residues = times.remainder(period).long()
    active_residues = active_times.remainder(period).long()
    bins = torch.div(residues * PHASE_BINS, period, rounding_mode='floor')
    active_bins = torch.div(active_residues * PHASE_BINS, period, rounding_mode='floor')
    dates = torch.div(times.long(), DAY_MS, rounding_mode='floor')
    unique_days = dates.unique(sorted=True)
    by_date = []
    for day in unique_days.tolist():
        selected = bins[dates == day]
        counts = torch.bincount(selected, minlength=PHASE_BINS)
        by_date.append(dict(unix_day=day,
                            utc_date=datetime.fromtimestamp(day * 86400, timezone.utc).date().isoformat(),
                            occurrences=selected.numel(), bin_counts=counts.tolist()))
    date_bins = torch.zeros(PHASE_BINS, dtype=torch.int64)
    for row in by_date:
        date_bins += torch.tensor(row['bin_counts']) > 0
    # Unique (phase, date) pairs remove repeated windows from this diagnostic.
    pairs = torch.stack((residues, dates), -1).unique(dim=0)
    _, phase_date_counts = pairs[:, 0].unique(return_counts=True)
    counts = torch.bincount(bins, minlength=PHASE_BINS)
    active_counts = torch.bincount(active_bins, minlength=PHASE_BINS)
    span = (times.max() - times.min()).item()
    return dict(period_ms=period, span_in_periods=span / period,
                complete_period_lengths_in_range=int(span // period),
                phase_bins=PHASE_BINS, bin_edges_fraction=[i / PHASE_BINS for i in range(PHASE_BINS + 1)],
                valid_bin_counts=counts.tolist(), active_bin_counts=active_counts.tolist(),
                occupied_valid_bins=int((counts > 0).sum()),
                occupied_active_bins=int((active_counts > 0).sum()),
                unique_active_exact_phases=int(active_residues.unique().numel()),
                unique_exact_phases=int(residues.unique().numel()),
                unique_exact_phases_on_multiple_dates=int((phase_date_counts > 1).sum()),
                exact_phase_repeat_fraction=float((phase_date_counts > 1).double().mean()),
                dates_per_phase_bin=date_bins.tolist(),
                bins_seen_on_multiple_dates=int((date_bins > 1).sum()), by_utc_date=by_date)


def audit_histories(items, lengths, timestamps, indices, batch_size=BATCH_SIZE):
    """Audit a fixed sample of observed TRAIN windows, never their targets.

    Events are counted as occurrences in windows, not as distinct interactions.
    PASS only certifies nondegenerate coverage and millisecond representation;
    it is not evidence that a behavioural period exists.
    """
    if any(t.device.type != 'cpu' for t in (items, lengths, timestamps)):
        raise ValueError('Coverage must stay on CPU')
    if timestamps.dtype != torch.float64 or items.shape != timestamps.shape or items.ndim != 2:
        raise ValueError('Expected exact float64 TRAIN input histories [B,L]')
    if lengths.shape != (items.shape[0],) or len(indices) != items.shape[0]:
        raise ValueError('TRAIN sample length/index mismatch')
    if items.shape[1] > 50 or not 1 <= len(indices) <= MAX_EXAMPLES or batch_size < 1:
        raise ValueError('Coverage scope/batch drift')
    valid = items != 0
    expected = torch.arange(items.shape[1])[None] < lengths[:, None]
    if (lengths < 1).any() or not torch.equal(valid, expected):
        raise ValueError('Expected nonempty right-padded TRAIN histories')
    times = timestamps[valid]
    finite = bool(torch.isfinite(times).all())
    integer_ms = finite and bool((times == times.round()).all())
    exact_integer_range = finite and bool((times.abs() <= 2 ** 53 - 1).all())
    semantics = dict(dtype='float64', finite=finite, integral_milliseconds=integer_ms,
                     within_exact_integer_range=exact_integer_range,
                     unit='milliseconds since Unix epoch', origin='Unix epoch, fixed globally',
                     source='PreciseHistoryDataset copies original timestamp values before RecBole float32 conversion',
                     user_timezone='NOT_RECORDED', local_time_interpretation=False)
    hashes = dict(item_history=tensor_fingerprint(items), history_length=tensor_fingerprint(lengths),
                  precise_history_timestamps=tensor_fingerprint(timestamps))
    result = dict(scope='fixed TRAIN input-history sample; all valid positions in each sampled window',
                  examples=len(indices), timestamp_semantics=semantics,
                  selected_input_hashes=hashes, selected_records_sha256=digest(hashes),
                  model_forward_count=0, target_fields_read=False, training_rng_consumed=False,
                  counts_scope='history event occurrences; windows can overlap; unique timestamps are not unique events',
                  user_history_span_scope='available windows of at most 50 events, not full user lifetimes',
                  distinct_users='NOT_RECORDED', distinct_interactions='NOT_RECORDED')
    reasons = []
    if not all((finite, integer_ms, exact_integer_range)):
        reasons.append('TRAIN input timestamps are not finite, exactly represented integer milliseconds')
    if reasons:
        return dict(result, status='BLOCKED', blocking_reason='; '.join(reasons))
    gaps, active = history_gaps(timestamps, valid)
    adjacent = timestamps[:, 1:] - timestamps[:, :-1]
    active_count = int(active.sum())
    negative_count = int((adjacent[active[:, 1:]] < 0).sum())
    same_count = int((adjacent[active[:, 1:]] == 0).sum())
    active_times = timestamps[active]
    first, last = timestamps[:, 0], timestamps[torch.arange(len(lengths)), lengths - 1]
    chronological_span = last - first
    cumulative_span = gaps.sum(1)
    unique_times, multiplicity = times.unique(return_counts=True)
    minimum, maximum = times.min().item(), times.max().item()
    try:
        utc_range = [datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()
                     for value in (minimum, maximum)]
        periods = [_phase_summary(times, active_times, period) for period in PERIODS_MS]
    except (ValueError, OverflowError, OSError) as exc:
        return dict(result, status='BLOCKED', blocking_reason='Unix millisecond calendar representation failed: ' + str(exc))
    if minimum <= 0:
        reasons.append('Expected positive Unix milliseconds in the frozen KuaiRand data')
    if active_count == 0:
        reasons.append('No active adjacent observed events in audited TRAIN histories')
    if maximum - minimum < max(PERIODS_MS):
        reasons.append('Audited TRAIN timestamps span less than one full 24-hour basis period')
    for row in periods:
        if row['unique_active_exact_phases'] < 2:
            reasons.append(f"Active TRAIN phases are constant for {row['period_ms']} ms")
        if len(row['by_utc_date']) < 2:
            reasons.append('Audited TRAIN timestamps do not cover two UTC dates')
    result.update(status='BLOCKED' if reasons else 'PASS',
                  blocking_reason='; '.join(reasons) if reasons else None,
                  decision_rule='finite exact integer positive Unix milliseconds; active events with nonconstant exact phases for each period; range >=24h and >=2 UTC dates as conservative identifiability checks. No phase-bin occupancy or metric-based threshold.',
                  timestamp_min_ms=minimum, timestamp_max_ms=maximum, utc_range=utc_range,
                  range_ms=maximum - minimum, valid_event_occurrences=times.numel(),
                  active_event_occurrences=active_count, first_event_occurrences=len(indices),
                  padding_occurrences=int((~valid).sum()), distinct_timestamps=unique_times.numel(),
                  repeated_timestamp_occurrence_fraction=1 - unique_times.numel() / times.numel(),
                  occurrences_with_nonunique_timestamp_fraction=float(multiplicity[multiplicity > 1].sum()) / times.numel(),
                  equal_adjacent_timestamp_count=same_count,
                  equal_adjacent_timestamp_fraction=same_count / active_count if active_count else None,
                  negative_adjacent_gap_count=negative_count,
                  negative_gap_policy='unchanged inherited clamp_min(0); negative gaps counted explicitly',
                  chronological_history_span_ms=_distribution(chronological_span),
                  clamped_cumulative_history_span_ms=_distribution(cumulative_span),
                  history_length=_distribution(lengths), periods=periods,
                  interpretation='Coverage of basis periods does not establish behavioural cycles; UTC is only the technical global clock.')
    return result


def run(base):
    from . import config as c
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError('Coverage requires a separate subprocess with CUDA hidden before Torch import')
    c.plan()
    result = dict(base)
    result.update(status='RUNNING', stage='TRAIN_PHASE_COVERAGE', started_at=now(),
                  scientific_fit_started=False, scientific_fits_started=0,
                  model_instances_created=0, model_forward_count=0,
                  train_loaders_created=0, valid_loaders_created=0, test_loaders_created=0,
                  test_evaluation_count=0, target_fields_read=False, split='TRAIN',
                  periods_ms=list(PERIODS_MS), phase_bins=PHASE_BINS,
                  max_history=50, max_examples=MAX_EXAMPLES, batch_size=BATCH_SIZE,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  cuda_initialized_before=False, cuda_available=False)
    create(c.COVERAGE, result)
    try:
        from recbole.config import Config
        from recbole.utils import init_seed
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        from experiments.mamba3_timeaware.dataset import PreciseHistoryDataset
        from experiments.mamba3_timeaware.run import verify_protocol
        from experiments.mamba3_time_confirmation.config import MANIFEST_SHA, STATS_SHA
        torch.set_num_threads(4)
        cfg = Config(model=ThreeTimeMamba3Rec, config_dict=c.settings('baseline_dual', seed=2026, device='cpu'))
        if torch.cuda.is_initialized() or cfg['device'].type != 'cpu' or cfg['use_gpu']:
            raise RuntimeError('CPU coverage Config reopened CUDA')
        init_seed(cfg['seed'] + cfg['local_rank'], cfg['reproducibility'])
        protocol = verify_protocol(cfg, check_sha=True)
        dataset = PreciseHistoryDataset(cfg)
        train_ds, reserved_valid, reserved_test = dataset.build()
        del reserved_valid, reserved_test
        population = len(train_ds)
        if population != 1062567 or train_ds.item_num != 7112:
            raise ValueError('Frozen TRAIN data count/mapping changed')
        indices = fixed_indices(population)
        tensor_indices = torch.tensor(indices, dtype=torch.long)
        item_field = cfg['ITEM_ID_FIELD'] + cfg['LIST_SUFFIX']
        time_field = cfg['TIME_FIELD'] + cfg['LIST_SUFFIX']
        length_field = cfg['ITEM_LIST_LENGTH_FIELD']
        audit = audit_histories(train_ds.inter_feat[item_field][tensor_indices],
                                train_ds.inter_feat[length_field][tensor_indices],
                                train_ds.inter_feat[time_field][tensor_indices], indices)
        result.update(audit, train_population=population, sampled_examples=len(indices),
                      index_rule=INDEX_RULE, selected_train_indices=indices,
                      selected_indices_sha256=digest(indices),
                      input_field_names_read=[item_field, time_field, length_field],
                      protocol=protocol, manifest_sha256=MANIFEST_SHA,
                      frozen_train_time_stats_sha256=STATS_SHA,
                      config_label='ThreeTimeMamba3Rec, CPU data preparation only',
                      split_preparation='existing dataset.build describes all splits; only TRAIN input histories are inspected',
                      rng_scope='audit statistics consume no RNG; isolated RecBole data setup retains historical init_seed',
                      cuda_initialized_after=torch.cuda.is_initialized())
        result['summary'] = {key: result.get(key) for key in (
            'status', 'scope', 'sampled_examples', 'valid_event_occurrences',
            'active_event_occurrences', 'timestamp_min_ms', 'timestamp_max_ms',
            'utc_range', 'range_ms', 'equal_adjacent_timestamp_fraction',
            'negative_adjacent_gap_count', 'chronological_history_span_ms',
            'clamped_cumulative_history_span_ms', 'selected_records_sha256',
            'decision_rule', 'blocking_reason')}
        result['summary']['periods'] = [{key: row[key] for key in (
            'period_ms', 'span_in_periods', 'complete_period_lengths_in_range',
            'occupied_valid_bins', 'occupied_active_bins', 'unique_active_exact_phases',
            'unique_exact_phases_on_multiple_dates', 'bins_seen_on_multiple_dates')}
            for row in result.get('periods', [])]
        if result['cuda_initialized_after']:
            raise RuntimeError('Coverage initialized CUDA')
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['finished_at'] = now()
        update(c.COVERAGE, result)
    return result


def main():
    from . import provenance as p
    commit = os.environ.get('RUN_COMMIT', '')
    if len(commit) != 40 or any(x not in '0123456789abcdef' for x in commit):
        raise ValueError('Coverage requires frozen exact RUN_COMMIT')
    result = run(p.bindings(commit, p.verify()))
    print(json.dumps({key: result.get(key) for key in
                      ('status', 'sampled_examples', 'active_event_occurrences', 'utc_range', 'blocking_reason')}))
    if result['status'] != 'PASS':
        raise SystemExit('TRAIN periodic coverage blocked GPU submission')


if __name__ == '__main__':
    main()
