"""Detached counters from the existing VALID forward, last valid query only.

The bounded bottom-hash reservoir uses stream ordinals, not any training RNG.
Quantiles are approximate; counts, sums and extrema cover every observed query.
"""
import hashlib
import heapq
import math

import torch

from .memory import select

REFERENCE_MS = 838393.0
ANCHORS = (1.0, 4.0, 16.0, 32.0)
RESERVOIR_CAPACITY = 256
SAVED_EXAMPLES = 8


class ScalarStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None

    def update(self, values):
        values = values.detach().reshape(-1).double().cpu()
        if not values.numel():
            return
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite diagnostic value')
        self.count += values.numel()
        self.total += values.sum().item()
        low, high = values.min().item(), values.max().item()
        self.minimum = low if self.minimum is None else min(self.minimum, low)
        self.maximum = high if self.maximum is None else max(self.maximum, high)

    def result(self, sample=()):
        result = dict(count=self.count, sum=self.total,
                      mean=self.total / self.count if self.count else None,
                      min=self.minimum, max=self.maximum)
        values = [float(v) for v in sample if v is not None]
        result['reservoir_value_count'] = len(values)
        result['approximate_quantiles'] = None
        if values:
            q = torch.tensor(values, dtype=torch.float64)
            result['approximate_quantiles'] = dict(zip(
                ('p00', 'p25', 'p50', 'p75', 'p90', 'p99', 'p100'),
                torch.quantile(q, torch.tensor([0, .25, .5, .75, .9, .99, 1],
                                               dtype=torch.float64)).tolist()))
        return result


class QueryReservoir:
    """Fixed-size deterministic sample independent of random/NumPy/Torch state."""
    def __init__(self, capacity=RESERVOIR_CAPACITY):
        if capacity < 1:
            raise ValueError('Positive reservoir capacity required')
        self.capacity = capacity
        self.heap = []

    def add(self, ordinal, row):
        rank = int.from_bytes(hashlib.sha256(
            ('mamba3-time-memory-query-v1:' + str(ordinal)).encode()).digest(), 'big')
        entry = (-rank, -ordinal, row)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
        elif entry[:2] > self.heap[0][:2]:
            heapq.heapreplace(self.heap, entry)

    def rows(self):
        return [entry[2] for entry in sorted(self.heap, key=lambda x: (-x[0], -x[1]))]


def _last(tensor, positions):
    return tensor[torch.arange(len(positions), device=positions.device), positions].detach().cpu()


def _quantile_values(rows, key, mode=None):
    values = []
    for row in rows:
        value = row[mode][key] if mode else row.get(key)
        values.extend(value if isinstance(value, list) else [value])
    return values


class AddressDiagnostics:
    """Metadata-only addressing audit; no representations or model execution."""
    def __init__(self, scope, capacity=RESERVOIR_CAPACITY):
        self.scope = scope
        self.count = 0
        self.nonempty_history_count = 0
        self.four_past = 0
        self.same_sets = 0
        self.history_histogram = {}
        self.slot_histogram = [0] * 5
        self.time_anchor_reached = [0] * 4
        self.out_of_range = {mode: [0] * 4 for mode in ('index_memory', 'time_memory')}
        self.anchor_available = {mode: [0] * 4 for mode in self.out_of_range}
        self.stats = {key: ScalarStats() for key in
                      ('history_length', 'temporal_span_ms', 'temporal_span_R0',
                       'selected_set_jaccard')}
        self.mode_stats = {mode: {key: ScalarStats() for key in
                                 ('selected_ages_R0', 'selected_ages_ms', 'selected_event_lags')}
                           for mode in self.out_of_range}
        self.reservoir = QueryReservoir(capacity)

    @torch.no_grad()
    def update(self, timestamps, valid, selections=None, extras=None, example_ids=None):
        if timestamps.shape != valid.shape or timestamps.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError('Expected timestamp/boolean valid histories [B,L]')
        if timestamps.dtype != torch.float64:
            raise ValueError('Diagnostics require exact float64 history timestamps')
        modes = ('index_memory', 'time_memory')
        if selections is None:
            selections = {mode: select(timestamps, valid, mode) for mode in modes}
        if set(selections) != set(modes):
            raise ValueError('Both selectors required on the same inputs')
        positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
        last = torch.where(valid, positions, -1).amax(1).clamp_min(0)
        lengths = valid.sum(1).detach().cpu()
        nonempty = valid.any(1).detach().cpu()
        self.nonempty_history_count += int(nonempty.sum())
        self.four_past += int((lengths >= 5).sum())
        self.stats['history_length'].update(lengths)
        for value, count in zip(*torch.unique(lengths, return_counts=True)):
            value = str(int(value))
            self.history_histogram[value] = self.history_histogram.get(value, 0) + int(count)
        clock = selections['time_memory']['clock_ms']
        first = torch.where(valid, positions, valid.shape[1]).amin(1).clamp_max(valid.shape[1] - 1)
        span = _last(clock, last) - _last(clock, first)
        span = torch.where(nonempty, span, 0.0)
        self.stats['temporal_span_ms'].update(span)
        self.stats['temporal_span_R0'].update(span / REFERENCE_MS)
        reached = span[:, None] / REFERENCE_MS >= torch.tensor(ANCHORS, dtype=torch.float64)
        for k in range(4):
            self.time_anchor_reached[k] += int(reached[:, k].sum())
        selected = {}
        for mode in modes:
            source = selections[mode]
            selected[mode] = {key: _last(source[key], last) for key in
                              ('indices', 'mask', 'ages', 'event_lags',
                               'anchor_mask', 'anchor_out_of_range')}
            row = selected[mode]
            if (row['mask'] & ~nonempty[:, None]).any():
                raise ValueError('Empty history has a memory slot')
            for k in range(4):
                eligible = row['anchor_mask'][:, k]
                self.anchor_available[mode][k] += int(eligible.sum())
                self.out_of_range[mode][k] += int((row['anchor_out_of_range'][:, k] & eligible).sum())
            for key, values in (('selected_ages_R0', row['ages']),
                                ('selected_ages_ms', row['ages'] * REFERENCE_MS),
                                ('selected_event_lags', row['event_lags'])):
                self.mode_stats[mode][key].update(values[row['mask']])
        nslots = selected[modes[0]]['mask'].sum(1)
        if not torch.equal(nslots, selected[modes[1]]['mask'].sum(1)):
            raise ValueError('Index/time memory slot-count drift')
        expected = (lengths - 1).clamp(min=0, max=4)
        if not torch.equal(nslots, expected):
            raise ValueError('Memory slot count does not match available past positions')
        for k in range(5):
            self.slot_histogram[k] += int((nslots == k).sum())
        if example_ids is not None and len(example_ids) != len(last):
            raise ValueError('Example-ID count drift')
        if extras is not None and len(extras) != len(last):
            raise ValueError('Extra diagnostic count drift')
        for b in range(len(last)):
            row = dict(stream_ordinal=self.count + b,
                       input_index=int(example_ids[b]) if example_ids is not None else self.count + b,
                       query_position=int(last[b]) if nonempty[b] else None,
                       history_length=int(lengths[b]), valid_slots=int(nslots[b]),
                       temporal_span_ms=float(span[b]), temporal_span_R0=float(span[b] / REFERENCE_MS))
            for mode in modes:
                current = selected[mode]
                mask = current['mask'][b]
                row[mode] = dict(indices=current['indices'][b][mask].tolist(),
                                 selected_ages_R0=current['ages'][b][mask].tolist(),
                                 selected_ages_ms=(current['ages'][b][mask] * REFERENCE_MS).tolist(),
                                 selected_event_lags=current['event_lags'][b][mask].tolist(),
                                 anchor_out_of_range=current['anchor_out_of_range'][b].tolist(),
                                 anchor_valid=current['anchor_mask'][b].tolist())
            a, bset = (set(row[mode]['indices']) for mode in modes)
            row['selected_sets_equal'] = a == bset
            self.same_sets += int(a == bset)
            row['selected_set_jaccard'] = len(a & bset) / len(a | bset) if a or bset else 1.0
            self.stats['selected_set_jaccard'].update(torch.tensor([row['selected_set_jaccard']], dtype=torch.float64))
            if extras is not None:
                row.update(extras[b])
            self.reservoir.add(self.count + b, row)
        self.count += len(last)

    def result(self):
        rows = self.reservoir.rows()
        denominator = self.count
        result = dict(scope=self.scope, query_scope='last valid query of each input example only',
                      examples=denominator, nonempty_histories=self.nonempty_history_count,
                      reservoir=dict(method='lowest SHA256 priorities of fixed stream ordinals',
                                     capacity=self.reservoir.capacity, retained=len(rows),
                                     training_rng_consumed=False,
                                     approximate_quantiles='computed from retained queries; not all-query exact quantiles'),
                      history_length_histogram=self.history_histogram,
                      valid_slot_count_histogram={str(k): value for k, value in enumerate(self.slot_histogram)},
                      at_least_four_past_events_count=self.four_past,
                      at_least_four_past_events_fraction=self.four_past / denominator if denominator else None,
                      empty_memory_count=self.slot_histogram[0],
                      empty_memory_fraction=self.slot_histogram[0] / denominator if denominator else None,
                      selected_sets_equal_count=self.same_sets,
                      selected_sets_equal_fraction=self.same_sets / denominator if denominator else None,
                      time_anchor_reach=[dict(anchor=a, windows_reaching=self.time_anchor_reached[k],
                                             fraction=self.time_anchor_reached[k] / denominator if denominator else None)
                                         for k, a in enumerate(ANCHORS)],
                      examples_reservoir=rows[:SAVED_EXAMPLES])
        for key, stat in self.stats.items():
            result[key] = stat.result(_quantile_values(rows, key))
        for mode, stats in self.mode_stats.items():
            available = sum(self.anchor_available[mode])
            outside = sum(self.out_of_range[mode])
            result[mode] = {key: stat.result(_quantile_values(rows, key, mode))
                            for key, stat in stats.items()}
            result[mode]['anchor_out_of_range'] = dict(
                definition='anchor outside min/max ages among remaining candidates at that greedy step',
                eligible_anchor_count=available, count=outside,
                fraction=outside / available if available else None,
                by_anchor=[dict(anchor=a, eligible_count=self.anchor_available[mode][k],
                                count=self.out_of_range[mode][k],
                                fraction=self.out_of_range[mode][k] / self.anchor_available[mode][k]
                                if self.anchor_available[mode][k] else None) for k, a in enumerate(ANCHORS)])
        return result


class MemoryDiagnostics:
    """Observer installed only around the normal VALID epoch by the trainer."""
    def __init__(self, mode):
        if mode not in ('index_memory', 'time_memory'):
            raise ValueError('Only memory variants have memory diagnostics')
        self.mode = mode
        self.addresses = AddressDiagnostics('existing VALID forwards; no additional dataset/model pass')
        self.stats = {key: ScalarStats() for key in
                      ('reader_weights', 'query_norm', 'memory_norm', 'output_norm',
                       'memory_to_output_norm_ratio', 'residual_to_output_norm_ratio')}
        self.beta = self.lam = None
        self.zero_output_norm = 0
        self.forward_calls = 0
        self.max_input_batch = 0
        self.max_history = 0
        self.max_reader_bank_elements = 0
        self.observed_reader_shapes = set()

    @torch.no_grad()
    def observe(self, payload):
        if payload['mode'] != self.mode:
            raise ValueError('Wrong variant supplied to VALID observer')
        beta, lam = float(payload['beta']), float(payload['lambda'])
        if not math.isfinite(beta) or not math.isfinite(lam):
            raise ValueError('Nonfinite memory gate')
        if self.beta is not None and (beta != self.beta or lam != self.lam):
            raise ValueError('Gate changed inside one VALID epoch')
        self.beta, self.lam = beta, lam
        valid, selection = payload['valid'], payload['selection']
        positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
        last = torch.where(valid, positions, -1).amax(1).clamp_min(0)
        slot_mask = _last(selection['mask'], last)
        weights = _last(payload['reader']['weights'], last)
        query = _last(payload['H'], last).double().norm(dim=-1)
        memory = _last(payload['reader']['readout'], last).double().norm(dim=-1)
        output = _last(payload['output'], last).double().norm(dim=-1)
        positive = output > 0
        self.zero_output_norm += int((~positive).sum())
        ratio = torch.zeros_like(output)
        ratio[positive] = memory[positive] / output[positive]
        values = dict(reader_weights=weights[slot_mask], query_norm=query,
                      memory_norm=memory, output_norm=output,
                      memory_to_output_norm_ratio=ratio[positive],
                      residual_to_output_norm_ratio=abs(lam) * ratio[positive])
        for key, value in values.items():
            self.stats[key].update(value)
        extras = [dict(reader_weights=weights[b][slot_mask[b]].tolist(),
                       query_norm=float(query[b]), memory_norm=float(memory[b]), output_norm=float(output[b]),
                       memory_to_output_norm_ratio=float(ratio[b]) if positive[b] else None,
                       residual_to_output_norm_ratio=abs(lam) * float(ratio[b]) if positive[b] else None)
                  for b in range(len(last))]
        other = 'index_memory' if self.mode == 'time_memory' else 'time_memory'
        selections = {self.mode: selection, other: select(payload['timestamps'], valid, other)}
        self.addresses.update(payload['timestamps'], valid, selections, extras)
        shape = tuple(int(x) for x in payload['reader']['memory_shape'])
        if shape != (valid.shape[0], valid.shape[1], 4, payload['H'].shape[-1]):
            raise ValueError('Reader gathered-bank shape drift')
        self.observed_reader_shapes.add(shape)
        self.max_reader_bank_elements = max(self.max_reader_bank_elements, math.prod(shape))
        self.max_input_batch = max(self.max_input_batch, valid.shape[0])
        self.max_history = max(self.max_history, valid.shape[1])
        self.forward_calls += 1

    def result(self):
        result = self.addresses.result()
        rows = self.addresses.reservoir.rows()
        result.update(status='MEASURED' if self.forward_calls else 'NOT_RECORDED',
                      memory_mode=self.mode, beta=self.beta, **{'lambda': self.lam},
                      negative_gate=self.lam < 0 if self.lam is not None else None,
                      forward_calls=self.forward_calls, zero_output_norm_count=self.zero_output_norm,
                      norm_ratio_definition='norm(readout) / norm(augmented output); zero denominator excluded',
                      max_input_batch=self.max_input_batch, max_history=self.max_history,
                      reader_bank_shapes=[list(shape) for shape in sorted(self.observed_reader_shapes)],
                      max_reader_bank_elements=self.max_reader_bank_elements,
                      no_persistent_representations=True,
                      interpretation='retrieval weights are not causal event importance')
        for key, stat in self.stats.items():
            result[key] = stat.result(_quantile_values(rows, key))
        return result
