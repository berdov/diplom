"""Finite synthetic SISO admission; never loads recommendation data or trains a fit."""
import copy
import json
import os
import traceback
import torch
from recbole.config import Config
from experiments.mamba3_three_time import suites
from experiments.mamba3_three_time.config import SyntheticCatalog, settings as mathematical_settings
from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
from experiments.mamba3_three_time.fixtures import nontrivial, scalar_loss, kernel_inputs, clone_inputs, kernel_output
from experiments.mamba3_three_time.diagnostics import frozen_separate
from experiments.mamba3_three_time.diagnostics_004 import full_output
from experiments.mamba3_three_time.evidence import case, compare, tensor_records
from experiments.mamba3_three_time.records_003 import Registry
from experiments.mamba3_three_time.initialization import seed_all
from experiments.mamba3_three_time.numerics_004 import cotangent
from experiments.mamba3_timeaware.time_inputs import history_gaps
from .config import GATE, PARENT, ACCEPTED, plan
from .policy import positional
from .provenance import identity, create_record, atomic_json, assert_upstream, backend_guard
from .preflight import run as preflight, initialization


def expected_ids():
    names = ['source_data_runtime', 'initialization']
    names += [f'{s}_{m}_L{n}' for s in ('A', 'B') for m in ('eval', 'train') for n in (50, 64)]
    names += ['C_leaf_gradient_sum', 'C_tied_calibrator_recovery']
    names += [f'VJP_L{n}' for n in (7, 50)]
    names += [f'prefix_{m}_L{n}_P{p}_x{x}' for m in ('base', 'dual', 'triple')
              for n, p in plan()['gate']['new_prefix_fixtures'] for x in (1, 16)]
    return names + ['padding_length1_zero_gap', 'H_state_dict_roundtrip']


def fresh(mode, seed=314159, device='cuda'):
    values = mathematical_settings('SISO', mode)
    values.update(use_gpu=device == 'cuda', device=device)
    cfg = Config(model=ThreeTimeMamba3Rec, config_dict=values)
    seed_all(seed)
    net = ThreeTimeMamba3Rec(cfg, SyntheticCatalog()).to(device).eval()
    backend_guard(net)
    return net


def histories(length, batch=2, seed=314159, device='cuda', padded=False, zero_gap=False):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    items = torch.randint(1, 7112, (batch, length), generator=gen)
    lengths = torch.full((batch,), length, dtype=torch.long)
    if padded and batch > 1:
        lengths[-1] = max(1, length - 7)
        items[-1, lengths[-1]:] = 0
    gaps = torch.randint(0, 8, (batch, length), generator=gen).double() * 300001
    if zero_gap:
        gaps.zero_()
    times = 1.6e12 + gaps.cumsum(1)
    return tuple(x.to(device) for x in (items, lengths, times))


def prefix_measure(net, data, prefix, multiplier, oracle=None, old=False):
    captured = []
    def capture(_m, _args, value):
        value.retain_grad()
        captured.append(value)
    hook = net.item_embedding.register_forward_hook(capture)
    net.zero_grad(set_to_none=True)
    seed_all(314159)
    try:
        assert_upstream()
        y = full_output(net, data, oracle, old)
        loss = scalar_loss(y[0, :prefix]) * multiplier
        loss.backward()
        g = captured[0].grad.detach()
        return y.detach(), g.clone(), positional(g[:1], prefix)
    finally:
        hook.remove()


def prefix_case(mode, length, prefix, multiplier):
    net = fresh(mode)
    if mode != 'base':
        nontrivial(net.times.calibrators)
    data = histories(length)
    items, lens, times = data
    changes = []
    it = items.clone(); it[0, prefix:] = (it[0, prefix:] + 37) % 7111 + 1
    tm = times.clone(); tm[0, prefix:] += 90000000
    changes.extend(((it, lens, times), (items, lens, tm)))
    targets = [('local_' + mode, net, None, False)]
    if mode == 'base':
        targets.append(('official_base', copy.deepcopy(net), 'official_base', False))
    elif mode == 'dual':
        targets.append(('frozen_separate', frozen_separate(net), None, True))
    checks, records, measured = {}, {}, {}
    for name, target, oracle, old in targets:
        y, g, stats = prefix_measure(target, data, prefix, multiplier, oracle, old)
        measured[name] = (y, g)
        records[name] = stats
        checks[name + ':residual_v1'] = dict(passed=stats['finite_precision_pass'], informative=stats['informative'])
        checks[name + ':finite'] = dict(passed=bool(torch.isfinite(y).all() and torch.isfinite(g).all()))
        checks[name + ':cross_user_gradient'] = compare(g[1:], torch.zeros_like(g[1:]), atol=0, rtol=0)
        with torch.no_grad():
            for index, changed in enumerate(changes):
                altered = full_output(target, changed, oracle, old)
                checks[f'{name}:prefix_intervention{index}'] = compare(y[:, :prefix], altered[:, :prefix])
                checks[f'{name}:cross_user{index}'] = compare(y[1], altered[1])
    if len(targets) == 2:
        (y, g), (ry, rg) = measured.values()
        checks.update(output_parity=compare(y, ry), positional_gradient_parity=compare(g, rg))
    row = case('prefix', 'positional gradients, fresh seed314159 items/gaps/backbone', checks)
    row.update(seed=314159, mode=mode, length=length, prefix=prefix, loss_multiplier=multiplier,
               input_tensors=tensor_records(dict(zip(('items', 'lengths', 'timestamps'), data))),
               residuals=records, legacy_exact_zero_reclassified=False,
               temporal_corrections='nonzero' if mode != 'base' else 'official base identity')
    return row


def measure_kernel(source, backend, go):
    values = clone_inputs(source, float_reference=backend == 'reference')
    y = kernel_output(values, 'SISO', official=backend == 'official', reference=backend == 'reference')
    unique = {}; seen = set()
    for k, v in values.items():
        if id(v) not in seen:
            unique[k] = v; seen.add(id(v))
    grads = torch.autograd.grad(y, tuple(unique.values()), grad_outputs=go.to(y.dtype), allow_unused=True)
    out = dict(output=y.detach())
    out.update({'gradient:' + k: g.detach() if g is not None else None for k, g in zip(unique, grads)})
    return out


def vjp_case(length):
    profile = json.loads((PARENT / 'test_plan_004.json').read_text())['reference']
    checks, evidence = {}, []
    for tied in (True, False):
        source = kernel_inputs('SISO', length=length, tied=tied)
        if not tied:
            with torch.no_grad():
                variation = torch.linspace(.7, 1.3, source['dw'].numel(), device='cuda').reshape_as(source['dw'])
                source['adt'].mul_(variation); source['dw'].mul_(variation.flip(-1)); source['dp'].mul_(1.8-variation)
        for kind in ('signed', 'nonnegative'):
            go, meta = cotangent(source['v'].shape, kind, 'cuda')
            reference = measure_kernel(source, 'reference', go)
            backends = ('official', 'local') if tied else ('local',)
            measured = {}
            for backend in backends:
                values = measure_kernel(source, backend, go)
                measured[backend] = values
                comparisons = suites.comparisons(values, reference, profile)
                checks.update({f'{tied}/{kind}/{backend}/{k}': v for k, v in comparisons.items()})
                evidence.append(dict(tied=tied, cotangent=meta, backend=backend,
                    input_tensors=tensor_records(source), checks=comparisons))
                if tied and backend == 'official' and not all(v['passed'] for v in comparisons.values()):
                    return dict(case('VJP', 'official reference calibration before triple', checks), evidence=evidence)
            if tied:
                checks.update({f'parity/{kind}/{k}': v for k, v in suites.comparisons(measured['local'], measured['official']).items()})
    return dict(case('VJP', 'common output-independent cotangents and same quantized inputs', checks), evidence=evidence)


def edges():
    from recbole.data.interaction import Interaction
    checks = {}
    net = fresh('triple'); nontrivial(net.times.calibrators)
    data = histories(17, padded=True)
    items, lens, times = data
    with torch.no_grad():
        y = net.encode_sequence(*data)
        changed = times.clone(); changed[items == 0] += 1e12
        yy = net.encode_sequence(items, lens, changed)
        checks['padding_time_ignored'] = compare(y[items != 0], yy[items != 0])
        interaction = Interaction({net.ITEM_SEQ: items, net.ITEM_SEQ_LEN: lens,
                                   net.time_sequence_field: times, 'timestamp': times[:, -1] + 1e9})
        a = net.full_sort_predict(interaction)
        interaction['timestamp'] += 2e9
        checks['target_timestamp_not_input'] = compare(a, net.full_sort_predict(interaction), atol=0, rtol=0)
    for label, data in [('length1', histories(1)), ('zero_gap', histories(17, zero_gap=True))]:
        net.zero_grad(set_to_none=True)
        y = net(*data); loss = scalar_loss(y); loss.backward()
        checks[label + ':finite'] = dict(passed=bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters()))
        gaps, active = history_gaps(data[2], data[0] != 0)
        checks[label + ':actual_gaps'] = dict(passed=not bool(active.any()) if label == 'length1' else
                                            bool(active.any()) and bool((gaps[active] == 0).all()))
    return case('edges', 'actual padded timestamps, length1, equal timestamps, no target input', checks)


def main():
    base = identity()
    result = dict(**base, status='RUNNING', cases=[], scientific_fits=0,
                  training_authorized=False, MIMO_training_authorized=False)
    create_record(GATE, result)
    registry = Registry(result, lambda: atomic_json(GATE, result), expected_ids())
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(0)
        torch.empty(1, device='cuda').zero_()
        def prerequisites():
            value = preflight(full_data=True)
            return dict(case('preflight', 'frozen data/runtime/raw evidence', {'verified': dict(passed=True)}), evidence=value)
        registry.run('source_data_runtime', prerequisites, gate=True)
        registry.run('initialization', lambda: case('initialization', 'scratch dual/triple', {'initialization': initialization('cuda')}), gate=True)
        for factory in (suites.official_parity, suites.dual_recovery, suites.triple_recovery):
            for row in factory('SISO'):
                registry.run(row['name'], lambda row=row: row, gate=True)
        for n in (7, 50):
            registry.run(f'VJP_L{n}', lambda n=n: vjp_case(n), gate=True)
        for mode in ('base', 'dual', 'triple'):
            for n, p in plan()['gate']['new_prefix_fixtures']:
                for x in (1, 16):
                    registry.run(f'prefix_{mode}_L{n}_P{p}_x{x}', lambda: prefix_case(mode, n, p, x), gate=True)
        registry.run('padding_length1_zero_gap', edges, gate=True)
        registry.run('H_state_dict_roundtrip', lambda: suites.roundtrip('SISO'), gate=True)
        if not registry.close():
            raise ValueError('Missing/failed required admission checks')
        result.update(status=ACCEPTED, training_authorized=True,
                      authorization_scope='Exactly two SISO fits in this pilot/allocation; not historical correctness')
    except BaseException:
        result.update(status='ADMISSION_FAIL', traceback=traceback.format_exc(), training_authorized=False)
        traceback.print_exc()
    finally:
        registry.close()
        atomic_json(GATE, result)
    return 0 if result['status'] == ACCEPTED else 1


if __name__ == '__main__':
    from experiments.mamba3_three_time.backends import selected
    with selected('upstream'):
        raise SystemExit(main())
