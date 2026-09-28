"""Mandatory MIMO GPU leaves, persisted before failure; no recommendation loader."""
import copy
import traceback
import torch
from . import config as c
from .records import case, create, update, read, Registry, accepted_cases
from .provenance import identity, runtime, assert_upstream, imported_sources
from .numerics import compare, comparisons, finite, residual, fixture, cotangent, measure
from experiments.mamba3_three_time.fixtures import nontrivial, scalar_loss, clone_inputs, kernel_output
from experiments.mamba3_three_time.evidence import tensor_records
from experiments.mamba3_three_time.initialization import seed_all


def fresh(mode, seed=314159):
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.config import SyntheticCatalog
    cfg = Config(model=ThreeTimeMamba3Rec, config_dict=c.settings(mode, 'cuda'))
    seed_all(seed)
    model = ThreeTimeMamba3Rec(cfg, SyntheticCatalog()).cuda().eval()
    assert_upstream(model)
    return model


def histories(length, batch=2, zero=False, padded=False):
    generator = torch.Generator().manual_seed(314159)
    items = torch.randint(1, 7112, (batch, length), generator=generator)
    lengths = torch.full((batch,), length, dtype=torch.long)
    if padded and batch > 1:
        lengths[-1] = max(1, length-7)
        items[-1, lengths[-1]:] = 0
    gaps = torch.randint(0, 8, (batch, length), generator=generator).double()*300001
    if zero:
        gaps.zero_()
    timestamps = 1.6e12 + gaps.cumsum(1)
    return tuple(x.cuda() for x in (items, lengths, timestamps))


def model_measure(model, data, oracle=None):
    captured = []
    def retain(_m, _a, x):
        x.retain_grad()
        captured.append(x)
    hook = model.item_embedding.register_forward_hook(retain)
    model.zero_grad(set_to_none=True)
    seed_all(314159)
    try:
        sequence = model.encode_sequence(*data, oracle=oracle)
        output = model.gather_indexes(sequence, data[1]-1)
        loss = scalar_loss(output)
        loss.backward()
        result = dict(output=output.detach(), loss=loss.detach(), input_gradient=captured[0].grad.detach().clone())
        result.update({'gradient:' + k: v.grad.detach().clone() if v.grad is not None else None for k, v in model.named_parameters()})
        return result
    finally:
        hook.remove()


def structural(spec):
    net = fresh(spec['mode']).train(spec['training'])
    if spec['mode'] == 'dual':
        nontrivial(net.times.calibrators)
    ref = copy.deepcopy(net)
    data = histories(spec['length'], padded=True)
    checks = comparisons(model_measure(net, data), model_measure(ref, data, 'official_' + spec['mode']))
    return case(checks, fixture_seed=314159, input_tensors=tensor_records(dict(zip(('items','lengths','timestamps'), data))),
                calibrators='nonzero/nontrivial' if spec['mode'] == 'dual' else 'absent')


def boundary(spec):
    from experiments.mamba3_three_time.length_adapter import prepare
    n = spec['length']
    x = fixture(n, tied=True)
    go, meta = cotangent(x['v'].shape, 'signed', 'cuda')
    official, local = measure(x, 'official', go), measure(x, 'local', go)
    checks = comparisons(local, official)
    for backend in ('official', 'local'):
        raw = clone_inputs(x)
        padded, mapping = prepare(raw)
        output = kernel_output(padded, 'MIMO', official=backend == 'official', native=backend == 'official')
        output.retain_grad()
        output[:, :n].backward(go)
        manual = dict(output=output[:, :n].detach())
        seen = set()
        for k, v in raw.items():
            if id(v) not in seen:
                manual['gradient:' + k] = v.grad
                seen.add(id(v))
        original = official if backend == 'official' else local
        checks.update({backend + ':crop:' + k: v for k, v in comparisons(manual, {k: v for k, v in original.items() if k != 'loss'}).items()})
        checks[backend + ':no_tail_loss'] = compare(output.grad[:, n:], torch.zeros_like(output.grad[:, n:]))
    return case(checks, length_mapping=mapping, fixture_seed=314159, cotangent=meta)


def reference(spec, save):
    from experiments.mamba3_three_time.numerics_004 import loss_report
    n, seed = spec['length'], spec['seed']
    checks, evidence = {}, []
    profile = dict(atol=.003, rtol=.08, relative_norm_limit=.08)
    for tied in (True, False):
        source = fixture(n, seed, tied=tied)
        for kind in ('signed', 'nonnegative'):
            go, meta = cotangent(source['v'].shape, kind, 'cuda')
            expected = measure(source, 'reference', go)
            backends = ('official', 'local') if tied else ('local',)
            for backend in backends:
                actual = measure(source, backend, go)
                required = comparisons({k:v for k,v in actual.items() if k != 'loss'},
                                       {k:v for k,v in expected.items() if k != 'loss'}, 'output')
                prefix = f'{tied}/{kind}/{backend}/'
                checks.update({prefix+k: v for k,v in required.items()})
                evidence.append(dict(tied=tied, backend=backend, cotangent=meta, input_tensors=tensor_records(source), checks=required))
                save(case(checks, measurements=evidence, fixture_seed=seed, baseline_before_triple=True))
                if not all(v['passed'] for v in required.values()):
                    return case(checks, measurements=evidence, fixture_seed=seed, baseline_before_triple=True,
                                triple_status='INCONCLUSIVE' if tied else 'FAIL')
        old_reference = measure(source, 'reference')
        legacy = {}
        for backend in (('official', 'local') if tied else ('local',)):
            actual = measure(source, backend)
            legacy[backend] = dict(checks=comparisons(actual, old_reference, 'output'),
                                   loss_diagnostics=loss_report(actual['output'], old_reference['output'], profile))
            checks[f'{tied}/{backend}/legacy_loss_mixed'] = compare(actual['loss'], old_reference['loss'], 'output', mixed_only=True)
            for k, v in actual.items():
                checks[f'{tied}/{backend}/legacy_finite/{k}'] = finite(v)
        evidence.append(dict(tied=tied, legacy_nonlinear_loss=legacy, required=False,
                             role='Legacy norm caps remain measured, not relabelled PASS; shared VJP is mandatory'))
        save(case(checks, measurements=evidence, fixture_seed=seed, baseline_before_triple=True))
    return case(checks, measurements=evidence, fixture_seed=seed, baseline_before_triple=True)


def prefix(spec, save):
    from .prefix_checks import check_prefix
    mode, n, p, multiplier = (spec[k] for k in ('mode', 'length', 'prefix', 'multiplier'))
    net = fresh(mode)
    if mode != 'base':
        nontrivial(net.times.calibrators)
    data = histories(n)
    items, lens, times = data
    it = items.clone(); it[0,p:] = (it[0,p:] + 37) % 7111 + 1
    tm = times.clone(); tm[0,p:] += 90000000
    checks, diagnostics, phases = {}, {}, {}
    for label, oracle in [('local', None)] + ([('official', 'official_' + mode)] if mode != 'triple' else []):
        def persist(partial, phase):
            checks.update({label+':'+k: v for k,v in partial.items()})
            phases[label] = phase
            if 'residual' in partial:
                diagnostics[label] = partial['residual']
            save(dict(checks=dict(checks), required_keys=sorted(checks), residuals=diagnostics,
                      prefix_phases=phases, fixture_seed=314159, nonzero_calibrators=mode != 'base'))
        check_prefix(net, data, p, multiplier, oracle, ((it,lens,times),(items,lens,tm)), persist)
    return case(checks, residuals=diagnostics, prefix_phases=phases, fixture_seed=314159, nonzero_calibrators=mode != 'base',
                legacy_exact_zero_reclassified=False, input_tensors=tensor_records(dict(zip(('items','lengths','timestamps'),data))))


def d_oracle():
    from experiments.mamba3_three_time.numerics_004 import d_oracles, packed
    source = fixture(7, tied=True, d_only=True)
    checks, evidence = {}, []
    for kind in ('signed', 'nonnegative'):
        go, meta = cotangent(source['v'].shape, kind, 'cuda')
        oracles = d_oracles(source, go)
        expected = measure(source, 'reference', go)
        checks[kind+':reference_analytic_D'] = compare(expected['gradient:d'], oracles['ideal_fp64'], 'vjp')
        for backend in ('official', 'local'):
            actual = measure(source, backend, go)
            checks.update({kind+'/'+backend+'/'+k:v for k,v in comparisons({k:v for k,v in actual.items() if k!='loss'},
                           {k:v for k,v in expected.items() if k!='loss'}, 'output').items()})
            checks[kind+'/'+backend+'/analytic_D'] = compare(actual['gradient:d'], oracles['ideal_fp64'], 'vjp')
        evidence.append(dict(cotangent=meta, D_oracles={k:packed(v) for k,v in oracles.items()},
                             meaning='D skip weight; rounded versions are continuous/STE diagnostics, not exact discrete derivatives'))
    return case(checks, measurements=evidence)


def edges():
    from experiments.mamba3_timeaware.time_inputs import history_gaps
    from recbole.data.interaction import Interaction
    net = fresh('triple'); nontrivial(net.times.calibrators)
    checks = {}
    for label, data in [('padded', histories(17,padded=True)), ('length1', histories(1)), ('zero_gap', histories(17,zero=True))]:
        items, lengths, times = data
        net.zero_grad(set_to_none=True)
        out = net(*data); scalar_loss(out).backward()
        checks[label+':finite'] = finite(out)
        checks.update({label+':gradient:'+k:finite(v.grad) for k,v in net.named_parameters()})
        gaps, active = history_gaps(times, items != 0)
        for i, scale in enumerate(net.times(gaps,active)):
            checks[f'{label}:neutral{i}'] = compare(scale[~active], torch.ones_like(scale[~active]))
        if label == 'zero_gap':
            checks['real_zero_gap_active'] = dict(passed=bool(active[:,1:].all()) and not bool(active[:,0].any()) and bool((gaps[active]==0).all()))
        if label == 'padded':
            altered = times.clone(); altered[items==0] = float('nan')
            with torch.no_grad():
                checks['padding_timestamp_ignored'] = compare(out, net(items,lengths,altered))
                interaction = Interaction({net.ITEM_SEQ:items, net.ITEM_SEQ_LEN:lengths,
                    net.time_sequence_field:times, 'timestamp':times[:,-1]+1e9})
                scores = net.full_sort_predict(interaction)
                interaction['timestamp'] += 2e9
                checks['target_timestamp_ignored'] = compare(scores, net.full_sort_predict(interaction))
    return case(checks)


def dispatch(spec, save):
    suite = spec['suite']
    if suite == 'initialization':
        from .state import initialization
        return case({'initialization':initialization('cuda', save)}, runtime=runtime(True))
    if suite == 'native':
        values = measure(fixture(16,tied=True), 'official', native=True)
        return case({k:finite(v) for k,v in values.items()}, native=True, rank=4, chunk=8)
    if suite == 'boundary':
        return boundary(spec)
    if suite == 'structural':
        return structural(spec)
    if suite == 'reference':
        return reference(spec, save)
    if suite == 'prefix':
        return prefix(spec, save)
    if suite == 'D_oracle':
        return d_oracle()
    if suite == 'edges':
        return edges()
    from experiments.mamba3_three_time import suites
    if suite in ('tied_slots', 'tied_calibrators'):
        rows = suites.triple_recovery('MIMO')
        row = next(rows) if suite == 'tied_slots' else list(rows)[1]
    elif suite == 'optimizer':
        row = next(r for r in suites.optimizer_steps('MIMO') if r['name'].endswith(spec['mode']))
    elif suite == 'roundtrip':
        row = suites.roundtrip('MIMO')
    else:
        raise ValueError(suite)
    return case(row['checks'], historical_fixture_seed=2026, historical_suite=row)


def main():
    base = identity()
    result = dict(**base, status='RUNNING', scientific_fits=0, cases=[], training_authorized=False)
    create(c.GATE, result)
    registry = Registry(c.GATE, result, c.plan()['required_cases'])
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(0)
        torch.empty(1,device='cuda').zero_()
        assert_upstream()
        for spec in c.plan()['required_cases']:
            registry.run(spec, lambda save, spec=spec: dispatch(spec, save))
        if not accepted_cases(result['cases'], c.plan()['required_cases']):
            raise ValueError('Incomplete required admission evidence')
        result.update(status=c.ACCEPTED, training_authorized=True, imported_sources=imported_sources())
    except BaseException:
        result.update(status='FAIL', traceback=traceback.format_exc(), training_authorized=False)
    update(c.GATE, result)
    return 0 if result['status'] == c.ACCEPTED else 1


if __name__ == '__main__':
    raise SystemExit(main())
