"""Synthetic admission fixtures; no recommendation loader or scientific fit."""
import io
import torch
from . import config as c
from .memory import select, read_memory
from .model import TimeMemoryMamba3Rec, transfer_common
from experiments.mamba3_three_time.initialization import seed_all
from experiments.mamba3_three_time.confirmation.state import difference, rng_record
from experiments.mamba3_mimo_time.records import case
from experiments.mamba3_mimo_time.admission import histories, model_measure
from experiments.mamba3_mimo_time.provenance import assert_upstream
from experiments.mamba3_three_time.evidence import tensor_records
from experiments.mamba3_head_timescales.checks import nonzero


SELECTOR_KEYS = [
    'cpu_cuda_selection', 'deterministic_ties', 'uniform_clock', 'nonuniform_clock',
    'causal_unique_slots', 'variable_lengths', 'padding_ignored', 'sentinel_safe',
    'empty_backward', 'dtype_device', 'weights_normalized', 'query_gradient',
    'selected_value_gradient', 'beta_gradient_at_zero', 'signed_beta',
    'index_clock_unchanged', 'time_clock_changes', 'same_set_output_gradient',
    'bank_shape', 'zero_age', 'fixed_h_clock_effect',
]
EDGE_KEYS = ['parameter_count', 'roundtrip', 'length1_backward', 'incomplete_batch',
             'batch_permutation', 'no_cross_batch_memory', 'no_persistent_bank',
             'padding_neutral', 'target_timestamp_ignored', 'hook_cleanup']
CAUSAL_KEYS = ['residual', 'cross_user_gradient', 'finite_output',
               'prefix_intervention0', 'cross_user_intervention0',
               'prefix_intervention1', 'cross_user_intervention1',
               'selected_prefix_item', 'selected_prefix_timestamp',
               'selected_cross_user', 'input_interventions_real', 'hook_lifecycle']
NEGATIVE_KEYS = ['leaking_detected', 'detached_leaking_detected', 'hook_cleanup']


def required_cases(common_parameter_keys):
    common = list(common_parameter_keys)
    parity = ['output', 'loss', 'input_gradient', 'dropout_rng', 'initial_output_bitwise',
              'common_state_exact', 'parameter_count', 'hook_cleanup', 'adam_json_roundtrip']
    parity += ['gradient:' + key for key in common]
    parity += ['first_adam:' + key for key in common]
    specs = [dict(id='no_memory_historical', required_keys=sorted(parity))]
    for mode in ('index_memory', 'time_memory'):
        specs.append(dict(id=mode + '_beta0', required_keys=sorted(parity + ['beta_zero', 'beta_gradient_finite'])))
    specs.append(dict(id='selector_reader', required_keys=sorted(SELECTOR_KEYS)))
    specs += [dict(id=mode + '_edges', required_keys=sorted(EDGE_KEYS)) for mode in ('index_memory', 'time_memory')]
    specs += [dict(id=f'{mode}_prefix{prefix}', required_keys=sorted(CAUSAL_KEYS))
              for mode in ('index_memory', 'time_memory') for prefix in (8, 9)]
    specs.append(dict(id='negative_controls', required_keys=sorted(NEGATIVE_KEYS)))
    return specs


def fresh(mode, device='cpu', historical=False):
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.config import SyntheticCatalog
    cfg = Config(model=ThreeTimeMamba3Rec, config_dict=c.settings(mode, device))
    seed_all(314159)
    cls = ThreeTimeMamba3Rec if historical else TimeMemoryMamba3Rec
    return cls(cfg, SyntheticCatalog()).to(device).eval()


def emit_to(save, checks):
    def emit(key, value):
        checks[key] = value
        save(case(dict(checks)))
    return emit


def ok(condition, **evidence):
    return dict(passed=bool(condition), **evidence)


def parity(mode, save):
    import json
    from experiments.mamba3_three_time.confirmation.state import compare_optimizer_settings
    reference = fresh('no_memory', 'cuda', historical=mode == 'no_memory')
    for cal in reference.times.calibrators.values():
        nonzero(cal)
    net = fresh(mode, 'cuda')
    transfer_common(reference, net)
    mapping = dict(common_keys=sorted(reference.state_dict()), beta='Target scalar remains exactly zero when present')
    assert_upstream(reference); assert_upstream(net)
    common = c.plan()['common_parameter_keys']
    checks = {}; emit = emit_to(save, checks)
    emit('common_state_exact', ok(all(torch.equal(value, net.state_dict()[key]) for key, value in reference.state_dict().items())))
    emit('parameter_count', ok(sum(p.numel() for p in net.parameters()) == 715020 + (mode != 'no_memory')))
    if mode != 'no_memory':
        emit('beta_zero', ok(net.beta.shape == torch.Size([]) and net.beta.dtype == torch.float32 and net.beta.item() == 0.))
    reference.train(); net.train(); data = histories(50, batch=3, padded=True)
    left = model_measure(reference, data); left_rng = rng_record()
    right = model_measure(net, data); right_rng = rng_record()
    for key in ('output', 'loss', 'input_gradient'):
        emit(key, difference(right.get(key), left.get(key)))
    for key in common:
        emit('gradient:' + key, difference(right.get('gradient:' + key), left.get('gradient:' + key)))
    emit('dropout_rng', ok(left_rng == right_rng))
    emit('initial_output_bitwise', ok(checks['output']['bitwise_equal'] and checks['loss']['bitwise_equal']))
    if mode != 'no_memory':
        emit('beta_gradient_finite', ok(net.beta.grad is not None and torch.isfinite(net.beta.grad)))
    a = torch.optim.Adam(reference.parameters(), lr=.001)
    b = torch.optim.Adam(net.parameters(), lr=.001)
    settings = lambda optimizer: [{k: v for k, v in row.items() if k != 'params'} for row in optimizer.param_groups]
    serialized = json.loads(json.dumps(settings(a), allow_nan=False))
    compare_optimizer_settings(serialized, settings(b))
    emit('adam_json_roundtrip', ok(True, settings=serialized))
    a.step(); b.step()
    for key in common:
        emit('first_adam:' + key, difference(dict(net.named_parameters())[key], dict(reference.named_parameters())[key]))
    emit('hook_cleanup', ok(not reference.item_embedding._forward_hooks and not net.item_embedding._forward_hooks))
    return case(checks, state_mapping=mapping, fixture_seed=314159,
                first_adam_scope='Common parameters only; no post-update output identity is required',
                input_tensors=tensor_records(dict(zip(('items', 'lengths', 'timestamps'), data))))


def addressing_fixture(device):
    # Last-query physical ages are [64,32,16,8,4,2,1,0] R0.
    # Ordinal sorted set [0,1,3,6]; elapsed-time sorted set [1,2,4,6].
    relative = torch.tensor([0,32,48,56,60,62,63,64], dtype=torch.float64, device=device)
    times = 1.6e12 + relative[None] * 838393.
    return times, torch.ones((1, 8), dtype=torch.bool, device=device)


def selector_reader(save, device='cuda'):
    checks = {}; emit = emit_to(save, checks)
    times, valid = addressing_fixture(device)
    index, timed = (select(times, valid, mode) for mode in ('index_memory', 'time_memory'))
    cpu = select(times.cpu(), valid.cpu(), 'time_memory')
    emit('cpu_cuda_selection', ok(torch.equal(cpu['indices'], timed['indices'].cpu()) and torch.equal(cpu['mask'], timed['mask'].cpu())))
    emit('nonuniform_clock', ok(index['indices'][0,-1].tolist() == [0,1,3,6] and timed['indices'][0,-1].tolist() == [1,2,4,6]))
    uniform = 1.6e12 + torch.arange(8, device=device, dtype=torch.float64)[None] * 838393.
    ui, ut = (select(uniform, valid, mode) for mode in ('index_memory', 'time_memory'))
    emit('uniform_clock', ok(torch.equal(ui['indices'], ut['indices']) and torch.equal(ui['mask'], ut['mask'])))
    zero = select(torch.full_like(times, 1.6e12), valid, 'time_memory')
    emit('deterministic_ties', ok(zero['anchor_indices'][0,-1].tolist() == [6,5,4,3] and zero['indices'][0,-1].tolist() == [3,4,5,6]))
    emit('zero_age', ok(torch.equal(zero['ages'][zero['mask']], torch.zeros_like(zero['ages'][zero['mask']]))))
    positions = torch.arange(8, device=device)[None,:,None]
    causal = all(bool(((s['indices'] < positions) | ~s['mask']).all()) and
                 all(len(set(row[mask].tolist())) == int(mask.sum()) for row, mask in zip(s['indices'].reshape(-1,4), s['mask'].reshape(-1,4)))
                 for s in (index, timed))
    emit('causal_unique_slots', ok(causal))
    lengths = torch.tensor([1,2,3,4,8], device=device)
    vm = torch.arange(8, device=device)[None] < lengths[:,None]
    ts = times.expand(5,-1).clone(); ps = select(ts, vm, 'time_memory')
    expected = torch.arange(8, device=device).clamp(max=4)[None].expand(5,-1) * vm
    emit('variable_lengths', ok(torch.equal(ps['mask'].sum(-1), expected), batch=5, lengths=lengths.tolist()))
    ts[~vm] = -8.7e17; changed = select(ts, vm, 'time_memory')
    emit('padding_ignored', ok(torch.equal(ps['indices'], changed['indices']) and torch.equal(ps['mask'], changed['mask'])))
    h = (torch.arange(8*64, device=device, dtype=torch.float32).reshape(1,8,64) / 1000. + .1).requires_grad_()
    beta = torch.tensor(.3, dtype=torch.float32, device=device, requires_grad=True)
    out, details = read_memory(h, timed, beta)
    details['values'].retain_grad()
    (out[:,-1] * torch.linspace(.2,1.2,64,device=device)).sum().backward()
    emit('selected_value_gradient', ok(details['values'].grad is not None and torch.isfinite(details['values'].grad).all() and details['values'].grad[0,-1].abs().sum() > 0))
    emit('query_gradient', ok(h.grad is not None and torch.isfinite(h.grad).all() and h.grad[0,-1].abs().sum() > 0))
    emit('weights_normalized', ok(torch.isfinite(details['weights']).all() and torch.allclose(details['weights'].sum(-1), timed['mask'].any(-1).float(), atol=1e-6, rtol=1e-5) and (details['weights'][~timed['mask']] == 0).all()))
    emit('bank_shape', ok(list(details['values'].shape) == [1,8,4,64], actual=list(details['values'].shape)))
    emit('sentinel_safe', ok((details['values'][~timed['mask']] == 0).all() and torch.equal(out[:,0], h[:,0])))
    emit('dtype_device', ok(out.dtype == torch.float32 and details['weights'].dtype == torch.float32 and out.device == h.device and timed['ages'].dtype == torch.float64))
    b0 = torch.tensor(0., device=device, requires_grad=True)
    o0, _ = read_memory(h.detach(), timed, b0); o0[:,-1].sum().backward()
    emit('beta_gradient_at_zero', ok(torch.equal(o0, h.detach()) and b0.grad is not None and torch.isfinite(b0.grad) and b0.grad.abs() > 0, gradient=float(b0.grad)))
    optimizer = torch.optim.Adam([b0], lr=.001); optimizer.step()
    emit('signed_beta', ok(b0.item() < 0, beta=b0.item(), lambda_value=torch.tanh(b0).item()))
    empty_h = torch.ones((2,1,64), device=device, requires_grad=True)
    empty_s = select(torch.ones((2,1),device=device,dtype=torch.float64), torch.ones((2,1),device=device,dtype=torch.bool), 'time_memory')
    empty_b = torch.tensor(.3, device=device, requires_grad=True)
    empty_o, empty_d = read_memory(empty_h, empty_s, empty_b); empty_o.sum().backward()
    emit('empty_backward', ok(torch.equal(empty_o, empty_h) and (empty_d['readout'] == 0).all() and (empty_d['weights'] == 0).all() and torch.isfinite(empty_h.grad).all() and empty_b.grad is not None and torch.isfinite(empty_b.grad)))
    emit('index_clock_unchanged', ok(torch.equal(index['indices'], ui['indices'])))
    emit('time_clock_changes', ok(not torch.equal(timed['indices'], ut['indices'])))
    def measure(selection):
        value = h.detach().clone().requires_grad_(); gate = torch.tensor(.3,device=device,requires_grad=True)
        result, _ = read_memory(value, selection, gate)
        result.sum().backward(); return result.detach(), value.grad, gate.grad
    a, b = measure(ui), measure(ut)
    emit('same_set_output_gradient', ok(all(difference(x,y)['passed'] for x,y in zip(a,b))))
    oi, _ = read_memory(h.detach(), index, beta.detach()); ot, _ = read_memory(h.detach(), timed, beta.detach())
    emit('fixed_h_clock_effect', ok(not torch.equal(oi[:,-1], ot[:,-1])))
    return case(checks, input_tensors=tensor_records(dict(timestamps=times, valid=valid)),
                manual_last_indices=dict(index=[0,1,3,6], time=[1,2,4,6]))


def model_edges(mode, save):
    from recbole.data.interaction import Interaction
    checks = {}; emit = emit_to(save, checks)
    net = fresh(mode, 'cuda'); assert_upstream(net)
    with torch.no_grad(): net.beta.fill_(.3)
    initial_attributes = set(vars(net)); initial_buffers = set(dict(net.named_buffers()))
    data = histories(50, batch=3, padded=True)
    data[1][0] = 1; data[0][0,1:] = 0
    data[2][0,1:] = -5e14
    with torch.no_grad():
        out = net.encode_sequence(*data)
        order = torch.tensor([2,0,1],device='cuda')
        permuted = net.encode_sequence(*(x[order] for x in data))
        net(*histories(17, batch=2, zero=True))
        repeated = net.encode_sequence(*data)
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        raw = ThreeTimeMamba3Rec.encode_sequence(net, *data)
    emit('parameter_count', ok(sum(p.numel() for p in net.parameters()) == 715021))
    emit('incomplete_batch', ok(list(out.shape) == [3,50,64] and torch.isfinite(out).all(), actual_shape=list(out.shape)))
    emit('batch_permutation', difference(permuted, out[order]))
    emit('no_cross_batch_memory', difference(repeated, out))
    emit('no_persistent_bank', ok(set(vars(net)) == initial_attributes and set(dict(net.named_buffers())) == initial_buffers and not any('memory' in key for key in initial_buffers)))
    emit('padding_neutral', difference(out[data[0] == 0], raw[data[0] == 0]))
    one = histories(1, batch=3); net.zero_grad(set_to_none=True)
    net(*one).sum().backward()
    emit('length1_backward', ok(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())))
    stream = io.BytesIO(); torch.save(net.state_dict(),stream); stream.seek(0)
    saved = torch.load(stream,map_location='cuda',weights_only=True)
    net.load_state_dict(saved,strict=True)
    with torch.no_grad(): recovered = net.encode_sequence(*data)
    emit('roundtrip', difference(recovered,out))
    interaction = Interaction({net.ITEM_SEQ:data[0], net.ITEM_SEQ_LEN:data[1], net.time_sequence_field:data[2],
                               c.settings(mode)['TIME_FIELD']:torch.ones(3,device='cuda')})
    with torch.no_grad():
        before = net._encode(interaction); interaction[c.settings(mode)['TIME_FIELD']].fill_(9e14)
        after = net._encode(interaction)
    emit('target_timestamp_ignored', difference(after,before))
    emit('hook_cleanup', ok(not net.output_norm._forward_hooks and not net.item_embedding._forward_hooks))
    return case(checks)


def causal(mode, prefix, save):
    from experiments.mamba3_mimo_time.prefix_checks import check_prefix
    from experiments.mamba3_head_timescales.progress import progress_callback, snapshot
    net = fresh(mode, 'cuda')
    for cal in net.times.calibrators.values(): nonzero(cal)
    with torch.no_grad(): net.beta.fill_(.3)
    data = histories(50, batch=3)
    items, lens, times = data
    changed_items = items.clone(); changed_items[0,prefix:] = (changed_items[0,prefix:] + 37) % 7111 + 1
    changed_times = times.clone(); changed_times[0,prefix:] += 900000000000.
    interventions = ((changed_items,lens,times),(items,lens,changed_times))
    latest = {}; persist = progress_callback(save,latest)
    checks = check_prefix(net,data,prefix,1.,None,interventions,persist)
    original = select(times,items != 0,mode)
    selections = [select(row[2],row[0] != 0,mode) for row in interventions]
    for label, selected in zip(('item','timestamp'),selections):
        checks['selected_prefix_' + label] = ok(all(torch.equal(selected[key][:,:prefix],original[key][:,:prefix]) for key in ('indices','mask')))
    checks['selected_cross_user'] = ok(all(all(torch.equal(selected[key][1:],original[key][1:]) for key in ('indices','mask')) for selected in selections))
    inputs = tensor_records(dict(items=items,timestamps=times,changed_items=changed_items,changed_timestamps=changed_times))
    checks['input_interventions_real'] = ok(not torch.equal(items,changed_items) and not torch.equal(times,changed_times) and torch.equal(items[:,:prefix],changed_items[:,:prefix]) and torch.equal(times[:,:prefix],changed_times[:,:prefix]))
    phase = latest['phase']
    checks['hook_lifecycle'] = ok(phase['hook_removed'] and phase['capture_count'] == 1 and not net.item_embedding._forward_hooks and all(not s['grad_enabled'] for s in phase['stages'] if s['stage'].startswith('intervention')))
    return snapshot(case(checks,hook_phase=phase,input_tensors=inputs,selection_tensors=tensor_records(dict(original=original['indices'],item=selections[0]['indices'],timestamp=selections[1]['indices'])),legacy_exact_zero_reclassified=False))


class LeakingFixture(torch.nn.Module):
    """Deliberately wrong; ensures interventions catch detached future leakage."""
    def __init__(self, detached=False, device='cpu'):
        super().__init__(); self.item_embedding = torch.nn.Embedding(80,8).to(device)
        self.detached = detached
        with torch.no_grad(): self.item_embedding.weight.copy_(torch.arange(640,device=device).reshape(80,8).float()/640.)

    def encode_sequence(self, items, lengths, timestamps, *, oracle=None):
        hidden = self.item_embedding(items)
        future = hidden[:,-1:] + timestamps[:,-1:,None].float() / 100.
        if self.detached: future = future.detach()
        return hidden + future


def negative_controls(save, device='cuda'):
    from experiments.mamba3_mimo_time.prefix_checks import check_prefix
    from experiments.mamba3_head_timescales.progress import progress_callback
    checks = {}; details = {}; clean = True
    for detached in (False,True):
        net = LeakingFixture(detached,device)
        items = torch.arange(18,device=device).reshape(2,9) + 1
        lengths = torch.full((2,),9,device=device,dtype=torch.long)
        times = torch.arange(18,device=device,dtype=torch.float64).reshape(2,9)
        changed_items = items.clone(); changed_items[0,4:] += 23
        changed_times = times.clone(); changed_times[0,4:] += 1000
        latest = {}
        # Production callback receives the real prefix check's initially empty progress.
        captured = []
        def save_progress(value): captured.append(value)
        raw = check_prefix(net,(items,lengths,times),4,1.,None,
                           ((changed_items,lengths,times),(items,lengths,changed_times)),
                           progress_callback(save_progress,latest))
        key = 'detached_leaking_detected' if detached else 'leaking_detected'
        checks[key] = ok(not raw['prefix_intervention0']['passed'] and not raw['prefix_intervention1']['passed'],
                         positional_residual_detected=not raw['residual']['passed'],
                         first_progress_empty=not captured[0]['checks'])
        clean &= not net.item_embedding._forward_hooks and latest['phase']['hook_removed']
        details[key] = dict(observed_checks=raw,hook_phase=latest['phase'],
                            input_tensors=tensor_records(dict(items=items,timestamps=times,
                                changed_items=changed_items,changed_timestamps=changed_times)))
        save(case(dict(checks),negative_evidence=details))
    checks['hook_cleanup'] = ok(clean)
    return case(checks,negative_evidence=details)
