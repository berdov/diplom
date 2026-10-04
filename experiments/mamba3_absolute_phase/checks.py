"""Fixed synthetic phase admission; no recommendation data or scientific fit."""
import io
import json
import math
import torch
from experiments.mamba3_mimo_time.records import case
from experiments.mamba3_mimo_time.numerics import compare, finite
from experiments.mamba3_head_timescales.progress import progress_callback, snapshot
from experiments.mamba3_three_time.initialization import seed_all
from experiments.mamba3_three_time.confirmation.state import difference, rng_record
from experiments.mamba3_three_time.evidence import tensor_records

SYNTHETIC_SEED = 314159
MODES = ('baseline_dual', 'relative_phase', 'absolute_phase')
MODIFIED = MODES[1:]
LENGTHS = (7, 8, 9, 17, 50, 65)
OPERATION_KEYS = [
    'fp64_feature_reference', 'fp64_correction_reference', 'gradcheck',
    'cpu_device_agreement', 'feature_shape_dtype', 'first_padding_neutral',
    'real_zero_gap_active', 'valid_nonfinite_rejected', 'padding_sanitized',
    'large_timestamp_finite', 'relative_origin_and_gaps', 'period6h', 'period24h',
    'period_boundary_continuity', 'frequency6h_gradient', 'frequency24h_gradient',
    'bf16_rounding_measured', 'zero_init_rng',
]
SHIFT_KEYS = [
    'unchanged_gaps_2h', 'features_2h', 'correction_2h', 'output_2h',
    'features_24h', 'correction_24h', 'output_24h', 'zero_W_output_2h',
    'zero_W_output_24h', 'first_padding_neutral', 'same_mixer_nonphase_exact',
    'shared_heads', 'native_phase_finite', 'W_gradient_informative', 'observer_output_parity', 'hook_cleanup',
]
EDGE_KEYS = [
    'parameter_count', 'roundtrip', 'length1_backward', 'incomplete_batch',
    'batch_permutation', 'no_cross_batch_state', 'padding_neutral',
    'target_timestamp_absent', 'target_timestamp_ignored', 'hook_cleanup',
] + [f'length{n}:{part}' for n in LENGTHS for part in ('output', 'backward')]
CAUSAL_KEYS = [
    'residual', 'cross_user_gradient', 'finite_output',
    'prefix_intervention0', 'cross_user_intervention0',
    'prefix_intervention1', 'cross_user_intervention1',
    'features_prefix_item', 'features_prefix_timestamp', 'features_cross_user',
    'input_interventions_real', 'hook_lifecycle',
]
REFERENCE_KEYS = ['output', 'W_gradient', 'W_gradient_informative', 'native_phase_finite']
NEGATIVE_KEYS = ['leaking_detected', 'detached_leaking_detected', 'hook_cleanup']


def required_cases(common_parameter_keys):
    parity = ['output', 'loss', 'input_gradient', 'dropout_rng', 'initial_rng',
              'initial_output_bitwise', 'common_state_exact', 'parameter_count',
              'hook_cleanup', 'adam_json_roundtrip', 'historical_mode_absent']
    parity += ['gradient:' + key for key in common_parameter_keys]
    parity += ['first_adam:' + key for key in common_parameter_keys]
    result = [dict(id=mode + '_W0', required_keys=sorted(parity +
                   ([] if mode == MODES[0] else ['W_zero', 'W_gradient_finite'])))
              for mode in MODES]
    result += [dict(id='phase_operation', required_keys=sorted(OPERATION_KEYS)),
               dict(id='native_recurrence_reference', required_keys=sorted(REFERENCE_KEYS))]
    result += [dict(id=mode + '_shift', required_keys=sorted(SHIFT_KEYS)) for mode in MODES]
    result += [dict(id=mode + '_edges', required_keys=sorted(EDGE_KEYS)) for mode in MODIFIED]
    result += [dict(id=f'{mode}_prefix{prefix}', required_keys=sorted(CAUSAL_KEYS))
               for mode in MODIFIED for prefix in (7, 8, 9)]
    result += [dict(id='negative_controls', required_keys=sorted(NEGATIVE_KEYS))]
    return result


def ok(condition, **evidence):
    return dict(passed=bool(condition), **evidence)


def packed(value):
    """Small fixtures retain numeric values as well as byte hashes."""
    return dict(**tensor_records({'value': value})['value'], values=value.detach().cpu().tolist())


def emit_to(save, checks):
    def emit(key, value):
        if key in checks:
            raise ValueError('Duplicate numerical leaf: ' + key)
        checks[key] = value
        save(case(dict(checks)))
    return emit


def fresh(mode, device='cpu', historical=False, seed=SYNTHETIC_SEED):
    from recbole.config import Config
    from . import config as c
    from .model import AbsolutePhaseMamba3Rec
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.config import SyntheticCatalog
    values = c.settings(mode, seed=seed, device=device)
    if historical:
        values.pop('phase_mode', None)
    cfg = Config(model=ThreeTimeMamba3Rec, config_dict=values)
    seed_all(seed)
    cls = ThreeTimeMamba3Rec if historical else AbsolutePhaseMamba3Rec
    return cls(cfg, SyntheticCatalog()).to(device).eval()


def histories(length, batch=2, zero=False, padded=False, device='cuda'):
    generator = torch.Generator().manual_seed(SYNTHETIC_SEED)
    items = torch.randint(1, 7112, (batch, length), generator=generator)
    lengths = torch.full((batch,), length, dtype=torch.long)
    if padded and batch > 1:
        lengths[-1] = max(1, length - 7)
        items[-1, lengths[-1]:] = 0
    gaps = torch.randint(0, 8, (batch, length), generator=generator).double() * 300001
    if zero:
        gaps.zero_()
    timestamps = 1.6e12 + gaps.cumsum(1)
    return tuple(value.to(device) for value in (items, lengths, timestamps))


def nonzero_W(net):
    if net.phase_adapter is not None:
        with torch.no_grad():
            w = net.phase_adapter.W
            w.copy_(.8 * torch.sin(torch.arange(w.numel(), device=w.device).reshape_as(w) * .37 + .2))


def parity(mode, save):
    from .model import transfer_common
    from . import config as c
    from experiments.mamba3_mimo_time.admission import model_measure
    from experiments.mamba3_mimo_time.provenance import assert_upstream
    from experiments.mamba3_head_timescales.checks import nonzero
    from experiments.mamba3_three_time.confirmation.state import compare_optimizer_settings
    reference = fresh('baseline_dual', 'cuda', historical=True)
    left_initial_rng = rng_record()
    net = fresh(mode, 'cuda')
    right_initial_rng = rng_record()
    for cal in reference.times.calibrators.values():
        nonzero(cal)
    mapping = transfer_common(reference, net)
    assert_upstream(reference)
    assert_upstream(net)
    common = c.plan()['common_parameter_keys']
    checks = {}
    emit = emit_to(save, checks)
    emit('historical_mode_absent', ok(not hasattr(reference, 'phase_mode')))
    emit('initial_rng', ok(left_initial_rng == right_initial_rng))
    emit('common_state_exact', ok(all(torch.equal(v, net.state_dict()[k]) for k, v in reference.state_dict().items())))
    emit('parameter_count', ok(sum(p.numel() for p in net.parameters()) == 715020 + (128 if mode in MODIFIED else 0)))
    if mode in MODIFIED:
        emit('W_zero', ok(net.phase_adapter.W.shape == (32, 4) and net.phase_adapter.W.dtype == torch.float32 and not torch.count_nonzero(net.phase_adapter.W)))
    reference.train()
    net.train()
    data = histories(50, batch=3, padded=True)
    left = model_measure(reference, data)
    left_rng = rng_record()
    right = model_measure(net, data)
    right_rng = rng_record()
    for key in ('output', 'loss', 'input_gradient'):
        emit(key, difference(right.get(key), left.get(key)))
    for key in common:
        emit('gradient:' + key, difference(right.get('gradient:' + key), left.get('gradient:' + key)))
    emit('dropout_rng', ok(left_rng == right_rng))
    emit('initial_output_bitwise', ok(checks['output']['bitwise_equal'] and checks['loss']['bitwise_equal']))
    if mode in MODIFIED:
        emit('W_gradient_finite', finite(net.phase_adapter.W.grad))
    a = torch.optim.Adam(reference.parameters(), lr=.001)
    b = torch.optim.Adam(net.parameters(), lr=.001)
    settings = lambda optimizer: [{k: v for k, v in row.items() if k != 'params'} for row in optimizer.param_groups]
    serialized = json.loads(json.dumps(settings(a), allow_nan=False))
    compare_optimizer_settings(serialized, settings(b))
    emit('adam_json_roundtrip', ok(True, settings=serialized))
    a.step()
    b.step()
    for key in common:
        emit('first_adam:' + key, difference(dict(net.named_parameters())[key], dict(reference.named_parameters())[key]))
    emit('hook_cleanup', ok(not reference.item_embedding._forward_hooks and not net.item_embedding._forward_hooks))
    return case(checks, state_mapping=mapping, fixture_seed=SYNTHETIC_SEED,
                first_adam_scope='Common parameters only; W may move, no post-update output parity required',
                input_tensors=tensor_records(dict(zip(('items', 'lengths', 'timestamps'), data))))


def phase_operations(save, device='cuda'):
    from .phase import PeriodicPhase, phase_clock, periodic_features, apply_correction
    from experiments.mamba3_timeaware.time_inputs import history_gaps
    checks = {}
    emit = emit_to(save, checks)
    t = torch.tensor([[1600000000123., 1600000000123., 1600000738246., 1600022223456.],
                      [1600080012345., 1600080123456., 0., 0.]], dtype=torch.float64, device=device)
    valid = torch.tensor([[True] * 4, [True, True, False, False]], device=device)
    gaps, active = history_gaps(t, valid)
    feature = periodic_features(t, valid, 'absolute_phase', dtype=torch.float64)
    reference = torch.zeros_like(feature)
    for b in range(2):
        for i in range(4):
            for k, period in enumerate((21600000., 86400000.)):
                clock = float(t[b, i]) if valid[b, i] else 0.
                angle = 2 * math.pi * (clock % period) / period
                reference[b, i, 2*k] = math.sin(angle) / math.sqrt(2)
                reference[b, i, 2*k+1] = math.cos(angle) / math.sqrt(2)
    emit('fp64_feature_reference', ok(torch.allclose(feature, reference, atol=1e-14, rtol=1e-14), max_abs=(feature-reference).abs().max().item()))
    w = torch.linspace(-.3, .4, 32*4, device=device, dtype=torch.float64).reshape(32, 4).requires_grad_()
    correction = apply_correction(feature, w, active)
    independent = torch.stack([torch.stack([torch.stack([
        torch.tanh(sum(w[a, k]*reference[b, i, k] for k in range(4))) if active[b, i] else w[a, 0]*0
        for a in range(32)]) for i in range(4)]) for b in range(2)])
    emit('fp64_correction_reference', ok(torch.allclose(correction, independent, atol=1e-14, rtol=1e-14), max_abs=(correction-independent).abs().max().item()))
    # Times avoid remainder boundaries. This tests the actual operation in FP64,
    # including temporal derivatives, without changing production W precision.
    small_t = torch.tensor([[1234.567, 9234.578, 23456.789]], dtype=torch.float64, device=device, requires_grad=True)
    small_w = w[:2].detach().clone().requires_grad_()
    small_v = valid[:1, :3]
    def operation(times, weights):
        f = periodic_features(times * 1000., small_v, 'absolute_phase', dtype=torch.float64)
        return apply_correction(f, weights, active[:1, :3])
    # A seconds-valued differentiation variable feeds the production millisecond
    # clock, avoiding Unix-magnitude cancellation in numerical differences.
    emit('gradcheck', ok(torch.autograd.gradcheck(operation, (small_t, small_w), eps=1e-4, atol=1e-7, rtol=1e-4)))
    cpu_f = periodic_features(t.cpu(), valid.cpu(), 'absolute_phase', dtype=torch.float64)
    emit('cpu_device_agreement', compare(feature.cpu(), cpu_f))
    emit('feature_shape_dtype', ok(list(feature.shape) == [2, 4, 4] and feature.dtype == torch.float64 and feature.device == t.device))
    emit('first_padding_neutral', ok((correction[~active] == 0).all() and not active[:, 0].any()))
    emit('real_zero_gap_active', ok(active[0, 1] and gaps[0, 1] == 0 and correction[0, 1].abs().sum() > 0))
    bad = t.clone()
    bad[0, 1] = float('nan')
    rejected = False
    try:
        periodic_features(bad, valid, 'absolute_phase')
    except ValueError:
        rejected = True
    emit('valid_nonfinite_rejected', ok(rejected))
    sanitized = t.clone()
    sanitized[~valid] = float('nan')
    clean = periodic_features(sanitized, valid, 'absolute_phase', dtype=torch.float64)
    emit('padding_sanitized', ok(torch.isfinite(clean).all() and torch.equal(clean, feature)))
    large = t + 1e18
    emit('large_timestamp_finite', finite(periodic_features(large, valid, 'absolute_phase')))
    relative = phase_clock(t, valid, 'relative_phase')
    emit('relative_origin_and_gaps', ok(torch.equal(relative, torch.where(valid, gaps.cumsum(1), 0.)) and not relative[:, 0].any()))
    for k, period in enumerate((21600000., 86400000.)):
        shifted = periodic_features(t + period, valid, 'absolute_phase', dtype=torch.float64)
        emit('period6h' if k == 0 else 'period24h', compare(shifted[..., 2*k:2*k+2], feature[..., 2*k:2*k+2]))
    boundary = torch.tensor([[21600000.-.001, 21600000.+.001]], dtype=torch.float64, device=device)
    near = periodic_features(boundary, torch.ones_like(boundary, dtype=torch.bool), 'absolute_phase', dtype=torch.float64)
    emit('period_boundary_continuity', ok((near[0, 0, :2]-near[0, 1, :2]).abs().max() < 1e-8))
    for k, key in enumerate(('frequency6h_gradient', 'frequency24h_gradient')):
        zero = torch.zeros((32, 4), device=device, dtype=torch.float64, requires_grad=True)
        grad = torch.autograd.grad(apply_correction(feature, zero, active).sum(), zero)[0]
        emit(key, ok(torch.isfinite(grad).all() and grad[:, 2*k:2*k+2].norm() > 0, l2=grad[:, 2*k:2*k+2].norm().item()))
    rounding = correction.float().to(torch.bfloat16).float() - correction.float()
    emit('bf16_rounding_measured', ok(torch.isfinite(rounding).all(), max_abs=rounding.abs().max().item(), l2=rounding.norm().item(),
                                   scope='Hypothetical correction BF16 cast; production MIMO raw angles remain FP32'))
    before = rng_record()
    adapter = PeriodicPhase(32, 'absolute_phase').to(device)
    after = rng_record()
    emit('zero_init_rng', ok(before == after and adapter.W.dtype == torch.float32 and not torch.count_nonzero(adapter.W)))
    return case(checks, timestamps=packed(t), features=packed(feature), correction=packed(correction))


def informative_history(device='cuda'):
    # Frozen before metrics: distinct content, a real zero-gap, nonuniform gaps,
    # more than one event and both periodic phases. No search for a passing fixture.
    items = torch.tensor([[3, 71, 109, 23, 401, 82, 9, 55]], device=device)
    lengths = torch.tensor([8], device=device)
    offsets = torch.tensor([[0, 0, 738123, 3615123, 7623456, 15001357, 27009123, 61012345]], dtype=torch.float64, device=device)
    return items, lengths, offsets + 1600000000123.


def native_phase(values):
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
    from experiments.mamba3_three_time.length_adapter import prepare
    names = ('q', 'k', 'v', 'adt', 'dw', 'dp', 'trap', 'qb', 'kb', 'angles', 'd', 'z', 'mv', 'mz', 'mo')
    prepared, _ = prepare({key: values[key] for key in names})
    theta, _ = angle_dt_fwd(prepared['angles'].contiguous(), prepared['dp'].contiguous(), chunk_size=8, return_output_state=True)
    return theta


def phase_shift(mode, save):
    from .phase import periodic_features
    from experiments.mamba3_timeaware.time_inputs import history_gaps
    from experiments.mamba3_three_time.fixtures import scalar_loss
    net = fresh(mode, 'cuda')
    # An explicit informative synthetic projection scale, never a training change.
    with torch.no_grad():
        for layer in net.layers:
            layer.mixer.dt_bias.fill_(math.log(math.expm1(.25)))
    nonzero_W(net)
    data = informative_history()
    items, lengths, timestamps = data
    valid = items != 0
    checks = {}
    emit = emit_to(save, checks)
    observed = {}
    def run(label, shift):
        capture = {}
        def observer(index, values):
            if index == 0:
                capture.update({k: v.detach().clone() for k, v in values.items() if torch.is_tensor(v)})
        with torch.no_grad():
            output = net.encode_sequence(items, lengths, timestamps + shift, observer=observer)
            theta = native_phase(capture)
        observed[label] = dict(timestamps=packed(timestamps+shift), output=packed(output),
                               theta=packed(theta), raw_angles=packed(capture['angles']),
                               DT_phase=packed(capture['dp']),
                               phase_increments_fp32=packed(math.pi*capture['angles'].tanh()*capture['dp'].permute(0, 2, 1).unsqueeze(-1)),
                               periodic_feature_role='not consumed by baseline' if mode == 'baseline_dual' else 'model input',
                               correction=packed(capture['raw_angle_correction']),
                               features=packed(periodic_features(timestamps+shift, valid,
                                     'relative_phase' if mode == 'relative_phase' else 'absolute_phase')))
        return output, capture, theta
    original, native, theta = run('original', 0.)
    with torch.no_grad():
        unobserved = net.encode_sequence(*data)
    emit('observer_output_parity', difference(original, unobserved))
    shifted, later, theta2 = run('plus_2h', 7200000.)
    periodic, daily, theta24 = run('plus_24h', 86400000.)
    gaps, active = history_gaps(timestamps, valid)
    emit('unchanged_gaps_2h', ok(torch.equal(gaps, history_gaps(timestamps+7200000., valid)[0])))
    feature_mode = 'relative_phase' if mode == 'relative_phase' else 'absolute_phase'
    f = periodic_features(timestamps, valid, feature_mode)
    f2 = periodic_features(timestamps+7200000., valid, feature_mode)
    f24 = periodic_features(timestamps+86400000., valid, feature_mode)
    if mode == 'absolute_phase':
        for key, value, reference in [('features_2h', f2, f),
                ('correction_2h', later['raw_angle_correction'], native['raw_angle_correction']),
                ('output_2h', shifted, original)]:
            diff = difference(value, reference)
            emit(key, ok(not diff['passed'] and diff.get('max_abs', 0) > 0,
                         measurement={('within_invariance_tolerance' if k == 'passed' else k): v for k, v in diff.items()},
                         scope='Deliberately informative fixed fixture'))
    else:
        emit('features_2h', ok(True, scope='No periodic input in baseline') if mode == 'baseline_dual' else difference(f2, f))
        emit('correction_2h', difference(later['raw_angle_correction'], native['raw_angle_correction']))
        emit('output_2h', difference(shifted, original))
    emit('features_24h', difference(f24, f))
    emit('correction_24h', difference(daily['raw_angle_correction'], native['raw_angle_correction']))
    emit('output_24h', difference(periodic, original))
    emit('first_padding_neutral', ok((native['raw_angle_correction'][~active] == 0).all()))
    invariant = ('q', 'k', 'v', 'adt', 'dw', 'dp', 'trap', 'qb', 'kb', 'd', 'z', 'mv', 'mz', 'mo', 'raw_angles_before')
    emit('same_mixer_nonphase_exact', ok(all(torch.equal(native[k], later[k]) for k in invariant), keys=list(invariant),
                                       scope='Same first-mixer input; downstream second-layer changes are expected'))
    increment = native['angles'] - native['raw_angles_before']
    emit('shared_heads', ok(torch.equal(increment[:, :, 0], increment[:, :, 1])))
    emit('native_phase_finite', ok(all(torch.isfinite(x).all() for x in (theta, theta2, theta24)),
                                 max_abs_2h=(theta2-theta).abs().max().item()))
    if mode in MODIFIED:
        net.zero_grad(set_to_none=True)
        scalar_loss(net(*data)).backward()
        gradient = net.phase_adapter.W.grad
        emit('W_gradient_informative', ok(gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0,
                                         l2=gradient.norm().item() if gradient is not None else None))
        with torch.no_grad():
            net.phase_adapter.W.zero_()
    else:
        emit('W_gradient_informative', ok(net.phase_adapter is None, scope='Baseline has no W'))
    with torch.no_grad():
        zero = net(*data)
        zero2 = net(items, lengths, timestamps+7200000.)
        zero24 = net(items, lengths, timestamps+86400000.)
    emit('zero_W_output_2h', difference(zero2, zero))
    emit('zero_W_output_24h', difference(zero24, zero))
    emit('hook_cleanup', ok(not net.item_embedding._forward_hooks and not net.output_norm._forward_hooks))
    return case(checks, fixture_seed=SYNTHETIC_SEED, numeric_fixture=observed,
                W_definition='0.8*sin(arange(A*4)*0.37+0.2)', dt_bias_fixture='inverse_softplus(0.25)',
                zero_W_outputs={k: packed(v) for k, v in [('original', zero), ('plus_2h', zero2), ('plus_24h', zero24)]})


def native_reference(save):
    from .phase import periodic_features, apply_correction
    from experiments.mamba3_mimo_time.numerics import fixture, cotangent
    from experiments.mamba3_three_time.fixtures import kernel_output
    from experiments.mamba3_three_time.reference import recurrence
    source = fixture(7, seed=SYNTHETIC_SEED, tied=True)
    timestamps = informative_history()[2][:, :7]
    valid = torch.ones_like(timestamps, dtype=torch.bool)
    active = valid.clone()
    active[:, 0] = False
    features = periodic_features(timestamps, valid, 'absolute_phase')
    w = (.15 * torch.sin(torch.arange(128, device='cuda').reshape(32, 4)*.37+.2)).requires_grad_()
    values = {k: v.detach() for k, v in source.items()}
    values['dp'] = values['dw']
    values['angles'] = values['angles'] + apply_correction(features, w, active).unsqueeze(2)
    out = kernel_output(values, 'MIMO')
    incoming, meta = cotangent(out.shape, 'signed', 'cuda')
    grad = torch.autograd.grad(out, w, incoming)[0]
    reference_w = w.detach().double().requires_grad_()
    reference = {k: v.detach().double() for k, v in source.items()}
    reference['angles'] = reference['angles'] + apply_correction(features.double(), reference_w, active).unsqueeze(2)
    expected = recurrence(**reference)
    expected_grad = torch.autograd.grad(expected, reference_w, incoming.double())[0]
    checks = dict(output=compare(out, expected, 'output'), W_gradient=compare(grad, expected_grad, 'vjp'),
                  W_gradient_informative=ok(torch.isfinite(grad).all() and grad.norm() > 0, l2=grad.norm().item()),
                  native_phase_finite=finite(native_phase(values)))
    save(case(checks))
    return case(checks, cotangent=meta, output=packed(out), reference_output=packed(expected),
                W_gradient=packed(grad), reference_W_gradient=packed(expected_grad),
                scope='Small synthetic FP64 independent recurrence; inherited output/VJP policies unchanged')


def model_edges(mode, save):
    from recbole.data.interaction import Interaction
    checks = {}
    emit = emit_to(save, checks)
    net = fresh(mode, 'cuda')
    nonzero_W(net)
    data = histories(50, batch=3, padded=True)
    data[1][0] = 1
    data[0][0, 1:] = 0
    with torch.no_grad():
        out = net.encode_sequence(*data)
        order = torch.tensor([2, 0, 1], device='cuda')
        permuted = net.encode_sequence(*(x[order] for x in data))
        net(*histories(17, batch=2, zero=True))
        repeated = net.encode_sequence(*data)
        changed = data[2].clone()
        changed[data[0] == 0] = float('nan')
        padded = net.encode_sequence(data[0], data[1], changed)
    emit('parameter_count', ok(sum(p.numel() for p in net.parameters()) == 715148))
    emit('incomplete_batch', ok(list(out.shape) == [3, 50, 64] and torch.isfinite(out).all()))
    emit('batch_permutation', difference(permuted, out[order]))
    emit('no_cross_batch_state', difference(repeated, out))
    emit('padding_neutral', difference(padded, out))
    for n in (1,) + LENGTHS:
        net.zero_grad(set_to_none=True)
        value = net(*histories(n, batch=2, padded=n > 1))
        value.square().mean().backward()
        backward = ok(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
        if n == 1:
            emit('length1_backward', backward)
        else:
            emit(f'length{n}:output', finite(value))
            emit(f'length{n}:backward', backward)
    stream = io.BytesIO()
    torch.save(net.state_dict(), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location='cuda', weights_only=True)
    net.load_state_dict(saved, strict=True)
    with torch.no_grad():
        recovered = net.encode_sequence(*data)
    emit('roundtrip', difference(recovered, out))
    inputs = {net.ITEM_SEQ: data[0], net.ITEM_SEQ_LEN: data[1], net.time_sequence_field: data[2]}
    with torch.no_grad():
        missing = net._encode(Interaction(inputs))
        explicit = Interaction(dict(inputs, timestamp=torch.ones(3, device='cuda')))
        before = net._encode(explicit)
        explicit['timestamp'].fill_(9e14)
        after = net._encode(explicit)
    emit('target_timestamp_absent', difference(missing, before))
    emit('target_timestamp_ignored', difference(after, before))
    emit('hook_cleanup', ok(not net.output_norm._forward_hooks and not net.item_embedding._forward_hooks))
    return case(checks, lengths=list(LENGTHS), roundtrip='weights_only=True')


def causal(mode, prefix, save):
    from .phase import periodic_features
    from experiments.mamba3_mimo_time.prefix_checks import check_prefix
    from experiments.mamba3_head_timescales.checks import nonzero
    net = fresh(mode, 'cuda')
    for cal in net.times.calibrators.values():
        nonzero(cal)
    nonzero_W(net)
    data = histories(50, batch=3)
    items, lens, times = data
    changed_items = items.clone()
    changed_items[0, prefix:] = (changed_items[0, prefix:]+37) % 7111+1
    changed_times = times.clone()
    changed_times[0, prefix:] += 900000000000.
    interventions = ((changed_items, lens, times), (items, lens, changed_times))
    latest = {}
    checks = check_prefix(net, data, prefix, 1., None, interventions, progress_callback(save, latest))
    original = periodic_features(times, items != 0, mode)
    features = [periodic_features(row[2], row[0] != 0, mode) for row in interventions]
    for label, altered in zip(('item', 'timestamp'), features):
        checks['features_prefix_'+label] = ok(torch.equal(altered[:, :prefix], original[:, :prefix]))
    checks['features_cross_user'] = ok(all(torch.equal(f[1:], original[1:]) for f in features))
    checks['input_interventions_real'] = ok(not torch.equal(items, changed_items) and not torch.equal(times, changed_times) and
        torch.equal(items[:, :prefix], changed_items[:, :prefix]) and torch.equal(times[:, :prefix], changed_times[:, :prefix]))
    phase = latest['phase']
    checks['hook_lifecycle'] = ok(phase['hook_removed'] and phase['capture_count'] == 1 and not net.item_embedding._forward_hooks and
        all(not s['grad_enabled'] for s in phase['stages'] if s['stage'].startswith('intervention')))
    return snapshot(case(checks, hook_phase=phase, legacy_exact_zero_reclassified=False,
        input_tensors=tensor_records(dict(items=items, timestamps=times, changed_items=changed_items, changed_timestamps=changed_times))))


def negative_controls(save, device='cuda'):
    # Reuse the proven detached/non-detached leaking fixtures and production
    # prefix callback. No old model, memory branch or fit is run here.
    from experiments.mamba3_time_memory.checks import negative_controls as inherited
    return inherited(save, device=device)


def dispatch(spec, save):
    name = spec['id']
    if name == 'phase_operation':
        return phase_operations(save)
    if name == 'native_recurrence_reference':
        return native_reference(save)
    if name == 'negative_controls':
        return negative_controls(save)
    for mode in MODES:
        if name == mode+'_W0':
            return parity(mode, save)
        if name == mode+'_shift':
            return phase_shift(mode, save)
        if name == mode+'_edges':
            return model_edges(mode, save)
        for prefix in (7, 8, 9):
            if name == f'{mode}_prefix{prefix}':
                return causal(mode, prefix, save)
    raise ValueError('Unknown frozen case: ' + name)
