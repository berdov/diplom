"""Read-only state snapshots, strict name mapping, fixed structural comparisons."""
import hashlib
import json
import math
import random
import numpy as np
import torch
from experiments.mamba3_context_time.provenance import tensor_hash, rng_hash

ATOL, RTOL = 1e-6, 1e-5


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state().clone(),
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def rng_record(loader=None):
    state = capture_rng()
    n = state['numpy']
    digest = lambda x: hashlib.sha256(json.dumps(x).encode()).hexdigest()
    value = dict(python=digest(state['python']), numpy=digest([n[0], n[1].tolist(), *n[2:]]),
                 cpu=tensor_hash({'state': state['cpu']}),
                 cuda=[tensor_hash({'state': s}) for s in state['cuda']], aggregate=rng_hash())
    if loader is not None:
        value['loader_generator'] = tensor_hash({'state': loader.generator.get_state()})
    return value


def precision():
    return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def normalized(values):
    result = {}
    for key, value in values.items():
        target = 'times.calibrators.' + key[len('mechanisms.calibrators.'):] if key.startswith('mechanisms.calibrators.') else key
        if target in result:
            raise ValueError('Ambiguous state mapping: ' + target)
        result[target] = value
    return result


def strict_transfer(left, right):
    source, target = normalized(left.state_dict()), right.state_dict()
    if set(source) != set(target):
        raise ValueError('State keys mismatch: ' + repr(sorted(set(source) ^ set(target))))
    for k in source:
        if source[k].shape != target[k].shape or source[k].dtype != target[k].dtype:
            raise ValueError('State shape/dtype mismatch: ' + k)
    right.load_state_dict(source, strict=True)
    if any(not torch.equal(v, right.state_dict()[k]) for k, v in source.items()):
        raise ValueError('State values not transferred exactly')
    return dict(mapping='mechanisms.calibrators.* -> times.calibrators.*; other keys identical',
                keys=sorted(source), all_keys_shapes_dtypes_values_equal=True)


def difference(value, reference):
    if value is None or reference is None or value.shape != reference.shape or value.dtype != reference.dtype:
        return dict(passed=False, reason='missing tensor or shape/dtype mismatch')
    a, b = value.detach().cpu().double(), reference.detach().cpu().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return dict(passed=False, reason='NaN/Inf')
    delta = a-b
    bits = lambda t: t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return dict(passed=bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)), bitwise_equal=torch.equal(bits(value), bits(reference)),
                max_abs=delta.abs().max().item() if a.numel() else 0., mean_abs=delta.abs().mean().item() if a.numel() else 0.,
                l2_error=delta.norm().item(), reference_norm=b.norm().item(), atol=ATOL, rtol=RTOL,
                shape=list(value.shape), dtype=str(value.dtype))


def compare_named(value, reference):
    rows = {k: difference(value.get(k), reference.get(k)) for k in sorted(set(value) | set(reference))}
    return dict(passed=bool(rows) and all(x['passed'] for x in rows.values()), tensors=rows,
                failed_names=[k for k, v in rows.items() if not v['passed']])


def initial(model, loader):
    from experiments.mamba3_three_time.validation_pilot.provenance import backbone_hash
    calibrators = model.times.calibrators
    return dict(initial_backbone_sha256=backbone_hash(model),
                initial_calibrator_hashes={k: tensor_hash(v.state_dict()) for k, v in calibrators.items()},
                rng_components=rng_record(loader), rng_before_fit_sha256=rng_hash())


def canonical_optimizer_settings(settings):
    """Copy JSON-safe metadata without altering optimizer groups or scalar types."""
    def visit(value, path):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if type(value) in (list, tuple):
            return [visit(v, f'{path}[{i}]') for i, v in enumerate(value)]
        if type(value) is dict and all(type(k) is str for k in value):
            return {k: visit(v, f'{path}.{k}') for k, v in value.items()}
        raise ValueError(f'Unsupported optimizer metadata at {path}: {value!r} ({type(value).__name__})')
    if type(settings) not in (list, tuple) or not all(type(g) is dict for g in settings):
        raise ValueError('optimizer_settings must be an ordered sequence of groups')
    return visit(settings, 'optimizer_settings')


def compare_optimizer_settings(a, b):
    missing = object()
    def compare(left, right, path):
        if type(left) is dict and type(right) is dict:
            for key in sorted(set(left) | set(right)):
                compare(left.get(key, missing), right.get(key, missing), f'{path}.{key}')
            return
        if type(left) is list and type(right) is list:
            for i in range(max(len(left), len(right))):
                compare(left[i] if i < len(left) else missing,
                        right[i] if i < len(right) else missing, f'{path}[{i}]')
            return
        if type(left) is not type(right) or left != right:
            def show(v):
                return '<missing>' if v is missing else f'{v!r} ({type(v).__name__})'
            raise ValueError(f'Pair mismatch: {path}: {show(left)} != {show(right)}')
    compare(canonical_optimizer_settings(a), canonical_optimizer_settings(b), 'optimizer_settings')


def paired(a, b, first_batch=False):
    for key in ('initial_backbone_sha256', 'rng_components', 'rng_before_fit_sha256', 'protocol',
                 'manifest_sha256', 'train_time_stats_sha256', 'verified_history_stats', 'precision'):
        if a[key] != b[key]:
            raise ValueError('Pair mismatch: ' + key)
    compare_optimizer_settings(a['optimizer_settings'], b['optimizer_settings'])
    ca, cb = a['initial_calibrator_hashes'], b['initial_calibrator_hashes']
    if ca['decay'] != cb['decay'] or any(ca['scan'] != cb[k] for k in ('write', 'phase')):
        raise ValueError('Pair calibrator initialization mismatch')
    if first_batch and a['first_train_batch_sha256'] != b['first_train_batch_sha256']:
        raise ValueError('Pair consumed batch mismatch')
