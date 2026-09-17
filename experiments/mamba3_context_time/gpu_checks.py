"""Mandatory synthetic CUDA gates; no dataset or recommendation evaluation."""
import copy
import json
import os
import traceback

from .config import GPU_EVIDENCE, RUNTIME, plan, settings
from .provenance import atomic_json, exclusive, now, require_submission, save_state_dict, PIN, backbone_hash, rng_hash


def compare(a, b):
    import torch
    if a is None or b is None:
        return dict(passed=False, reason='missing tensor/gradient')
    delta = (a.float() - b.float()).abs()
    finite = bool(torch.isfinite(delta).all() and torch.isfinite(a).all() and torch.isfinite(b).all())
    return dict(passed=finite and bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5)),
                max_abs_error=float(delta.max()) if finite else None,
                mean_abs_error=float(delta.mean()) if finite else None, finite=finite)


class Catalog:
    def num(self, field):
        return 7112 if field == 'item_id' else 23952


def nonzero(module):
    import torch
    generator = torch.Generator().manual_seed(619)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if 'last.' in name:
                p.copy_((.04 * torch.randn(p.shape, generator=generator) + .03).to(p))


def side(model, batch, vanilla=False):
    import torch
    from experiments.mamba3_baseline.model import Mamba3Rec
    capture = []
    def hook(module, args, output):
        output.retain_grad()
        capture.append(output)
    handle = model.item_embedding.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(2026)
        items, lengths, times, targets = batch
        output = Mamba3Rec.forward(model, items, lengths) if vanilla else model(items, lengths, times)
        loss = model.loss_fct(output @ model.item_embedding.weight.T, targets)
        loss.backward()
        gradients = {k: p.grad.detach().clone() if p.grad is not None else None for k, p in model.named_parameters()}
        return output.detach(), loss.detach(), capture[0].grad.detach(), gradients
    finally:
        handle.remove()


def comparison(left, right, *, temporal_map=None):
    result = {k: compare(a, b) for k, a, b in zip(('output', 'loss', 'embedding_output_gradient'), left[:3], right[:3])}
    for name, value in left[3].items():
        if name.startswith('mechanisms.'):
            if temporal_map is None:
                continue
            other = temporal_map(name)
        else:
            other = name
        result['gradient:' + name] = compare(value, right[3].get(other))
    return result


def case(training, length):
    import torch
    from recbole.config import Config
    from experiments.mamba3_time_mechanisms.model import MechanismMamba3Rec
    from .model import ContextMamba3Rec
    from .temporal import ContextTime, COUNTS

    def new(mode):
        task = next(t for t in plan()['tasks'] if t['mode'] == mode)
        cfg = Config(model=ContextMamba3Rec, config_dict=settings(task))
        return ContextMamba3Rec(cfg, Catalog()).cuda().train(training)

    torch.manual_seed(811)
    cfg = Config(model=ContextMamba3Rec, config_dict=settings(plan()['tasks'][0]))
    ref = MechanismMamba3Rec(cfg, Catalog()).cuda().train(training)
    items = torch.randint(1, 7112, (3, length), device='cuda')
    lengths = torch.tensor([length, length-7, 1], device='cuda')
    items[torch.arange(length, device='cuda')[None, :] >= lengths[:, None]] = 0
    times = (torch.randint(1, 2000000, (3, length), device='cuda').double().cumsum(1) + 1.6e12)
    times[:, 1] = times[:, 0]
    times[items == 0] = float('nan')
    batch = items, lengths, times, torch.tensor([3, 7, 19], device='cuda')
    common = {k: v for k, v in ref.state_dict().items() if not k.startswith('mechanisms.')}
    expected = side(ref, batch, vanilla=True)
    checks = {}
    counts = {}
    for mode in COUNTS:
        model = new(mode)
        model.load_state_dict({**model.state_dict(), **common})
        counts[mode] = sum(p.numel() for p in model.parameters())
        checks['identity:' + mode] = comparison(expected, side(model, batch))
        del model
    nonzero(ref.mechanisms)
    wrapper = new('separate_replay')
    wrapper.load_state_dict(ref.state_dict())
    trained_like = side(ref, batch)
    checks['separate_wrapper'] = comparison(trained_like, side(wrapper, batch), temporal_map=lambda n: n)
    single = new('uniform')
    single.mechanisms = ContextTime('uniform', experts=1).cuda()
    single.load_state_dict({**single.state_dict(), **common})
    for p in ('decay', 'scan'):
        single.mechanisms.bank[0][p].load_state_dict(ref.mechanisms.calibrators[p].state_dict())
    checks['one_expert_reduction'] = comparison(trained_like, side(single, batch),
        temporal_map=lambda n: n.replace('mechanisms.calibrators.', 'mechanisms.bank.0.'))
    uniform, routed = new('uniform'), new('routed')
    uniform.load_state_dict({**uniform.state_dict(), **common})
    nonzero(uniform.mechanisms)
    routed.load_state_dict({**routed.state_dict(), **uniform.state_dict()})
    with torch.no_grad():
        routed.mechanisms.router.weight.zero_()
        routed.mechanisms.router.bias.zero_()
    checks['uniform_router_reduction'] = comparison(side(uniform, batch), side(routed, batch), temporal_map=lambda n: n)
    grads = dict(routed.named_parameters())
    checks['router_signal'] = dict(passed=all(grads[n].grad is not None and torch.isfinite(grads[n].grad).all().item()
        and grads[n].grad.abs().sum().item() > 0 for n in ('mechanisms.router.weight', 'mechanisms.router.bias')))
    checks['finite_backward'] = dict(passed=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in routed.parameters()))
    from experiments.mamba3_timeaware.time_inputs import history_gaps
    gaps, active = history_gaps(times, items != 0)
    with torch.no_grad():
        before = routed.mechanisms(routed.item_embedding(items), gaps, active)[2]['log_experts'].clone()
    optimizer = torch.optim.Adam(routed.parameters(), lr=.001)
    optimizer.step()
    with torch.no_grad():
        after = routed.mechanisms(routed.item_embedding(items), gaps, active)[2]['log_experts']
    checks['expert_update'] = dict(passed=not torch.equal(before, after) and bool(torch.isfinite(after).all()))
    path = RUNTIME / 'gpu_gate' / f'roundtrip_{int(training)}_{length}.pth'
    if path.exists():
        raise FileExistsError('Synthetic checkpoint exists')
    save_state_dict(routed, path)
    clone = new('routed')
    clone.load_state_dict(torch.load(path, weights_only=True, map_location='cuda'), strict=True)
    checks['safe_roundtrip'] = comparison(side(routed, batch), side(clone, batch), temporal_map=lambda n: n)
    return dict(training=training, length=length, counts=counts, checks=checks)


def all_pass(checks):
    if 'passed' in checks:
        return checks['passed'] is True
    return bool(checks) and all(all_pass(v) for v in checks.values())


def initialization_gate():
    import torch
    from recbole.config import Config
    from recbole.utils import init_seed
    from .model import ContextMamba3Rec
    from .provenance import tensor_hash
    rows = []
    for task in plan()['tasks']:
        cfg = Config(model=ContextMamba3Rec, config_dict=settings(task))
        init_seed(2026, True)
        model = ContextMamba3Rec(cfg, Catalog()).cuda()
        rows.append(dict(mode=task['mode'], backbone=backbone_hash(model), rng=rng_hash(),
            bank=tensor_hash(model.mechanisms.bank.state_dict()) if task['mode'] in ('uniform', 'routed') else None))
        del model
    if len({r['backbone'] for r in rows}) != 1 or len({r['rng'] for r in rows}) != 1 or rows[3]['bank'] != rows[4]['bank']:
        raise ValueError('Initialization or RNG parity failed')
    return rows


def main():
    manifest = require_submission()
    if GPU_EVIDENCE.exists():
        raise FileExistsError('GPU evidence exists')
    exclusive(RUNTIME / 'gpu_gate/gate.lock', dict(stage='GPU_GATE', started_at=now()))
    result = dict(status='RUNNING', stage='GPU_GATE', source_hash=manifest['source_hash'],
                  pinned_mamba=PIN, atol=1e-6, rtol=1e-5, synthetic_only=True, cases=[], test_evaluation_count=0,
                  execution_commit=os.environ['RUN_COMMIT'], job_id=os.environ['SLURM_JOB_ID'],
                  node=os.environ.get('SLURMD_NODENAME'), started_at=now())
    atomic_json(GPU_EVIDENCE, result)
    try:
        import torch
        from .preflight import runtime_check
        runtime_check(require_cuda=True)
        result['gpu'] = torch.cuda.get_device_name()
        result['initialization'] = initialization_gate()
        for training in (False, True):
            for length in (50, 64):
                row = case(training, length)
                row['passed'] = all_pass(row['checks'])
                result['cases'].append(row)
                atomic_json(GPU_EVIDENCE, result)
                if not row['passed']:
                    raise ValueError(f'GPU equivalence failed: training={training}, L={length}')
                torch.cuda.empty_cache()
        result.update(status='PASS', stage='COMPLETED')
    except BaseException as exc:
        result.update(status='GPU_GATE_FAIL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['finished_at'] = now()
        atomic_json(GPU_EVIDENCE, result)


if __name__ == '__main__':
    main()
