"""Synthetic adapter checks shared by CPU tests and mandatory GPU gates."""
import copy
import math

import torch
from torch.nn import functional as F

from .adapter import InputAdapter, MODES, REFERENCE


def compare(a, b, atol=1e-6, rtol=1e-5):
    if a is None or b is None:
        return dict(passed=False, reason='missing tensor/gradient')
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a.double()-b.double()).abs()
    return dict(passed=finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)), finite=finite,
                max_abs_error=float(delta.max()) if finite else None,
                mean_abs_error=float(delta.mean()) if finite else None, atol=atol, rtol=rtol)


def all_pass(checks):
    if 'passed' in checks:
        return checks['passed'] is True
    return bool(checks) and all(all_pass(v) for v in checks.values())


def nonzero(module):
    generator = torch.Generator().manual_seed(619)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if 'last.' in name or 'out.' in name:
                p.copy_((.08 * torch.randn(p.shape, generator=generator) + .03).to(p))


def fixture(device='cpu', length=7):
    gen = torch.Generator().manual_seed(913)
    u = torch.randn(3, length, 64, generator=gen).to(device)
    lengths = torch.tensor([length, length-2, 1], device=device)
    valid = torch.arange(length, device=device)[None, :] < lengths[:, None]
    times = torch.tensor([0,0,17,120,900000,9000000,27000000], dtype=torch.float64, device=device)
    if length != 7:
        times = torch.arange(length, dtype=torch.float64, device=device) * 838393
        times[1] = 0
    times = times[None, :].expand(3, -1).clone() + 1.6e12
    times[~valid] = float('nan')
    return u, times, valid


def semantics(device='cpu'):
    u, times, valid = fixture(device)
    checks = {}
    for mode in MODES[1:]:
        adapter = InputAdapter(mode).to(device)
        nonzero(adapter)
        y, details = adapter(u, times, valid, details=True)
        modified = u.clone()
        modified[~valid] = 90000
        changed_times = times.clone()
        changed_times[~valid] = -1e200
        checks[mode + ':masked_values'] = compare(y[valid], adapter(modified, changed_times, valid)[valid])
        checks[mode + ':finite_padding'] = dict(passed=bool(torch.isfinite(y).all()))
        checks[mode + ':padding_residual'] = compare(y[~valid], u[~valid])
        checks[mode + ':time_shift'] = compare(y, adapter(u, times+2**30, valid))
        leaf = u.clone().requires_grad_()
        early = adapter(leaf, times, valid)[:, :2].sum()
        grad = torch.autograd.grad(early, leaf)[0]
        checks[mode + ':future_gradient'] = compare(grad[:, 2:], torch.zeros_like(grad[:, 2:]), 0, 0)
        modified = u.clone()
        modified[:, 2:] *= -70
        changed_times = times.clone()
        changed_times[:, 2:] += 1000000
        checks[mode + ':suffix_invariance'] = compare(y[:, :2], adapter(modified, changed_times, valid)[:, :2])
        for batch in range(3):
            for length in range(1, int(valid[batch].sum())+1):
                prefix = adapter(u[batch:batch+1,:length], times[batch:batch+1,:length], valid[batch:batch+1,:length])
                checks[f'{mode}:prefix:{batch}:{length}'] = compare(prefix, y[batch:batch+1,:length])
        if 'weights' in details:
            weights = details['weights']
            forbidden = ~details['allowed'][:, None].expand_as(weights)
            checks[mode + ':forbidden_weights'] = compare(weights[forbidden], torch.zeros_like(weights[forbidden]), 0, 0)
            checks[mode + ':normalization'] = compare(weights.sum(-1).transpose(1,2)[valid], torch.ones_like(weights.sum(-1).transpose(1,2)[valid]))
            checks[mode + ':equal_time_causality'] = compare(weights[:, :, 0, 1], torch.zeros_like(weights[:, :, 0, 1]), 0, 0)
        if mode == 'attention_content':
            checks[mode + ':timestamp_independent'] = compare(y, adapter(u, times*2, valid), 0, 0)
        else:
            altered = times.clone()
            altered[:, 2:] += 900000
            z = adapter(u, altered, valid)
            checks[mode + ':interval_sensitive'] = dict(passed=bool((z[valid]-y[valid]).abs().max() > 1e-6), max_abs_difference=float((z[valid]-y[valid]).abs().max()))
        if mode == 'time_add':
            checks['time_add:first_identity'] = compare(y[:,0], u[:,0], 0, 0)
            checks['time_add:active_zero_gap'] = dict(passed=bool((y[0,1]-u[0,1]).abs().max()>0))
            altered = times.clone()
            altered[0,0] -= 70000
            checks['time_add:adjacent_only'] = compare(y[:,2:], adapter(u, altered, valid)[:,2:], 0, 0)
    return checks


def reduction(device='cpu'):
    u, times, valid = fixture(device)
    content, temporal = InputAdapter('attention_content').to(device), InputAdapter('attention_time').to(device)
    nonzero(content)
    temporal.load_state_dict({**temporal.state_dict(), **content.state_dict()})
    left = u.clone().requires_grad_()
    right = u.clone().requires_grad_()
    a, b = content(left,times,valid), temporal(right,times,valid)
    a.square().sum().backward()
    b.square().sum().backward()
    checks = dict(output=compare(a,b), input_gradient=compare(left.grad,right.grad))
    for name, p in content.named_parameters():
        checks['gradient:'+name] = compare(p.grad, dict(temporal.named_parameters())[name].grad)
    return checks


def independent_loop(adapter, u, times, valid):
    """Small independent per-query/head oracle: enumerate allowed keys, no broadcast masks."""
    rows = []
    for batch in range(u.shape[0]):
        events = []
        for i in range(u.shape[1]):
            if not bool(valid[batch,i]):
                events.append(u[batch,i])
                continue
            keys = [j for j in range(i+1) if bool(valid[batch,j])]
            q = F.linear(u[batch,i], adapter.q.weight).reshape(4,8)
            k = F.linear(u[batch,keys], adapter.k.weight).reshape(len(keys),4,8)
            v = F.linear(u[batch,keys], adapter.v.weight).reshape(len(keys),4,8)
            bias = None
            if adapter.mode == 'attention_time':
                intervals = torch.stack([(times[batch,i]-times[batch,j]).clamp_min(0) for j in keys])
                r = torch.log1p(intervals/REFERENCE).to(u.dtype)[:,None]
                bias = F.linear(F.silu(F.linear(r,adapter.time_first.weight,adapter.time_first.bias)),adapter.time_last.weight)
            heads = []
            for h in range(4):
                score = (q[h]*k[:,h]).sum(-1)/math.sqrt(8)
                if bias is not None:
                    score = score+bias[:,h]
                heads.append((score.softmax(0)[:,None]*v[:,h]).sum(0))
            events.append(u[batch,i]+F.linear(torch.cat(heads),adapter.out.weight))
        rows.append(torch.stack(events))
    return torch.stack(rows)


def reference_check(device='cpu'):
    checks = {}
    u,times,valid = fixture(device)
    for mode in MODES[2:]:
        a = InputAdapter(mode).to(device)
        nonzero(a)
        b = copy.deepcopy(a)
        x,y = u.clone().requires_grad_(), u.clone().requires_grad_()
        actual, expected = a(x,times,valid), independent_loop(b,y,times,valid)
        actual.square().mean().backward()
        expected.square().mean().backward()
        checks[mode+':output'] = compare(actual,expected,1e-5,1e-4)
        checks[mode+':input_gradient'] = compare(x.grad,y.grad,1e-5,1e-4)
        for name,p in a.named_parameters():
            checks[mode+':gradient:'+name] = compare(p.grad,dict(b.named_parameters())[name].grad,1e-5,1e-4)
    return checks
