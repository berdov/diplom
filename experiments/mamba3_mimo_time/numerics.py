"""Measured finite-precision policy; never modifies gradients or model arithmetic."""
import base64
import torch
from .config import POLICY
from .records import read


def compare(actual, reference, profile='structural', mixed_only=False):
    limits = dict(read(POLICY)[profile])
    if mixed_only:
        limits.pop('relative_l2_cap', None)
    if actual is None or reference is None or actual.shape != reference.shape:
        return dict(passed=False, reason='missing expected tensor/gradient or shape mismatch', **limits)
    dtype_a, dtype_b = str(actual.dtype), str(reference.dtype)
    a, b = actual.detach().double(), reference.detach().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return dict(passed=False, reason='NaN/Inf', **limits)
    error = a-b
    norm, ref_norm = error.norm().item(), b.norm().item()
    relative = norm / ref_norm if ref_norm else None
    zero_status = 'NONZERO_REFERENCE' if ref_norm else ('EXACT_ZERO' if norm == 0 else 'NONZERO_ERROR_WITH_ZERO_REFERENCE')
    mixed = bool(torch.allclose(a, b, atol=limits['atol'], rtol=limits['rtol']))
    cap_pass = ('relative_l2_cap' not in limits or (relative <= limits['relative_l2_cap'] if relative is not None else norm == 0))
    return dict(passed=mixed and cap_pass, finite=True, mixed_pass=mixed, norm_cap_pass=cap_pass,
                max_abs=error.abs().max().item() if error.numel() else 0.,
                mean_abs=error.abs().mean().item() if error.numel() else 0., l2_error=norm,
                actual_norm=a.norm().item(), reference_norm=ref_norm, relative_l2=relative,
                zero_reference_status=zero_status, actual_dtype=dtype_a, reference_dtype=dtype_b, **limits)


def comparisons(actual, reference, profile='structural'):
    return {key: compare(actual.get(key), reference.get(key),
                         'vjp' if profile == 'output' and key.startswith('gradient:') else profile)
            for key in sorted(set(actual) | set(reference))}


def finite(value):
    return dict(passed=value is not None and bool(torch.isfinite(value).all()))


def residual(gradient, prefix):
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        return dict(passed=False, reason='Missing gradient or NaN/Inf', rho=None)
    g = gradient.detach().double()
    p, f = g[:, :prefix].norm().item(), g[:, prefix:].norm().item()
    cap = read(POLICY)['backward_residual']['rho_cap']
    rho = f / (p+f) if p+f else None
    informative = p+f > 0
    return dict(passed=bool(torch.isfinite(g).all()) and informative and rho <= cap,
                prefix_l2=p, future_l2=f, rho=rho, rho_cap=cap,
                zero_reference_status='ZERO_SIGNAL' if not informative else 'MEASURED',
                legacy_exact_zero_pass=f == 0, future_nonzero_count=int(torch.count_nonzero(g[:, prefix:])))


def cotangent(shape, kind, device):
    from experiments.mamba3_three_time.numerics_004 import cotangent as frozen
    from experiments.mamba3_three_time.evidence import tensor_bytes
    value, evidence = frozen(shape, kind, device)
    evidence['actual_bf16_bytes_base64'] = base64.b64encode(tensor_bytes(value)).decode()
    return value, evidence


def fixture(length, seed=314159, tied=False, d_only=False):
    if seed == 2026:
        from experiments.mamba3_three_time.mimo_diagnostics_004 import fixture as old
        return old(length, tied, d_only)
    gen = torch.Generator(device='cuda').manual_seed(seed)
    def rand(shape, scale=.15, shift=0., dtype=torch.bfloat16):
        return (torch.randn(shape, generator=gen, device='cuda', dtype=torch.float64)*scale+shift).to(dtype).requires_grad_(True)
    dt = (1, 2, length)
    x = dict(q=rand((1,length,4,1,128)), k=rand((1,length,4,1,128)), v=rand((1,length,2,64)),
             adt=rand(dt,.005,-.12,torch.float32), dw=rand(dt,.006,.23,torch.float32),
             dp=rand(dt,.01,.31,torch.float32), trap=rand(dt,.3),
             qb=rand((2,4,128),.04,.1,torch.float32), kb=rand((2,4,128),.04,.1,torch.float32),
             angles=rand((1,length,2,32),.03,.09,torch.float32), d=rand((2,),.02,.7,torch.float32),
             z=rand((1,length,2,64),.1,.4), mv=rand((2,4,64),.02,.25,torch.float32),
             mz=rand((2,4,64),.02,1.,torch.float32), mo=rand((2,4,64),.02,.25,torch.float32))
    with torch.no_grad():
        variation = torch.linspace(.7, 1.3, x['dw'].numel(), device='cuda').reshape(dt)
        x['adt'].mul_(variation)
        x['dw'].mul_(variation.flip(-1))
        x['dp'].mul_(1.8-variation)
        if d_only:
            for key in ('q', 'k', 'qb', 'kb'):
                x[key].zero_()
    if tied:
        x['dp'] = x['dw']
    return x


def measure(source, backend, go=None, native=False):
    from experiments.mamba3_three_time.fixtures import clone_inputs, kernel_output, scalar_loss
    x = clone_inputs(source, float_reference=backend == 'reference')
    # Reference receives exactly the same quantized source values, only promoted to fp32.
    if any(not torch.equal(source[k].double(), v.double()) for k, v in x.items()):
        raise ValueError('Reference input requantization')
    y = kernel_output(x, 'MIMO', official=backend == 'official', reference=backend == 'reference', native=native)
    loss = scalar_loss(y)
    incoming = torch.autograd.grad(loss, y, retain_graph=True)[0] if go is None else go.to(y.dtype)
    seen, unique = set(), {}
    for k, v in x.items():
        if id(v) not in seen:
            unique[k] = v
            seen.add(id(v))
    grads = torch.autograd.grad(y, tuple(unique.values()), incoming, allow_unused=True)
    result = dict(output=y.detach(), loss=loss.detach())
    result.update({'gradient:' + k: g.detach() if g is not None else None for k, g in zip(unique, grads)})
    return result
