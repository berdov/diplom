"""Diagnostic arithmetic only; never replaces historical acceptance profiles."""

import math
import torch
from .evidence import compare, tensor_records


def packed(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                values=tensor.detach().double().cpu().tolist())


def cotangent(shape, kind, device, dtype=torch.bfloat16):
    index = torch.arange(math.prod(shape), device=device, dtype=torch.float64)
    value = 1 + index.remainder(7) / 7
    if kind == "signed":
        value = value * torch.where(index.remainder(2) == 0, 1., -1.)
    elif kind != "nonnegative":
        raise ValueError(kind)
    normalized = value / value.norm()
    result = normalized.reshape(shape).to(dtype)
    return result, dict(kind=kind, rule="(1 + i%7/7), alternating signs or nonnegative; L2=1 before bf16 cast",
        pre_cast_l2=normalized.norm().item(), actual_l2=result.double().norm().item(),
        independent_of_outputs=True, tensor=tensor_records({"grad_output": result}))


def loss_report(actual, reference, profile):
    weights = torch.linspace(.1, .9, actual.numel(), device=actual.device,
                             dtype=torch.float32).reshape(actual.shape).double()
    y, a = reference.detach().double(), actual.detach().double()
    e = a-y
    terms = lambda v: dict(weighted=(weights*v).mean().item(), quadratic=(.03*v.square().mean()).item(),
        positive_weighted=(weights*v).clamp_min(0).mean().item(),
        negative_weighted=(weights*v).clamp_max(0).mean().item(),
        mean_absolute_weighted=(weights*v).abs().mean().item())
    ta, tr = terms(a), terms(y)
    la, lr = ta["weighted"]+ta["quadratic"], tr["weighted"]+tr["quadratic"]
    delta = abs(la-lr)
    bound = ((weights.abs()*e.abs()).mean() + .03*(2*y.abs()*e.abs()+e.square()).mean()).item()
    from .fixtures import scalar_loss
    old_a, old_r = scalar_loss(actual), scalar_loss(reference)
    old_delta = abs(old_a.item()-old_r.item())
    return dict(actual_terms_fp64=ta, reference_terms_fp64=tr, loss_actual_fp64=la, loss_reference_fp64=lr,
        absolute_error_fp64=delta, pure_relative_error_fp64=None if lr==0 else delta/abs(lr),
        reference_zero=lr==0, output_norm_actual=a.norm().item(), output_norm_reference=y.norm().item(),
        output_difference_norm=e.norm().item(), propagation_bound_fp64=bound,
        bound_explains_fp64_delta=delta <= bound + torch.finfo(torch.float64).eps*(abs(la)+abs(lr)+bound),
        legacy_loss_actual=old_a.item(), legacy_loss_reference=old_r.item(), legacy_absolute_error=old_delta,
        legacy_pure_relative_error=None if old_r.item()==0 else old_delta/abs(old_r.item()),
        legacy_reduction_rounding_difference=abs((old_a.item()-old_r.item())-(la-lr)),
        mixed_only=compare(old_a,old_r,atol=profile["atol"],rtol=profile["rtol"]),
        legacy_with_norm_cap=compare(old_a,old_r,atol=profile["atol"],rtol=profile["rtol"],
                                     relative_norm_limit=profile["relative_norm_limit"]),
        acceptance_changed=False)


def d_oracles(values, grad_output):
    """D is the skip weight, not DT_write/phase. No candidate gradient is used.

    Pinned forward: V -> Psi V -> +D Psi V -> SiLU(Z Zeta) -> Phi -> sum_r.
    Rounding is discrete: the rounded coefficient is a continuous/STE diagnostic,
    not an exact derivative of the discontinuous quantizer or bitwise tanh oracle.
    """
    v, mv, mz, mo, z = (values[k].detach().double() for k in ("v","mv","mz","mo","z"))
    g = grad_output.detach().double().unsqueeze(-2)
    vp = v.unsqueeze(-2)*mv
    gate = torch.nn.functional.silu(z.unsqueeze(-2)*mz)
    ideal = (g*vp*gate*mo).sum((0,1,3,4))
    bf = lambda t:t.to(torch.bfloat16).double()
    vp_round = bf(v.unsqueeze(-2)*bf(mv))
    half_z = z.unsqueeze(-2)*mz*.5
    gate_round = bf(half_z*half_z.tanh()+half_z)
    rounded = (g*vp_round*gate_round*bf(mo)).sum((0,1,3,4))
    # Independently transcribed pinned bwd arithmetic, including bf16 fragment writes.
    dz = bf(g*bf(mo))
    half_bwd = z.unsqueeze(-2)*bf(mz)*.5
    dz = bf(dz*(half_bwd*half_bwd.tanh()+half_bwd))
    bwd = bf(dz*v.unsqueeze(-2)*mv).sum((0,1,3,4))
    return dict(ideal_fp64=ideal, rounded_forward_coefficient=rounded, pinned_bwd_cast_diagnostic=bwd)
