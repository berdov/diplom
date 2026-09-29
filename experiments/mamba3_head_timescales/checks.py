"""Independent float64 calibrator reference and small deterministic checks."""
import copy
import io
import math
import torch
from torch.nn import functional as F
from .calibrator import LearnedReference
from experiments.mamba3_timeaware.time_inputs import TimeCalibrator
from experiments.mamba3_three_time.confirmation.state import difference


def nonzero(cal):
    with torch.no_grad():
        cal.first.weight.copy_(torch.linspace(-.2,.3,16,device=cal.reference.device).reshape(16,1))
        cal.first.bias.copy_(torch.linspace(.1,.2,16,device=cal.reference.device))
        cal.last.weight.copy_(torch.linspace(-.11,.17,32,device=cal.reference.device).reshape(2,16))
        cal.last.bias.copy_(torch.tensor([.07,-.03],device=cal.reference.device))
    return cal


def fresh_cal(variant,device='cpu'):
    old=nonzero(TimeCalibrator(2,838393).double().to(device))
    return LearnedReference(old,variant)


def reference(gaps,active,alpha,first_w,first_b,last_w,last_b):
    # Independent direct-ratio formula, used only on moderate positive fixtures.
    R=838393*torch.exp(math.log(4)*torch.tanh(alpha))
    z=torch.log1p(gaps.unsqueeze(-1)/R)
    z=z.expand(*gaps.shape,2)
    hidden=F.silu(z.unsqueeze(-1)*first_w[:,0]+first_b)
    raw=(hidden*last_w).sum(-1)+last_b
    s=torch.exp(math.log(2)*torch.tanh(raw))
    return torch.where(active.unsqueeze(-1),s,torch.ones_like(s))


def calibrator_checks(variant,device,emit):
    cal=fresh_cal(variant,device)
    with torch.no_grad():cal.alpha.fill_(.23)
    g=torch.linspace(.01,4,15,device=device,dtype=torch.float64).reshape(3,5)*838393
    active=torch.ones_like(g,dtype=torch.bool)
    weight=torch.linspace(.1,.9,30,device=device,dtype=torch.float64).reshape(3,5,2)
    names=['alpha','first.weight','first.bias','last.weight','last.bias']
    params=[dict(cal.named_parameters())[k] for k in names]
    leaves=[p.detach().clone().requires_grad_() for p in params]
    a,b=cal(g,active),reference(g,active,*leaves)
    emit('output',difference(a,b))
    ga=torch.autograd.grad((a*weight).sum(),params)
    gb=torch.autograd.grad((b*weight).sum(),leaves)
    for name,x,y in zip(names,ga,gb):
        emit('alpha_gradient' if name=='alpha' else name,difference(x,y))
    zero=torch.zeros_like(g)
    value=cal(zero,active)
    grad=torch.autograd.grad(value.sum(),cal.alpha)[0]
    emit('zero_alpha_gradient',difference(grad,torch.zeros_like(grad)))
    emit('active_zero',dict(passed=bool((value!=1).any())))
    masked=cal(g,~active)
    grad=torch.autograd.grad(masked.sum(),cal.alpha)[0]
    emit('masked_alpha_gradient',difference(grad,torch.zeros_like(grad)))
    emit('neutral_mask',dict(passed=bool(torch.equal(masked,torch.ones_like(masked))),bitwise_equal=True))
    extreme=torch.tensor([[0,1e-300,1e-10,1,838393,1e308]],device=device,dtype=torch.float64)
    y=cal(extreme,torch.ones_like(extreme,dtype=torch.bool))
    grads=torch.autograd.grad(y.sum(),params)
    emit('finite_extremes',dict(passed=bool(torch.isfinite(y).all()) and all(bool(torch.isfinite(v).all()) for v in grads)))
    emit('scale_bounds',dict(passed=bool(((y>=.5)&(y<=2)).all())))
    ratio=cal.log_reference_ratio().exp()
    emit('reference_bounds',dict(passed=bool(((ratio>=.25)&(ratio<=4)).all())))
    buffer=io.BytesIO();torch.save(cal.state_dict(),buffer);buffer.seek(0)
    twin=fresh_cal(variant,device);twin.load_state_dict(torch.load(buffer,map_location=device,weights_only=True))
    emit('roundtrip',difference(cal(g,active),twin(g,active)))
    original=cal(g,active).detach()
    with torch.no_grad():cal.alpha[0].add_(.3)
    changed=cal(g,active).detach()
    if variant=='head_tau':
        check=difference(changed[...,1],original[...,1])
        check['passed']=check['passed'] and bool((changed[...,0]!=original[...,0]).any())
    else:
        check=dict(passed=all(bool((changed[...,h]!=original[...,h]).any()) for h in (0,1)))
    emit('head_locality',check)


def tied_checks(device,emit):
    a,b=fresh_cal('shared_tau',device),fresh_cal('head_tau',device)
    with torch.no_grad():a.alpha.fill_(.31);b.alpha.fill_(.31)
    g=torch.linspace(.01,7,15,device=device,dtype=torch.float64).reshape(3,5)*838393
    active=torch.ones_like(g,dtype=torch.bool)
    weight=torch.linspace(.1,.9,30,device=device,dtype=torch.float64).reshape(3,5,2)
    x,y=a(g,active),b(g,active)
    emit('output',difference(x,y))
    pa,pb=dict(a.named_parameters()),dict(b.named_parameters())
    names=['alpha','first.weight','first.bias','last.weight','last.bias']
    ga=torch.autograd.grad((x*weight).sum(),[pa[k] for k in names])
    gb=torch.autograd.grad((y*weight).sum(),[pb[k] for k in names])
    emit('alpha_sum',difference(ga[0],gb[0].sum().reshape(1)))
    for name,left,right in zip(names[1:],ga[1:],gb[1:]):emit(name,difference(left,right))
