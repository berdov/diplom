"""Deterministic synthetic catalog/kernel tensors; no dataset access."""

from collections import OrderedDict
import torch
from .config import construct_config, SyntheticCatalog
from .initialization import seed_all


def model(architecture, mode, device="cuda"):
    from .model import ThreeTimeMamba3Rec
    config = construct_config(architecture, mode, device)
    seed_all()
    return ThreeTimeMamba3Rec(config, SyntheticCatalog()).to(device)


def nontrivial(calibrators):
    with torch.no_grad():
        for index, module in enumerate(calibrators.values()):
            module.last.weight.copy_(torch.linspace(-.06, .08, module.last.weight.numel(),
                device=module.last.weight.device).reshape_as(module.last.weight) * (index+1))
            module.last.bias.fill_(.04*(index+1))


def histories(length, batch=2, device="cuda"):
    items = (torch.arange(batch*length, device=device).reshape(batch, length)*7 % 7111)+1
    lengths = torch.full((batch,), length, device=device, dtype=torch.long)
    if batch > 1:
        lengths[-1] = max(1, length-7)
        items[-1, lengths[-1]:] = 0
    gaps = (torch.arange(batch*length, device=device).reshape(batch, length) % 7).double()*300001
    times = 1.6e12 + gaps.cumsum(1)
    return items, lengths, times


def kernel_inputs(architecture, length=7, batch=1, device="cuda", dtype=torch.float32,
                  tiny=False, tied=False):
    seed_all()
    mimo = architecture == "MIMO"
    n, p, h, rank = (8, 4, 2, 4) if tiny else (128, 64, 2, 4)
    low = dtype if device == "cpu" else torch.bfloat16
    def rand(shape, scale=.15, shift=0., kind=low):
        return (torch.randn(shape, device=device, dtype=torch.float64)*scale+shift).to(kind).requires_grad_(True)
    qshape = (batch, length, rank, 1, n) if mimo else (batch, length, 1, n)
    bshape = (h, rank, n) if mimo else (h, n)
    dtshape = (batch, h, length)
    x = OrderedDict(q=rand(qshape), k=rand(qshape), v=rand((batch,length,h,p)),
        adt=rand(dtshape, .005, -.12, dtype), dw=rand(dtshape,.006,.23,dtype),
        dp=rand(dtshape,.01,.31,dtype), trap=rand(dtshape,.3), qb=rand(bshape,.04,.1,dtype),
        kb=rand(bshape,.04,.1,dtype), angles=rand((batch,length,h,n//4),.03,.09, dtype if mimo else low),
        d=rand((h,), .02, .7, dtype), z=rand((batch,length,h,p),.1,.4))
    if mimo:
        x.update(mv=rand((h,rank,p),.02,.25,dtype), mz=rand((h,rank,p),.02,1.,dtype),
                 mo=rand((h,rank,p),.02,.25,dtype))
    if tied:
        x["dp"] = x["dw"]
    return x


def clone_inputs(values, *, float_reference=False):
    copies, seen = OrderedDict(), {}
    for key, value in values.items():
        if id(value) not in seen:
            seen[id(value)] = value.detach().to(torch.float32 if float_reference else value.dtype).clone().requires_grad_(True)
        copies[key] = seen[id(value)]
    return copies


def scalar_loss(output):
    weights = torch.linspace(.1, .9, output.numel(), device=output.device, dtype=torch.float32).reshape(output.shape)
    return (output.float()*weights).mean() + .03*output.float().square().mean()


def kernel_output(values, architecture, *, official=False, reference=False):
    if reference:
        from .reference import recurrence
        return recurrence(**values)
    if official:
        if values["dw"] is not values["dp"]:
            raise ValueError("Official oracle requires tied slots")
        x = dict(values)
        x.pop("dp")
        q, k, v, adt, dt, trap, qb, kb, angles, d, z = (x[k] for k in
            ("q", "k", "v", "adt", "dw", "trap", "qb", "kb", "angles", "d", "z"))
        if architecture == "SISO":
            from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
            return mamba3_siso_combined(q, k, v, adt, dt, trap, qb, kb, angles, d, z, chunk_size=64)
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
        return mamba3_mimo(q,k,v,adt,dt,trap,qb,kb,x["mv"],x["mz"],x["mo"],angles,d,z,16,4,v.dtype)
    from .kernels import siso, mimo
    return (siso if architecture == "SISO" else mimo)(**values)


def kernel_measure(values, architecture, **kwargs):
    output = kernel_output(values, architecture, **kwargs)
    loss = scalar_loss(output)
    loss.backward()
    result = {"output":output.detach(), "loss":loss.detach()}
    seen = set()
    for key, value in values.items():
        if id(value) not in seen:
            result["gradient:"+key] = value.grad.detach().clone() if value.grad is not None else None
            seen.add(id(value))
    return result
