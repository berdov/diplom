"""Device-independent new modulation checks; no recommendation data."""
import io
from types import SimpleNamespace
import torch
from .modulation import GapTrap, attach_projection
from experiments.mamba3_timeaware.time_inputs import history_gaps
from experiments.mamba3_three_time.confirmation.state import difference


def modulation_checks(device, emit):
    module=GapTrap().to(device)
    gaps=torch.tensor([[0.,1.,838393.,1e12]],dtype=torch.float64,device=device)
    active=torch.ones_like(gaps,dtype=torch.bool)
    emit('identity',difference(module(gaps,active),torch.zeros_like(gaps,dtype=torch.float32)))
    grad=torch.autograd.grad(module(gaps,active).sum(),module.alpha)[0]
    emit('gradient',dict(passed=bool(torch.isfinite(grad)) and float(grad)>0,initial_alpha_gradient=float(grad)))
    with torch.no_grad():module.alpha.fill_(.5)
    shift=module(gaps,active)
    emit('monotonic',dict(passed=bool((shift[:,1:]>=shift[:,:-1]).all()) and bool(((shift>=0)&(shift<=.5)).all())))
    times=torch.tensor([[1.,1.,5.,float('nan')]],device=device,dtype=torch.float64)
    mask=torch.tensor([[True,True,True,False]],device=device)
    g,a=history_gaps(times,mask);value=module(g,a)
    emit('masks',dict(passed=float(value[0,0])==float(value[0,3])==0. and a.tolist()==[[False,True,True,False]]))
    emit('zero_gap',dict(passed=bool(a[0,1]) and float(value[0,1])==0.))
    extreme=torch.tensor([[0.,1e-300,1e-10,1.,838393.,1e308]],dtype=torch.float64,device=device,requires_grad=True)
    out=module(extreme,torch.ones_like(extreme,dtype=torch.bool))
    grads=torch.autograd.grad(out.sum(),(extreme,module.alpha))
    emit('extreme',dict(passed=bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(x).all()) for x in grads)))
    stream=io.BytesIO();torch.save(module.state_dict(),stream);stream.seek(0)
    twin=GapTrap().to(device);twin.load_state_dict(torch.load(stream,map_location=device,weights_only=True))
    emit('roundtrip',difference(module(gaps,active),twin(gaps,active)))
    other=torch.nn.Parameter(torch.tensor(2.,device=device))
    before=other.detach().clone();model=SimpleNamespace(gap_trap_mode='gap_trap',gap_trap=module)
    opt=torch.optim.Adam([module.alpha,other],lr=.1);hook=attach_projection(model,opt)
    try:
        values=[]
        for start,gradient in ((0.,1.),(1.,-1.)):
            with torch.no_grad():module.alpha.fill_(start)
            opt.state.clear();opt.zero_grad(set_to_none=True);module.alpha.grad=torch.tensor(gradient,device=device)
            opt.step();values.append(float(module.alpha))
        emit('projection',dict(passed=values==[0.,1.] and torch.equal(other,before),values=values))
        with torch.no_grad():module.alpha.zero_()
        opt.state.clear();opt.zero_grad(set_to_none=True)
        (-module(gaps,active).sum()).backward();opt.step()
        emit('boundary_update',dict(passed=0.<float(module.alpha)<=1.,after=float(module.alpha),objective='negative shift sum'))
    finally:hook.remove()
    raw=torch.tensor([-1.,0.,1.],dtype=torch.bfloat16,device=device)
    with torch.no_grad():module.alpha.zero_()
    modified=(raw.float()+module(gaps,active)[0,1:]).to(raw.dtype)
    emit('dtype',dict(passed=modified.dtype==raw.dtype and torch.equal(raw,modified)))
