"""Device-independent checks for the changed centered feature only."""
import io
import math
from types import SimpleNamespace
import torch
from .modulation import GapTrap, centered_q, attach_projection, REFERENCE_MS as R0
from experiments.mamba3_timeaware.time_inputs import history_gaps
from experiments.mamba3_three_time.confirmation.state import difference

CHECK_KEYS = ['identity','gradient','monotonic','masks','zero_gap','extreme','roundtrip',
              'projection','boundary_update','dtype','direction','odds','antisymmetry','affine','bf16']


def modulation_checks(device, emit):
    module = GapTrap().to(device)
    g = torch.tensor([[0., .1*R0, R0, 10*R0, 1e308]], dtype=torch.float64, device=device)
    active = torch.ones_like(g, dtype=torch.bool)
    q = centered_q(g, active)
    emit('identity', difference(module(g,active), torch.zeros_like(q)))
    objective = -(module(g,active)*q.detach()).sum()
    grad = torch.autograd.grad(objective,module.alpha)[0]
    emit('gradient', dict(passed=bool(torch.isfinite(grad)) and float(grad)<0, value=float(grad), objective='negative shift times detached q'))
    with torch.no_grad(): module.alpha.fill_(.5)
    shift = module(g,active)
    emit('monotonic', dict(passed=bool((q[:,1:]>=q[:,:-1]).all()) and float(q.min())==-1. and float(q.max())<1., q=q.tolist()))
    emit('direction', dict(passed=float(shift[0,0])<0 and float(shift[0,1])<0 and float(shift[0,2])==0 and float(shift[0,3])>0))
    times = torch.tensor([[1.,1.,1.+R0,float('nan')]],dtype=torch.float64,device=device)
    mask = torch.tensor([[True,True,True,False]],device=device)
    gaps,a = history_gaps(times,mask)
    value = module(gaps,a)
    emit('masks',dict(passed=float(value[0,0])==float(value[0,3])==0. and a.tolist()==[[False,True,True,False]]))
    emit('zero_gap',dict(passed=bool(a[0,1]) and float(centered_q(gaps,a,torch.float64)[0,1])==-1. and float(value[0,1])==-.5))
    extreme = g.clone().requires_grad_()
    out = module(extreme,active)
    grads = torch.autograd.grad(out.sum(),(extreme,module.alpha))
    emit('extreme',dict(passed=bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(x).all()) for x in grads) and float(centered_q(extreme,active,torch.float64)[0,-1])<1.))
    ratios = torch.tensor([[.01,.1,.25,.5,1,2,4,10,100]],dtype=torch.float64,device=device)
    a = torch.ones_like(ratios,dtype=torch.bool)
    qc = centered_q(R0*ratios,a,torch.float64)
    emit('antisymmetry',difference(qc,-centered_q(R0/ratios,a,torch.float64)))
    emit('affine',difference(qc,2*ratios/(1+ratios)-1))
    raw = torch.tensor([[-2.],[0.],[2.]],device=device,dtype=torch.float64)
    p,pp = raw.sigmoid(),(raw+shift.double()).sigmoid()
    emit('odds',difference((pp/(1-pp))/(p/(1-p)),shift.double().exp().expand_as(pp)))
    stream=io.BytesIO();torch.save(module.state_dict(),stream);stream.seek(0)
    twin=GapTrap().to(device);twin.load_state_dict(torch.load(stream,map_location=device,weights_only=True))
    emit('roundtrip',difference(module(g,active),twin(g,active)))
    other=torch.nn.Parameter(torch.tensor(2.,device=device));before=other.detach().clone()
    opt=torch.optim.Adam([module.alpha,other],lr=.1)
    hook=attach_projection(SimpleNamespace(gap_trap_mode='centered_gap_trap',gap_trap=module),opt)
    try:
        values=[]
        for start,gradient in ((0.,1.),(1.,-1.)):
            with torch.no_grad():module.alpha.fill_(start)
            opt.state.clear();opt.zero_grad(set_to_none=True);module.alpha.grad=torch.tensor(gradient,device=device)
            opt.step();values.append(float(module.alpha))
        emit('projection',dict(passed=values==[0.,1.] and torch.equal(other,before),values=values))
        with torch.no_grad():module.alpha.zero_()
        opt.state.clear();opt.zero_grad(set_to_none=True)
        (-(module(g,active)*q.detach()).sum()).backward();opt.step()
        emit('boundary_update',dict(passed=0.<float(module.alpha)<=1.,after=float(module.alpha)))
    finally:hook.remove()
    raw=torch.tensor([[-2.],[0.],[2.]],dtype=torch.bfloat16,device=device)
    with torch.no_grad():module.alpha.zero_()
    modified=(raw.float()+module(g,active)).to(raw.dtype)
    emit('dtype',dict(passed=modified.dtype==raw.dtype and torch.equal(modified,raw.expand_as(modified))))
    small=(q*.0005).expand(3,-1);effective=(raw.float()+small).to(raw.dtype).float()-raw.float()
    emit('bf16',dict(passed=bool(torch.isfinite(effective).all()) and bool(((effective==0)&(small!=0)).any()),
                     shift_before_cast=small.tolist(),effective_after_cast=effective.tolist(),kernel_dtype='torch.bfloat16'))
