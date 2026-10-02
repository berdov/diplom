"""Inherited parity plus centered modulation, native isolation and chunk causality."""
import importlib
import traceback
from unittest.mock import patch
import torch
from .. import gate as parent
from . import config as c
from .reuse import bind
from .state import transfer_common
from .checks import modulation_checks
from .provenance import identity,runtime,require_stage
from ..mixer import gap_trap_forward
from experiments.mamba3_three_time import kernels
from experiments.mamba3_timeaware.time_inputs import history_gaps
from experiments.mamba3_mimo_time.records import create,update,Registry,case,accepted_cases,now

_engine=bind(parent,dict(c=c,transfer_common=transfer_common,modulation_checks=modulation_checks),__package__)
fresh,parity,calibration=(_engine[k] for k in ('fresh','parity','calibration'))


def causal(prefix,save):
    net=fresh('centered_gap_trap')
    for cal in net.times.calibrators.values():parent.nonzero(cal)
    with torch.no_grad():net.gap_trap.alpha.fill_(.4)
    items,lens,times=parent.histories(50)
    times=times.detach().requires_grad_()
    changed_items=items.clone();changed_items[0,prefix:]=(changed_items[0,prefix:]+37)%7111+1
    changed_times=times.detach().clone();changed_times[0,prefix:]+=90000000
    saved={};count=len(net.item_embedding._forward_hooks)
    checks=parent.check_prefix(net,(items,lens,times),prefix,1.,None,
        ((changed_items,lens,times),(items,lens,changed_times)),parent.progress_callback(save,saved))
    valid=times.grad is not None and bool(torch.isfinite(times.grad).all())
    checks['timestamp_residual']=parent.residual(times.grad[:1].unsqueeze(-1),prefix) if valid else dict(passed=False)
    save(parent.snapshot(case(checks,hook_phase=saved['phase'])))
    checks['cross_user_timestamp_gradient']=parent.difference(times.grad[1:],torch.zeros_like(times.grad[1:])) if valid else dict(passed=False)
    alpha=net.gap_trap.alpha.grad
    checks['alpha_gradient']=dict(passed=alpha is not None and bool(torch.isfinite(alpha)) and float(alpha.abs())>0,value=float(alpha) if alpha is not None else None)
    phase=saved['phase']
    checks['hook_lifecycle']=dict(passed=phase['hook_removed'] and phase['capture_count']==1 and len(net.item_embedding._forward_hooks)==count
        and all(not s['grad_enabled'] for s in phase['stages'] if s['stage'].startswith('intervention')))
    return parent.snapshot(case(checks,prefix=prefix,chunk=8,hook_phase=phase,legacy_exact_zero_reclassified=False))


NATIVE_KEYS=['only_trap_changed','exact_raw_shift','neutral_positions','short_long_direction',
             'native_forward_backward','rank_chunk_dtype_crop','finite_alpha_gradient','spies_removed','no_test']


def native_isolation(save):
    net=fresh('centered_gap_trap')
    items,lens,_=parent.histories(50,padded=True)
    gaps=torch.full_like(items,4*838393,dtype=torch.float64)
    gaps[:,0]=0;gaps[:,7]=0;gaps[:,8]=838393;gaps[:,9]=4*838393
    times=gaps.cumsum(1)+1.6e12
    g,active=history_gaps(times,items!=0)
    decay,write,phase=net.times(g,active)
    hidden=net.input_norm(net.input_dropout(net.item_embedding(items)))
    layer=net.layers[0];u=layer.norm1(hidden).to(layer.mixer_dtype)
    snapshots=[];calls={'forward':0,'backward':0};lengths=[]
    original=kernels.mimo
    fwd=importlib.import_module('mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd')
    bwd=importlib.import_module('mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd')
    original_f,original_b=fwd.mamba_mimo_forward,bwd.mamba_mimo_bwd_combined
    def capture(*args,**kw):
        snapshots.append(([x.detach().clone() if torch.is_tensor(x) else x for x in args],dict(kw)))
        return original(*args,**kw)
    def forward(*args,**kw):
        calls['forward']+=1;lengths.append(args[0].shape[1]);return original_f(*args,**kw)
    def backward(*args,**kw):
        calls['backward']+=1;return original_b(*args,**kw)
    with patch.object(kernels,'mimo',capture),patch.object(fwd,'mamba_mimo_forward',forward),patch.object(bwd,'mamba_mimo_bwd_combined',backward):
        shift0=net.gap_trap(g,active)
        with torch.no_grad():out0=gap_trap_forward(layer.mixer,u,decay,write,phase,shift0)
        with torch.no_grad():net.gap_trap.alpha.fill_(.5)
        shift=net.gap_trap(g,active)
        out=gap_trap_forward(layer.mixer,u,decay,write,phase,shift)
        weight=torch.linspace(-1.,1.,out.numel(),device=out.device).reshape_as(out)
        (out.float()*weight).sum().backward()
    (a,ka),(b,kb)=snapshots
    ta,tb=a[6],b[6]
    mask=(~active)|(g==838393)
    negative=active&(g<838393);positive=active&(g>838393)
    checks=dict(
        only_trap_changed=dict(passed=ka==kb and all(torch.equal(x,b[i]) for i,x in enumerate(a) if i!=6)),
        exact_raw_shift=dict(passed=torch.equal(tb,(ta.float()+shift[:,None,:]).to(ta.dtype))),
        neutral_positions=dict(passed=torch.equal(ta.permute(0,2,1)[mask],tb.permute(0,2,1)[mask])),
        short_long_direction=dict(passed=bool((tb.permute(0,2,1)[negative]<ta.permute(0,2,1)[negative]).all()) and bool((tb.permute(0,2,1)[positive]>ta.permute(0,2,1)[positive]).all())),
        native_forward_backward=dict(passed=calls['forward']==2 and calls['backward']==1,calls=calls),
        rank_chunk_dtype_crop=dict(passed=a[0].shape[2]==4 and ka['chunk']==8 and ta.dtype==torch.bfloat16 and lengths==[56,56] and out.shape==out0.shape and out.shape[1]==50,native_lengths=lengths),
        finite_alpha_gradient=dict(passed=net.gap_trap.alpha.grad is not None and bool(torch.isfinite(out).all()) and bool(torch.isfinite(net.gap_trap.alpha.grad)) and float(net.gap_trap.alpha.grad.abs())>0,value=float(net.gap_trap.alpha.grad)),
        spies_removed=dict(passed=kernels.mimo is original and fwd.mamba_mimo_forward is original_f and bwd.mamba_mimo_bwd_combined is original_b),
        no_test=dict(passed=True,source='SyntheticCatalog + synthetic histories only',test_evaluation_count=0))
    return case(checks,positions=dict(short=7,reference=8,long=9),same_mixer_input=True)


def main():
    base=identity();require_stage(c.INHERITED,base)
    result=dict(**base,status='RUNNING',scientific_fits=0);create(c.GATE,result)
    specs=c.plan()['required_cases'];registry=Registry(c.GATE,result,specs)
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        result['runtime']=runtime(True);update(c.GATE,result)
        functions=[lambda save:parity('fixed_replay',save),lambda save:parity('centered_gap_trap',save),
                   calibration,native_isolation,lambda save:causal(8,save),lambda save:causal(9,save)]
        if len(specs)!=len(functions):raise ValueError('Gate specification drift')
        for spec,function in zip(specs,functions):
            registry.run(spec,function);torch.cuda.empty_cache()
        if not accepted_cases(result['cases'],specs):raise ValueError('Incomplete required registry')
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    result['finished_at']=now();update(c.GATE,result)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
