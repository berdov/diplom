"""Nine targeted cases; partial evidence is durable before any assertion."""
import traceback
import torch
from . import config as c
from .provenance import identity,runtime,require_stage
from .checks import calibrator_checks,tied_checks,nonzero
from .state import transfer_common
from experiments.mamba3_mimo_time.records import create,update,Registry,case,accepted_cases,now
from experiments.mamba3_mimo_time.provenance import assert_upstream
from experiments.mamba3_mimo_time.admission import histories,model_measure
from experiments.mamba3_mimo_time.prefix_checks import check_prefix
from experiments.mamba3_mimo_time.numerics import residual
from experiments.mamba3_three_time.confirmation.state import difference
from experiments.mamba3_three_time.initialization import seed_all


def fresh(variant,old=False):
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.config import SyntheticCatalog
    from .model import HeadTimescaleMamba3Rec
    cfg=Config(model=ThreeTimeMamba3Rec,config_dict=c.settings(variant,'cuda'))
    seed_all(314159)
    model=(ThreeTimeMamba3Rec if old else HeadTimescaleMamba3Rec)(cfg,SyntheticCatalog()).cuda().eval()
    assert_upstream(model)
    return model


def parity(variant,save):
    old=fresh('fixed',old=True)
    for cal in old.times.calibrators.values():nonzero(cal)
    new=fresh(variant);mapping=transfer_common(old,new)
    data=histories(50,padded=True)
    a,b=model_measure(new,data),model_measure(old,data)
    checks={}
    for k in ['output','loss','input_gradient']+['gradient:'+n for n in c.plan()['common_parameter_keys']]:
        checks[k]=difference(a.get(k),b.get(k))
        save(case(checks,common_state_mapping=mapping,alpha_zero=True,nonzero_MLP=True))
    return case(checks,common_state_mapping=mapping,alpha_zero=True,nonzero_MLP=True)


def calibration(variant,save):
    checks={}
    def emit(k,v):
        checks[k]=v;save(case(checks))
    calibrator_checks(variant,'cuda',emit)
    return case(checks)


def tied(save):
    checks={}
    def emit(k,v):checks[k]=v;save(case(checks))
    tied_checks('cuda',emit)
    return case(checks)


def causal(variant,save):
    net=fresh(variant)
    for cal in net.times.calibrators.values():
        nonzero(cal)
        if hasattr(cal,'alpha'):
            with torch.no_grad():cal.alpha.copy_(torch.linspace(-.25,.35,cal.alpha.numel(),device='cuda'))
    items,lens,times=histories(50)
    times=times.detach().requires_grad_();data=(items,lens,times)
    changed_items=items.clone();changed_items[0,7:]=(changed_items[0,7:]+37)%7111+1
    changed_times=times.detach().clone();changed_times[0,7:]+=90000000
    saved={}
    def persist(checks,phase):
        saved.update(checks=checks,phase=phase)
        save(case(checks,hook_phase=phase))
    checks=check_prefix(net,data,7,1.,None,((changed_items,lens,times),(items,lens,changed_times)),persist)
    valid_gradient=times.grad is not None and times.grad.shape==times.shape and bool(torch.isfinite(times.grad).all())
    checks['timestamp_residual']=residual(times.grad[:1].unsqueeze(-1),7) if valid_gradient else dict(passed=False,reason='Missing/nonfinite timestamp gradient')
    save(case(checks,hook_phase=saved['phase']))
    checks['cross_user_timestamp_gradient']=difference(times.grad[1:],torch.zeros_like(times.grad[1:])) if valid_gradient else dict(passed=False,reason='Missing/nonfinite timestamp gradient')
    save(case(checks,hook_phase=saved['phase']))
    phase=saved['phase']
    checks['hook_lifecycle']=dict(passed=phase['hook_removed'] and phase['capture_count']==1
                                 and all(not s['grad_enabled'] for s in phase['stages'] if s['stage'].startswith('intervention')))
    return case(checks,hook_phase=phase,legacy_exact_zero_reclassified=False)


def main():
    base=identity()
    require_stage(c.INHERITED,base)
    result=dict(**base,status='RUNNING',scientific_fits=0)
    create(c.GATE,result)
    specs=c.plan()['required_cases'];registry=Registry(c.GATE,result,specs)
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        result['runtime']=runtime(True);update(c.GATE,result)
        functions=[lambda save:parity('fixed',save),lambda save:parity('shared_tau',save),lambda save:parity('head_tau',save),
                   lambda save:calibration('shared_tau',save),lambda save:calibration('head_tau',save),tied,
                   lambda save:causal('fixed',save),lambda save:causal('shared_tau',save),lambda save:causal('head_tau',save)]
        for spec,function in zip(specs,functions):
            registry.run(spec,function)
            torch.cuda.empty_cache()
        if not accepted_cases(result['cases'],specs):raise ValueError('Incomplete required registry')
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    result['finished_at']=now();update(c.GATE,result)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
