"""Targeted MIMO routing, gradient separation, and causality gate."""
import traceback
import torch
from . import config as c
from .checks import fresh,routing,nonzero,temporal_aliases
from .state import transfer_common
from .provenance import identity,runtime,require_stage
from experiments.mamba3_mimo_time.records import create,update,Registry,case,accepted_cases,now
from experiments.mamba3_mimo_time.admission import histories,model_measure
from experiments.mamba3_mimo_time.provenance import assert_upstream
from experiments.mamba3_mimo_time.prefix_checks import check_prefix
from experiments.mamba3_mimo_time.numerics import residual
from experiments.mamba3_head_timescales.progress import progress_callback,snapshot
from experiments.mamba3_three_time.confirmation.state import difference,rng_record


def parity(specific,nontrivial,save):
    a=fresh('shared_layers','cuda',historical=not specific)
    if nontrivial:
        for cal in a.times.calibrators.values():nonzero(cal)
    b=fresh('layer_specific' if specific else 'shared_layers','cuda');mapping=transfer_common(a,b)
    assert_upstream(a);assert_upstream(b)
    # Training mode also checks identical dropout draws for common execution.
    a.train();b.train();data=histories(50,padded=True)
    x=model_measure(a,data);rng_a=rng_record();y=model_measure(b,data);rng_b=rng_record();checks={}
    def emit(k,v):checks[k]=v;save(case(checks,state_mapping=mapping,nonzero_temporal_weights=nontrivial))
    for k in ('output','loss','input_gradient'):emit(k,difference(y.get(k),x.get(k)))
    for name in c.plan()['common_parameter_keys']:
        expected=x['gradient:'+name]
        actual=y['gradient:'+name]
        if specific and name.startswith('times.'):actual=actual+y['gradient:layer1_times.'+name[len('times.'):]]
        emit('gradient:'+name,difference(actual,expected))
    emit('dropout_rng',dict(passed=rng_a==rng_b))
    emit('initial_output_bitwise',dict(passed=checks['output']['bitwise_equal'] and checks['loss']['bitwise_equal']))
    if specific:
        emit('independent_storage',dict(passed=temporal_aliases(b)))
        for label,prefix in [('layer0','times.'),('layer1','layer1_times.')]:
            values=[v for k,v in y.items() if k.startswith('gradient:'+prefix)]
            emit(label+'_finite_gradients',dict(passed=all(v is not None and bool(torch.isfinite(v).all()) for v in values)))
            # Last layer has immediate gradients; zero first-layer gradients at exact init are expected.
            emit(label+'_nonzero_gradient',dict(passed=all(bool(y['gradient:'+prefix+'calibrators.'+m+'.last.weight'].abs().sum()>0) for m in ('decay','scan'))))
    return case(checks,state_mapping=mapping,nonzero_temporal_weights=nontrivial)


def routing_case(save):
    net=fresh('layer_specific','cuda');assert_upstream(net);checks={}
    def emit(k,v):checks[k]=v;save(case(checks))
    routing(net,histories(50,padded=True),emit)
    return case(checks)


def causal(prefix,save):
    net=fresh('layer_specific','cuda')
    for i,times in enumerate(net.temporal_sets()):
        for cal in times.calibrators.values():
            nonzero(cal)
            with torch.no_grad():cal.last.bias.add_(i*.1)
    items,lens,times=histories(50);times=times.detach().requires_grad_();data=(items,lens,times)
    changed_items=items.clone();changed_items[0,prefix:]=(changed_items[0,prefix:]+37)%7111+1
    changed_times=times.detach().clone();changed_times[0,prefix:]+=90000000
    saved={};persist=progress_callback(save,saved)
    checks=check_prefix(net,data,prefix,1.,None,((changed_items,lens,times),(items,lens,changed_times)),persist)
    valid=times.grad is not None and bool(torch.isfinite(times.grad).all())
    checks['timestamp_residual']=residual(times.grad[:1].unsqueeze(-1),prefix) if valid else dict(passed=False,reason='Missing timestamp gradient')
    checks['cross_user_timestamp_gradient']=difference(times.grad[1:],torch.zeros_like(times.grad[1:])) if valid else dict(passed=False)
    phase=saved['phase'];checks['hook_lifecycle']=dict(passed=phase['hook_removed'] and phase['capture_count']==1 and all(not s['grad_enabled'] for s in phase['stages'] if s['stage'].startswith('intervention')))
    return snapshot(case(checks,hook_phase=phase))


def main():
    base=identity();require_stage(c.INHERITED,base);result=dict(**base,status='RUNNING',scientific_fits=0)
    create(c.GATE,result);specs=c.plan()['required_cases'];registry=Registry(c.GATE,result,specs)
    try:
        torch.backends.cuda.matmul.allow_tf32=False;result['runtime']=runtime(True);update(c.GATE,result)
        functions=[lambda save:parity(False,True,save),lambda save:parity(True,False,save),lambda save:parity(True,True,save),routing_case,
                   lambda save:causal(8,save),lambda save:causal(9,save)]
        for spec,fn in zip(specs,functions):registry.run(spec,fn);torch.cuda.empty_cache()
        if not accepted_cases(result['cases'],specs):raise ValueError('Targeted gate incomplete')
        result['status']='PASS'
    except BaseException as exc:result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    result['finished_at']=now();update(c.GATE,result)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
