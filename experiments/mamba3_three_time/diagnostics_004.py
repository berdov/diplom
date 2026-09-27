"""Two drift fixtures and original-arithmetic causality, no scientific data."""

import copy
import torch
from .backends import selected
from .diagnostics_003 import measured, tensor_meta
from .drift_capture import invoke, snapshot_inputs, OUTPUTS
from .diagnostics import prefix_model, frozen_separate
from .fixtures import model, histories, nontrivial
from .initialization import rng_records
from .suites import measure_model, comparisons
from .evidence import case, compare


def trace_metadata(rows):
    return [{k:v for k,v in r.items() if k not in ("inputs","outputs","stages")} |
            dict(inputs=tensor_meta(r["inputs"]), outputs=tensor_meta(r["outputs"]),
                 stages=tensor_meta(r["stages"])) for r in rows]


def replay_dqkv(inputs):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dqkv as original
    from .stable_adt import compute_dqkv as candidate
    fingerprint=tensor_meta(inputs)
    rows={}
    for matched in (False,True):
        a,layout_a=snapshot_inputs(inputs)
        b,layout_b=snapshot_inputs(inputs)
        before_a,before_b=tensor_meta(a),tensor_meta(b)
        old,old_launch=invoke(original,a,matched=matched)
        new,new_launch=invoke(candidate,b,matched=matched)
        unchanged=(tensor_meta(a)==before_a and tensor_meta(b)==before_b and tensor_meta(inputs)==fingerprint)
        if not unchanged:raise RuntimeError('dqkv replay mutated immutable input snapshots')
        checks={k:compare(x,y) for k,x,y in zip(OUTPUTS,old,new) if x is not None or y is not None}
        rows['matched' if matched else 'autotuned']=dict(required=False,status='MEASURED',checks=checks,
            original_launches=old_launch,candidate_launches=new_launch,
            original_layout=layout_a,candidate_layout=layout_b,immutable_inputs_verified=True,
            unchanged_outputs_differ=[k for k,v in checks.items() if k!='dADT' and v.get('max_abs',0)!=0])
    return dict(input_tensors=fingerprint,comparisons=rows)


def trace_smoke():
    net = model("SISO","dual").train()
    nontrivial(net.times.calibrators)
    data = histories(50)
    with selected("upstream"):
        plain = measure_model(net,data)
    rng_plain = rng_records()
    traced, rows, _, _ = measured(net,data,"upstream")
    checks = {k:compare(traced[k],plain[k],atol=0,rtol=0) for k in plain}
    checks.update(two_invocations=dict(passed=len(rows)==2 and len({r["invocation_id"] for r in rows})==2),
        forward_backward_counts=dict(passed=bool(rows) and all(r["forward_calls"]==r["backward_calls"]==1 for r in rows)),
        owned_stages=dict(passed=bool(rows) and all(r["stages"] for r in rows)),
        rng_unchanged=dict(passed=rng_plain==rng_records()),
        launch_metadata=dict(passed=bool(rows) and all(r["launches"] and
            r["launches"][0]["config"]["num_warps"] is not None for r in rows)),
        immutable_inputs=dict(passed=bool(rows) and all(v["storage_independent"] for r in rows for v in r["input_layouts"].values())))
    row = case("trace_smoke","trace off/on, same weights/data/dropout RNG",checks)
    row.update(backend="upstream",traces=trace_metadata(rows),rng_plain=rng_plain,rng_traced=rng_records())
    return row


def drift(mode, training, length):
    net = model("SISO",mode).train(training)
    if mode != "base":nontrivial(net.times.calibrators)
    data = histories(length)
    original, trace, _, events = measured(net,data,"upstream")
    repeat, repeat_trace, _, _ = measured(net,data,"upstream")
    repeated = {k:compare(repeat[k],original[k],atol=0,rtol=0) for k in original}
    replay = [dict(invocation_id=r["invocation_id"],input_layouts=r["input_layouts"],
                   **replay_dqkv(r["inputs"])) for r in trace]
    hybrid = any(r["comparisons"]["autotuned"]["unchanged_outputs_differ"] for r in replay)
    variants = {}
    for backend in ["stable_adt"] + (["diagnostic_hybrid"] if hybrid else []):
        measured_value, traces, _, measured_events = measured(net,data,backend)
        by_name = dict(measured_events)
        stages = [dict(stage=name+"/"+k,**compare(by_name.get(name,{}).get(k),v))
                  for name,tensors in events for k,v in tensors.items()]
        first = next((r["stage"] for r in stages if not r["passed"]),None)
        variants[backend] = dict(required=False,backend=backend,checks=comparisons(measured_value,original),
            stages=stages,traces=trace_metadata(traces),first_measured_divergence=first or "NONE_MEASURED",
            unique_mathematical_cause="UNKNOWN",production_authorized=False)
    checks = dict(original_repeat_exact=dict(passed=all(v["passed"] for v in repeated.values())),
        trace_count=dict(passed=len(trace)==len(repeat_trace)==len(replay)==2),
        captured_layout=dict(passed=all(v["layout_preserved"] for r in trace for v in r["input_layouts"].values())),
        variants_captured=dict(passed=all(len(v["traces"])==2 for v in variants.values())))
    return dict(case("drift","original vs repeat and rejected ADT-only; fixed dqkv replay",checks),
        backend="upstream vs stable_adt",mode=mode,training=training,length=length,
        original_repeat=dict(required=False,checks=repeated),original_traces=trace_metadata(trace),
        identical_input_replays=replay,variants=variants,
        interpretation="First recorded divergence is not proof of a unique cause; no backend promotion.")


def full_output(net,data,oracle=None,old=False):
    if not old:return net.encode_sequence(*data,oracle=oracle)
    captured=[]
    hook=net.output_norm.register_forward_hook(lambda _m,_a,y:captured.append(y))
    try:
        net(*data)
        return captured[0]
    finally:hook.remove()


def causality(arch,mode,length,prefix,multiplier):
    net=model(arch,mode).eval()
    if mode!='base':nontrivial(net.times.calibrators)
    data=histories(length)
    local=prefix_model(net,data,prefix,multiplier,variant="upstream")
    items,lens,times=data
    changed_items,changed_times=items.clone(),times.clone()
    changed_items[0,prefix:]=(changed_items[0,prefix:]+37)%7111+1
    changed_times[0,prefix:]+=90000000
    variants={"local_"+mode:dict(backend="upstream",report=local["report"])}
    checks={}
    targets=[("local_"+mode,net,None,False)]
    if mode=='base':targets.append(("official_base",copy.deepcopy(net),"official_base",False))
    elif mode=='dual':
        targets.append(("frozen_separate" if arch=='SISO' else "official_dual",
                        frozen_separate(net) if arch=='SISO' else copy.deepcopy(net),
                        None if arch=='SISO' else "official_dual",arch=='SISO'))
    for name,target,oracle,old in targets:
        with torch.no_grad(),selected("upstream"):
            base=full_output(target,data,oracle,old)
            altered=full_output(target,(changed_items,lens,changed_times),oracle,old)
        checks[name+":suffix_invariance"]=compare(base[:,:prefix],altered[:,:prefix])
        checks[name+":cross_user"]=compare(base[1],altered[1])
        if name.startswith('local_'):continue
        other=prefix_model(target,data,prefix,multiplier,oracle=oracle,old=old)
        variants[name]=dict(backend=name,report=other["report"])
        checks[name+":output_parity"]=compare(local["output"],other["output"])
        checks[name+":input_gradient_parity"]=compare(local["gradient"],other["gradient"])
    for name,v in variants.items():
        checks[name+":residual_measured"]=dict(passed=v['report']['embedding_output_gradient']['status']=='MEASURED')
        v['required']=False
        v['legacy_exact_zero_status']='PASS' if v['report']['embedding_output_gradient']['exact_zero'] else 'FAIL'
    return dict(case("causality","prefix interventions and positional gradients; upstream arithmetic",checks),
        backend="upstream",mode=mode,length=length,prefix=prefix,loss_multiplier=multiplier,
        variants=variants,legacy_exact_zero={k:v['legacy_exact_zero_status'] for k,v in variants.items()},
        residual_acceptance="NOT_REDEFINED",training_authorized=False)
