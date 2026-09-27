"""Fixed attempt003 comparisons; mismatches remain measured, not reinterpreted."""

from contextlib import contextmanager
import torch
from .backends import selected
from .drift_capture import capturing, invoke, OUTPUTS
from .evidence import case, compare, tensor_records
from .fixtures import model, histories, nontrivial, kernel_inputs, clone_inputs, kernel_measure
from .suites import measure_model, comparisons
from .diagnostics import prefix_model, tail_stats


@contextmanager
def module_gradients(net, rows, events):
    hooks = []
    def capture(name):
        def forward(_m, _args, output):
            def backward(grad):
                rows[name] = grad.detach().clone()
                events.append(("model/"+name,{"output_gradient":rows[name]}))
            if output.requires_grad:
                output.register_hook(backward)
        return forward
    for name, module in net.named_modules():
        if name in ("item_embedding", "input_norm", "output_norm") or name.endswith(
                (".in_proj", ".norm1", ".norm2", ".dropout1", ".dropout2")):
            hooks.append(module.register_forward_hook(capture(name)))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def measured(net, data, backend):
    trace, outer, events = [], {}, []
    with selected(backend), capturing(trace,events), module_gradients(net, outer,events):
        final = measure_model(net, data)
    return final, trace, outer, events


def tensor_meta(values):
    records = tensor_records({k:v for k,v in values.items() if torch.is_tensor(v)})
    for key, record in records.items():
        record["stride"] = list(values[key].stride())
    return records


def replay_dqkv(inputs):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dqkv as original
    from .stable_adt import compute_dqkv as candidate
    rows = {}
    for matched in (False, True):
        a, launch_a = invoke(original, inputs, matched=matched)
        b, launch_b = invoke(candidate, inputs, matched=matched)
        checks = {k:compare(x,y) for k,x,y in zip(OUTPUTS,a,b) if x is not None or y is not None}
        rows["matched" if matched else "autotuned"] = dict(
            required=False, status="MEASURED", checks=checks,
            original_launches=launch_a, candidate_launches=launch_b,
            unchanged_outputs_differ=[k for k,v in checks.items() if k != "dADT" and v.get("max_abs",0) != 0])
    return dict(input_tensors=tensor_meta(inputs), comparisons=rows)


def drift(mode, training, length):
    net = model("SISO", mode).train(training)
    if mode != "base":
        nontrivial(net.times.calibrators)
    data = histories(length)
    original, traces, outer, events = measured(net, data, "upstream")
    repeat, repeat_traces, repeat_outer, repeat_events = measured(net, data, "upstream")
    repeat_checks = comparisons(repeat, original)
    # Reproducibility is established before attributing differences to a backend.
    repeat_exact = {k:compare(repeat[k],original[k],atol=0,rtol=0) for k in original}
    result = dict(mode=mode, training=training, length=length, loss="frozen scalar_loss",
        seed=2026, original_reproducibility=case("original_repeat", "same weights/data/dropout RNG", repeat_exact),
        original_launches=[t["launches"] for t in traces], paired_dqkv=[], variants={})
    if not all(c["passed"] for c in repeat_checks.values()):
        return dict(case("drift", "original repetition", repeat_checks), **result)
    for layer, trace in enumerate(traces):
        replay = replay_dqkv(trace["inputs"])
        replay["reverse_layer_order"] = layer
        result["paired_dqkv"].append(replay)
    extra = any(r["comparisons"]["autotuned"]["unchanged_outputs_differ"] for r in result["paired_dqkv"])
    variants = ["stable_adt", "stable_scan"] + (["diagnostic_hybrid"] if extra else [])
    candidate_checks = {}
    for backend in variants:
        final, local, local_outer, local_events = measured(net, data, backend)
        checks = comparisons(final, original)
        checks["unchanged_forward"] = compare(final["output"],original["output"],atol=0,rtol=0)
        stages = []
        matching = [name for name,_ in events] == [name for name,_ in local_events]
        checks["backward_event_order"] = dict(passed=matching)
        by_name = dict(local_events)
        for name,tensors in events:
            for key,tensor in tensors.items():
                stages.append(dict(stage=name+"/"+key,**compare(by_name.get(name,{}).get(key),tensor)))
        first = next((r["stage"] for r in stages if not r["passed"]), None)
        result["variants"][backend] = dict(required=backend == "stable_adt",
            status="MEASURED", checks=checks, stages=stages,
            first_structural_divergence_in_recorded_reverse_order=first,
            interpretation="Recorded stages are causal checkpoints, not proof of a unique mathematical cause.",
            launches=[r["launches"] for r in local],
            dqkv_inputs=[tensor_meta(r["inputs"]) for r in local],
            extra_cost="Two complete dqkv calls per layer; diagnostic only" if backend == "diagnostic_hybrid" else None)
        if backend == "stable_adt":
            candidate_checks = checks
            candidate_checks["candidate_launch_metadata"] = dict(passed=len(local)==2 and all(
                r["launches"] and r["launches"][0]["config"]["num_warps"] is not None for r in local))
    candidate_checks["original_repeat_exact"] = dict(passed=result["original_reproducibility"]["passed"])
    candidate_checks["trace_coverage"] = dict(passed=len(traces)==len(repeat_traces)==2 and
        all(len(t["launches"])==1 and t["launches"][0]["config"]["num_warps"] is not None for t in traces))
    candidate_checks["required_intermediates"] = dict(passed=all(
        set(t["outputs"])==set(OUTPUTS[:-1]) for t in traces) and
        [name for name,_ in events]==[name for name,_ in repeat_events])
    return dict(case("drift", "ADT-only vs reproduced original", candidate_checks), **result)


def prefix(mode, length, prefix_length, multiplier):
    net = model("SISO", mode).eval()
    if mode != "base":
        nontrivial(net.times.calibrators)
    source = prefix_model(net, histories(length), prefix_length, multiplier)
    new = prefix_model(net, histories(length), prefix_length, multiplier, variant="stable_adt")
    checks = dict(output=compare(source["output"],new["output"],atol=0,rtol=0),
        gradient=compare(new["gradient"],source["gradient"]),
        future_exact_zero=dict(passed=new["report"]["embedding_output_gradient"]["exact_zero"]),
        cross_user_exact_zero=dict(passed=new["report"]["cross_user"]["exact_zero"]))
    row = case("prefix", "original and ADT-only positional input gradients", checks)
    row.update(original=dict(required=False, **source["report"]), candidate=new["report"],
        original_exact_zero=source["report"]["embedding_output_gradient"]["exact_zero"],
        length=length, prefix=prefix_length, loss_multiplier=multiplier, mode=mode)
    return row


def paired_reference(arch, length, tied, profile):
    values = kernel_inputs(arch,length=length,tied=tied)
    if not tied:
        with torch.no_grad():
            variation = torch.linspace(.7,1.3,values["dw"].numel(),device="cuda").reshape_as(values["dw"])
            values["adt"].mul_(variation)
            values["dw"].mul_(variation.flip(-1))
            values["dp"].mul_(1.8-variation)
    with selected("upstream"):
        old = kernel_measure(clone_inputs(values),arch,official=tied)
    reference = kernel_measure(clone_inputs(values,float_reference=True),arch,reference=True)
    with selected("stable_adt" if arch == "SISO" else "upstream"):
        new = kernel_measure(clone_inputs(values),arch)
    old_checks, new_checks = comparisons(old,reference,profile), comparisons(new,reference,profile)
    row = case("paired_reference", "same quantized fixture: original/candidate/FP32 recurrence", {
        **{"original:"+k:v for k,v in old_checks.items()}, **{"candidate:"+k:v for k,v in new_checks.items()},
        "forward_unchanged":compare(old["output"],new["output"],atol=0,rtol=0)})
    row.update(length=length,tied=tied,input_tensors=tensor_meta(values),
        input_length=length,kernel_length=(length+7)//8*8 if arch=="MIMO" else length,
        original_oracle="pinned official + length adapter" if arch == "MIMO" and tied else
                        "pinned official" if tied else "local split-DT with upstream arithmetic",
        paired_fixture=True, baseline_before_candidate=True,
        calibration_status="PASS" if all(c["passed"] for c in old_checks.values()) else "FAIL",
        candidate_reference_confirmed=all(c["passed"] for c in old_checks.values()) and all(c["passed"] for c in new_checks.values()))
    return row


def angle(length, prefix_length, multiplier):
    from .angle_probe import reverse_scan
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_bwd as original
    from .stable_angle import angle_dt_bwd as stable
    from .initialization import seed_all
    seed_all()
    raw = (.1*torch.randn(1,length,2,32,device="cuda")).bfloat16()
    dt = torch.full((1,2,length),.23,device="cuda")
    go = torch.randn_like(raw)
    go[:,prefix_length:] = 0
    go = go * multiplier
    reverse = go.double().flip(1).cumsum(1).flip(1)
    scan = reverse_scan(go.float())
    tensors, advisory = {}, {}
    ideal_a = (reverse*dt.transpose(1,2)[...,None]*torch.pi*(1-raw.double().tanh().square())).to(raw.dtype)
    ideal_dt = (reverse*raw.double().tanh()*torch.pi).sum(-1).transpose(1,2)
    for backend_name, fn in (("upstream",original),("stable_angle",stable)):
        a,d,_ = fn(go,raw,dt,chunk_size=64)
        tensors[backend_name] = (a,d)
        advisory[backend_name] = dict(angle=compare(a,ideal_a),dt=compare(d,ideal_dt))
    checks = dict(arithmetic_angle=compare(tensors["upstream"][0],tensors["stable_angle"][0]),
        arithmetic_dt=compare(tensors["upstream"][1],tensors["stable_angle"][1]),
        reverse_scan=compare(scan,reverse),
        reverse_exact_tail=compare(scan[:,prefix_length:],torch.zeros_like(scan[:,prefix_length:]),atol=0,rtol=0))
    row = case("isolated_angle", "arithmetic pair and chunk64 reverse scan vs FP64 sum", checks)
    row.update(length=length,prefix=prefix_length,loss_multiplier=multiplier,
        ideal_elementary_functions=dict(required=False,status="ADVISORY",checks=advisory,
            oracle="FP64 PyTorch tanh vs pinned approximate tanh/sech2; not a scan oracle"),
        measured_advisory_mismatch=any(not c["passed"] for v in advisory.values() for c in v.values()))
    return row
