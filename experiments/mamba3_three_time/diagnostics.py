"""Fixed prefix-loss localization; no threshold clipping or tail correction."""

import copy
import torch
from .backends import VARIANTS, selected
from .evidence import case, compare
from .fixtures import (model, histories, nontrivial, scalar_loss, kernel_inputs,
                       clone_inputs, kernel_output, kernel_measure)
from .initialization import seed_all
from .suites import comparisons, measure_model


def tail_stats(gradient, prefix, time_axis=1):
    if gradient is None:
        return dict(status="MISSING", exact_zero=False)
    g = gradient.detach().double().movedim(time_axis, 1)
    a, b = g[:, :prefix], g[:, prefix:]
    finite = bool(torch.isfinite(g).all())
    if not finite:
        return dict(status="NONFINITE", exact_zero=False)
    indices = torch.nonzero(b != 0, as_tuple=False)
    first = indices[:8].cpu().tolist()
    for row in first:
        row[1] += prefix
    prefix_norm, future_norm = a.norm().item(), b.norm().item()
    denominator = max(prefix_norm, 1e-30)
    return dict(status="MEASURED", dtype=str(gradient.dtype),
        shape=list(gradient.shape), time_axis=time_axis,
        max_abs=b.abs().max().item() if b.numel() else 0.,
        future_l2=future_norm, prefix_l2=prefix_norm,
        safe_denominator=denominator, denominator_floor=1e-30,
        future_prefix_ratio=future_norm/denominator,
        nonzero_entries=indices.shape[0], first_nonzero_indices=first,
        index_axes=(["batch","position","head"] if time_axis == 2 else
                    ["batch","position","head","channel"] if g.ndim == 4 else
                    ["batch","position","channel"]),
        exact_zero=indices.shape[0] == 0)


def frozen_separate(net):
    from recbole.config import Config
    from experiments.mamba3_time_mechanisms.config import load_config
    from experiments.mamba3_time_mechanisms.model import MechanismMamba3Rec
    from .config import SyntheticCatalog
    config = Config(model=MechanismMamba3Rec, config_dict=load_config("separate"))
    seed_all()
    ref = MechanismMamba3Rec(config, SyntheticCatalog()).cuda().eval()
    ref.load_state_dict({k.replace("times.", "mechanisms."):v for k,v in net.state_dict().items()}, strict=True)
    return ref


def prefix_model(net, data, prefix, multiplier, *, variant="upstream", oracle=None, old=False):
    embeddings, outputs, trace = [], [], []
    def capture(_module, _args, y):
        y.retain_grad()
        embeddings.append(y)
    hook = net.item_embedding.register_forward_hook(capture)
    full_hook = net.output_norm.register_forward_hook(lambda _m,_a,y: outputs.append(y)) if old else None
    net.zero_grad(set_to_none=True)
    seed_all()
    try:
        with selected(variant, trace):
            if old:
                net(*data)
                output = outputs[0]
            else:
                output = net.encode_sequence(*data, oracle=oracle)
            # Preserve attempt001 scalar_loss and user0-only prefix fixture.
            loss = multiplier * scalar_loss(output[0, :prefix])
            loss.backward()
        stats = tail_stats(embeddings[0].grad[0:1], prefix)
        return dict(output=output.detach(), gradient=embeddings[0].grad.detach().clone(), trace=trace,
            report=dict(embedding_output_gradient=stats,
                loss=loss.item(), loss_multiplier=multiplier, loss_dtype=str(loss.dtype),
                embedding_dtype=str(embeddings[0].dtype), output_dtype=str(output.dtype),
                embedding_index_layout="batch, position, embedding channel (not temporal head)",
                scalar_loss="original mean(weighted y) + .03 * mean(y^2); only user0/prefix",
                cross_user=tail_stats(embeddings[0].grad[1:], 0),
                backward_stages=[dict(chunk=r["chunk"], backend=r["backend"],
                    reverse_layer_order=i,
                    tensors={k:tail_stats(v, prefix, 2 if k in
                        ("dADT", "dDT_write", "dDT_phase", "dTrap", "dScale", "dGamma") else 1)
                        for k,v in r["tensors"].items()}) for i,r in enumerate(trace)]))
    finally:
        hook.remove()
        if full_hook is not None:
            full_hook.remove()


def model_localization(plan, save_progress=lambda row:None):
    for length, prefix in plan["prefix_fixtures"]:
        data = histories(length)
        for mode in ("base", "dual", "triple"):
            net = model("SISO", mode).eval()
            if mode != "base":
                nontrivial(net.times.calibrators)
            for multiplier in plan["loss_multipliers"]:
                original = prefix_model(net, data, prefix, multiplier)
                rows = {"upstream":original["report"]}
                name = f"localize_model_{mode}_L{length}_P{prefix}_x{multiplier}"
                progress = dict(name=name,status="RUNNING",variants=rows)
                save_progress(progress)
                checks = {}
                if mode in ("base", "dual"):
                    ref = copy.deepcopy(net) if mode == "base" else frozen_separate(net)
                    official = prefix_model(ref, data, prefix, multiplier,
                        oracle="official_base" if mode == "base" else None, old=mode == "dual")
                    rows["official_base" if mode == "base" else "frozen_separate"] = official["report"]
                    save_progress(progress)
                    checks["official_output"] = compare(original["output"], official["output"], atol=0, rtol=0)
                    checks["official_embedding_gradient"] = compare(original["gradient"], official["gradient"])
                for variant in VARIANTS[1:]:
                    actual = prefix_model(net, data, prefix, multiplier, variant=variant)
                    rows[variant] = actual["report"]
                    rows[variant]["same_intermediate_inputs_as_upstream"] = [
                        {key:bool(torch.equal(a["tensors"][key],b["tensors"][key]))
                         for key in ("grad_after_z","dTheta")}
                        for a,b in zip(original["trace"],actual["trace"])]
                    save_progress(progress)
                    checks[variant+":unchanged_output"] = compare(actual["output"],original["output"],atol=0,rtol=0)
                    checks[variant+":gradient_structural"] = compare(actual["gradient"],original["gradient"])
                checks["stable_scan:exact_future_zero"] = dict(passed=rows["stable_scan"]["embedding_output_gradient"]["exact_zero"])
                checks["stable_scan:exact_cross_user_zero"] = dict(passed=rows["stable_scan"]["cross_user"]["exact_zero"])
                row = case(name,
                           "same weights/positional embeddings/prefix scalar_loss", checks)
                row.update(mode=mode,length=length,prefix=prefix,loss_multiplier=multiplier,
                           variants=rows,original_exact_zero=rows["upstream"]["embedding_output_gradient"]["exact_zero"])
                save_progress(dict(row,status="PASS" if row["passed"] else "FAIL"))
                yield row


def kernel_localization(plan, save_progress=lambda row:None):
    for length, prefix in plan["prefix_fixtures"]:
        for tied in (True, False):
            source = kernel_inputs("SISO",length=length,tied=tied)
            for multiplier in plan["loss_multipliers"]:
                variants, measured = {}, {}
                name = f"localize_kernel_{'tied' if tied else 'split'}_L{length}_P{prefix}_x{multiplier}"
                progress = dict(name=name,status="RUNNING",variants=variants)
                for variant in ("official", *VARIANTS):
                    if variant == "official" and not tied:
                        continue
                    values, trace = clone_inputs(source), []
                    with selected("upstream" if variant == "official" else variant, trace):
                        output = kernel_output(values,"SISO",official=variant == "official")
                        loss = multiplier * scalar_loss(output[:, :prefix])
                        loss.backward()
                    grads = {k:v.grad.detach().clone() if v.grad is not None else None for k,v in values.items()}
                    measured[variant] = dict(output=output.detach(), **{"gradient:"+k:v for k,v in grads.items()})
                    variants[variant] = dict(loss=loss.item(), loss_multiplier=multiplier,
                        gradients={k:tail_stats(grads[k],prefix,2 if k in ("adt","dw","dp","trap") else 1)
                                   for k in ("adt","dw","dp","angles","q","k","v","trap")},
                        backward_stages=[{k:tail_stats(v,prefix,2 if k in
                            ("dADT","dDT_write","dDT_phase","dTrap","dScale","dGamma") else 1)
                            for k,v in t["tensors"].items()} for t in trace])
                    save_progress(progress)
                checks = {}
                for variant in measured:
                    if variant == "upstream":
                        continue
                    checks.update({variant+":"+k:v for k,v in comparisons(measured[variant],measured["upstream"]).items()})
                    checks[variant+":output"] = compare(measured[variant]["output"],measured["upstream"]["output"],atol=0,rtol=0)
                for key, stats in variants["stable_scan"]["gradients"].items():
                    checks["stable_exact_zero:"+key] = dict(passed=stats["exact_zero"])
                row = case(name,
                           "prefix-only loss at raw kernel inputs",checks)
                row.update(length=length,prefix=prefix,tied=tied,variants=variants)
                save_progress(dict(row,status="PASS" if row["passed"] else "FAIL"))
                yield row


def isolated_angle(plan, save_progress=lambda row:None):
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_bwd as original
    from .stable_angle import angle_dt_bwd as stable
    for length, prefix in plan["prefix_fixtures"]:
        seed_all()
        angle = (.1*torch.randn(1,length,2,32,device="cuda")).to(torch.bfloat16)
        dt = torch.full((1,2,length),.23,device="cuda")
        grad = torch.randn_like(angle)
        grad[:,prefix:] = 0  # The fixture input is prefix-only; backward never sees prefix.
        for multiplier in plan["loss_multipliers"]:
            go = grad * multiplier
            reverse = go.float().flip(1).cumsum(1).flip(1)
            # Elementary-function differences from pinned approximations are recorded,
            # not conflated with the scan-only original/stable structural comparison.
            expected = ((reverse*dt.transpose(1,2)[...,None]*torch.pi*(1-angle.float().tanh().square())).to(angle.dtype),
                        (reverse*angle.float().tanh()*torch.pi).sum(-1).transpose(1,2))
            rows, tensors = {}, {}
            name = f"isolated_angle_L{length}_P{prefix}_x{multiplier}"
            progress = dict(name=name,status="RUNNING",variants=rows)
            for name, fn in (("upstream",original),("stable_scan",stable)):
                da, dd, _ = fn(go,angle,dt,chunk_size=64)
                tensors[name] = (da,dd)
                rows[name] = dict(angle=tail_stats(da,prefix),dt=tail_stats(dd,prefix,2),
                    vs_direct_reference_angle=compare(da,expected[0]),vs_direct_reference_dt=compare(dd,expected[1]))
                save_progress(progress)
            checks = {"angle_structural":compare(tensors["upstream"][0],tensors["stable_scan"][0]),
                      "dt_structural":compare(tensors["upstream"][1],tensors["stable_scan"][1]),
                      "stable_angle_exact_zero":dict(passed=rows["stable_scan"]["angle"]["exact_zero"]),
                      "stable_dt_exact_zero":dict(passed=rows["stable_scan"]["dt"]["exact_zero"])}
            row = case(name,"direct PyTorch reverse cumsum",checks)
            row.update(variants=rows,direct_reverse_tail=tail_stats(reverse,prefix),length=length,prefix=prefix,
                       loss_multiplier=multiplier,grad_out_dtype=str(go.dtype),chunk_size=64)
            save_progress(dict(row,status="PASS" if row["passed"] else "FAIL"))
            yield row


def original_stable_regression(arch):
    for mode in ("base","dual","triple"):
        for training in (False,True):
            for length in (50,64):
                net = model(arch,mode).train(training)
                if mode != "base":
                    nontrivial(net.times.calibrators)
                data = histories(length)
                with selected("upstream"):
                    original = measure_model(net,data)
                with selected("stable_scan"):
                    stable = measure_model(net,data)
                checks = comparisons(stable,original)
                checks["output_unchanged"] = compare(stable["output"],original["output"],atol=0,rtol=0)
                yield case(f"scan_regression_{mode}_{'train' if training else 'eval'}_L{length}",
                           "same forward, changed summation order only",checks)
