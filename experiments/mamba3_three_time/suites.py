"""Mandatory synthetic GPU suites; no Trainer, real data, or recommendation metrics."""

import copy
import io
import torch
from .evidence import case, compare, tensor_difference
from .fixtures import (model, nontrivial, histories, kernel_inputs, clone_inputs,
                       kernel_measure, scalar_loss)
from .initialization import seed_all
from .config import SyntheticCatalog


def comparisons(left, right, profile=None):
    checks = {}
    for key in sorted(set(left) | set(right)):
        kwargs = {}
        if profile:
            prefix = "gradient_" if key.startswith("gradient:") else ""
            kwargs = {name:profile[prefix+name] for name in ("atol", "rtol", "relative_norm_limit")}
        checks[key] = compare(left.get(key), right.get(key), **kwargs)
    return checks


def measure_model(net, data, oracle=None, old=False):
    captured = []
    def capture(_module, _args, output):
        output.retain_grad()
        captured.append(output)
    hook = net.item_embedding.register_forward_hook(capture)
    net.zero_grad(set_to_none=True)
    seed_all()
    try:
        if old or oracle is None:
            output = net(*data)
        else:
            all_hidden = net.encode_sequence(*data, oracle=oracle)
            output = net.gather_indexes(all_hidden, data[1]-1)
        loss = scalar_loss(output)
        loss.backward()
        result = dict(output=output.detach(), loss=loss.detach(), input_gradient=captured[0].grad.detach().clone())
        for name, param in net.named_parameters():
            name = name.replace("mechanisms.", "times.")
            result["gradient:"+name] = param.grad.detach().clone() if param.grad is not None else None
        return result
    finally:
        hook.remove()


def native(arch):
    values = kernel_inputs(arch, length=17 if arch == "MIMO" else 65, tied=True)
    measured = kernel_measure(values, arch, official=True)
    checks = {key:dict(passed=value is not None and bool(torch.isfinite(value).all()),
                       finite=value is not None and bool(torch.isfinite(value).all())) for key, value in measured.items()}
    return case("I_native_official", "pinned official forward/backward, no local wrapper", checks)


def official_parity(arch):
    for training in (False, True):
        for length in (50, 64):
            new = model(arch,"base").train(training)
            ref = copy.deepcopy(new)
            data = histories(length)
            yield case(f"A_{'train' if training else 'eval'}_L{length}", "official full recommender",
                comparisons(measure_model(new,data), measure_model(ref,data,oracle="official_base")))


def dual_recovery(arch):
    for training in (False, True):
        for length in (50,64):
            new = model(arch,"dual").train(training)
            nontrivial(new.times.calibrators)
            if arch == "SISO":
                from recbole.config import Config
                from experiments.mamba3_time_mechanisms.config import load_config
                from experiments.mamba3_time_mechanisms.model import MechanismMamba3Rec
                config = Config(model=MechanismMamba3Rec, config_dict=load_config("separate"))
                seed_all()
                ref = MechanismMamba3Rec(config, SyntheticCatalog()).cuda().train(training)
                ref.load_state_dict({k.replace("times.","mechanisms."):v for k,v in new.state_dict().items()}, strict=True)
                old = measure_model(ref,histories(length),old=True)
                oracle = "frozen SISO separate with nonzero calibrators"
            else:
                ref = copy.deepcopy(new)
                old = measure_model(ref,histories(length),oracle="official_dual")
                oracle = "official native MIMO recurrence: ADT_decay and common DT_scan for write/phase"
            yield case(f"B_{'train' if training else 'eval'}_L{length}", oracle,
                       comparisons(measure_model(new,histories(length)),old))


def triple_recovery(arch):
    tied = kernel_inputs(arch,tied=True)
    split = clone_inputs(tied)
    split["dp"] = split["dw"].detach().clone().requires_grad_(True)
    both = kernel_measure(tied,arch)
    separate = kernel_measure(split,arch)
    summed = separate.pop("gradient:dp") + separate["gradient:dw"]
    separate["gradient:dw"] = summed
    yield case("C_leaf_gradient_sum", "same leaf passed twice vs two independent leaf copies",
               comparisons(separate,both))
    dual, triple = model(arch,"dual").eval(), model(arch,"triple").eval()
    nontrivial(dual.times.calibrators)
    state = {}
    for key in triple.state_dict():
        source = key.replace(".write.",".scan.").replace(".phase.",".scan.")
        state[key] = dual.state_dict()[source]
    triple.load_state_dict(state, strict=True)
    left, right = measure_model(triple,histories(17)), measure_model(dual,histories(17))
    for name in tuple(left):
        if name.startswith("gradient:times.calibrators.write."):
            phase = name.replace(".write.",".phase.")
            left[name.replace(".write.",".scan.")] = left.pop(name)+left.pop(phase)
    checks = comparisons(left,right)
    checks["independent_parameter_objects"] = dict(passed=not (
        {id(p) for p in triple.times.calibrators["write"].parameters()} &
        {id(p) for p in triple.times.calibrators["phase"].parameters()}))
    yield case("C_tied_calibrator_recovery", "dual scan gradient = triple write + phase gradients", checks)


def reference_check(arch, length, profile, *, independent):
    values = kernel_inputs(arch, length=length, tied=not independent)
    if independent:
        with torch.no_grad():
            variation = torch.linspace(.7,1.3,values["dw"].numel(),device="cuda").reshape_as(values["dw"])
            values["adt"].mul_(variation)
            values["dw"].mul_(variation.flip(-1))
            values["dp"].mul_(1.8-variation)
    reference = clone_inputs(values,float_reference=True)
    actual = kernel_measure(values, arch, official=not independent)
    expected = kernel_measure(reference,arch,reference=True)
    checks = comparisons(actual,expected,profile)
    row = case(f"{'D_three_path' if independent else 'reference_calibration_official_tied'}_L{length}",
               "independent PyTorch full recurrence",checks,
               dtype=profile["kernel_dtype"]+" vs "+profile["reference_dtype"])
    row["independent_paths"] = independent
    row["path_gradient_norms"] = {k: actual["gradient:"+k].float().norm().item()
        for k in ("adt","dw","dp","trap","angles") if "gradient:"+k in actual}
    return row


def causal_padding(arch):
    net = model(arch,"triple").eval()
    nontrivial(net.times.calibrators)
    data = histories(17)
    items, lengths, times = data
    prefix = 7
    out = net.encode_sequence(*data)
    changed_items, changed_times = items.clone(), times.clone()
    changed_items[0,prefix:] = (changed_items[0,prefix:] + 37) % 7111 + 1
    changed_times[0,prefix:] += 90000000
    changed = net.encode_sequence(changed_items,lengths,changed_times)
    checks = {"suffix_invariance":compare(out[:,:prefix],changed[:,:prefix]),
              "no_cross_user":compare(out[1],changed[1])}
    capture = []
    def retain(_m,_a,y):
        y.retain_grad()
        capture.append(y)
    hook = net.item_embedding.register_forward_hook(retain)
    try:
        net.zero_grad(set_to_none=True)
        y = net.encode_sequence(*data)
        scalar_loss(y[0,:prefix]).backward()
        gradient = capture[0].grad
        checks["future_gradient_zero"] = compare(gradient[0,prefix:],torch.zeros_like(gradient[0,prefix:]),atol=0,rtol=0)
        checks["cross_user_gradient_zero"] = compare(gradient[1],torch.zeros_like(gradient[1]),atol=0,rtol=0)
    finally:
        hook.remove()
    short = histories(1,batch=1)
    padded_items = torch.zeros((1,17),device="cuda",dtype=torch.long)
    padded_items[:,0] = short[0][:,0]
    padded_times = torch.full((1,17),float("nan"),device="cuda",dtype=torch.float64)
    padded_times[:,0] = short[2][:,0]
    checks["length1_right_padding"] = compare(net(*short),net(padded_items,short[1],padded_times))
    from experiments.mamba3_timeaware.time_inputs import history_gaps
    valid = padded_items != 0
    gaps, active = history_gaps(padded_times,valid)
    for index, scale in enumerate(net.times(gaps,active)):
        checks[f"padding_first_neutral_{index}"] = compare(scale,torch.ones_like(scale),atol=0,rtol=0)
    equal = torch.ones((1,4),device="cuda",dtype=torch.float64)*1.6e12
    gaps,active = history_gaps(equal,torch.ones_like(equal,dtype=torch.bool))
    checks["real_zero_gap_active"] = dict(passed=bool(active[:,1:].all()) and not bool(active[:,0].any()))
    for index,scale in enumerate(net.times(gaps,active)):
        checks[f"real_zero_gap_conditioned_{index}"] = dict(passed=bool((scale[:,1:] != 1).any()) and bool(torch.isfinite(scale).all()))
    eq_items = torch.arange(1,5,device="cuda")[None,:]
    eq_y = net(eq_items,torch.tensor([4],device="cuda"),equal)
    checks["equal_timestamps_forward_finite"] = dict(passed=bool(torch.isfinite(eq_y).all()))
    return case("E_causality_padding", "prefix interventions, exact zero future/user derivatives",checks)


def boundaries(arch,lengths,profile):
    for length in lengths:
        tied = kernel_inputs(arch,length=length,tied=True)
        official = kernel_measure(clone_inputs(tied),arch,official=True)
        yield case(f"F_official_full_sequence_L{length}","official tied full sequence across chunks",
                   comparisons(kernel_measure(tied,arch),official))
        calibration = reference_check(arch,length,profile,independent=False)
        calibration["name"] = "F_"+calibration["name"]
        yield calibration
        if calibration["passed"]:
            row = reference_check(arch,length,profile,independent=True)
            row["name"] = "F_"+row["name"]
            yield row
        else:
            yield case(f"F_three_path_L{length}_INCONCLUSIVE", "reference calibration failed; no tolerance change",
                       {"baseline_calibration":dict(passed=False)})


def optimizer_steps(arch):
    for mode in ("dual","triple"):
        net = model(arch,mode).train()
        optimizer = torch.optim.Adam(net.parameters(),lr=.001)
        rows, checks = [], {}
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            seed_all(2026+step)
            loss = scalar_loss(net(*histories(17)))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite synthetic optimizer loss")
            loss.backward()
            grads = {name: (p.grad.float().norm().item() if p.grad is not None and torch.isfinite(p.grad).all() else None)
                     for name,p in net.times.named_parameters()}
            rows.append(dict(step=step,loss=loss.item(),gradient_norms=grads))
            checks[f"finite_gradients_step{step}"] = dict(passed=all(
                p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters()))
            optimizer.step()
        for name in rows[-1]["gradient_norms"]:
            checks["learnable:"+name] = dict(passed=any((r["gradient_norms"][name] or 0)>0 for r in rows[1:]))
        row = case("G_synthetic_optimizer_"+mode,"three synthetic steps; not a scientific fit",checks)
        row["steps"] = rows
        yield row


def roundtrip(arch):
    net = model(arch,"triple").eval()
    nontrivial(net.times.calibrators)
    stream = io.BytesIO()
    torch.save(net.state_dict(),stream)
    stream.seek(0)
    state = torch.load(stream,map_location="cuda",weights_only=True)
    restored = model(arch,"triple").eval()
    restored.load_state_dict(state,strict=True)
    checks = dict(state_dict=dict(passed=not tensor_difference(net.state_dict(),restored.state_dict())),
                  output=compare(net(*histories(17)),restored(*histories(17))))
    return case("H_state_dict_roundtrip","in-memory own pure state_dict, weights_only=True",checks)
