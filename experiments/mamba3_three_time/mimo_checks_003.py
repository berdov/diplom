"""Native support and adapter checks are separate, ordered gates."""

import torch
from .evidence import case, compare
from .fixtures import kernel_inputs, kernel_output, clone_inputs, kernel_measure, scalar_loss, model, histories, nontrivial
from .length_adapter import prepare, AXES
from .suites import comparisons, measure_model


def gate(arch, length, *, native, progress=lambda value:None):
    state = dict(architecture=arch,input_length=length,
        kernel_length=length if native or arch=="SISO" else (length+7)//8*8,
        forward="NOT_RUN",backward="NOT_RUN",native=native,
        oracle="pinned native official" if native else "official kernel + length adapter")
    progress(state)
    values = kernel_inputs(arch,length=length,tied=True)
    state["forward"] = "RUNNING"
    progress(state)
    output = kernel_output(values,arch,official=True,native=native)
    torch.cuda.synchronize()
    state["forward"] = "PASS" if bool(torch.isfinite(output).all()) else "FAIL"
    state["backward"] = "RUNNING"
    progress(state)
    scalar_loss(output).backward()
    torch.cuda.synchronize()
    checks = dict(output=dict(passed=state["forward"] == "PASS"))
    checks.update({"gradient:"+k:dict(passed=v.grad is not None and bool(torch.isfinite(v.grad).all())) for k,v in values.items()})
    state["backward"] = "PASS" if all(c["passed"] for c in checks.values()) else "FAIL"
    progress(state)
    return dict(case("native" if native else "adapter_gate",state["oracle"],checks), **state)


def local_gate(mode, length):
    net = model("MIMO",mode).eval()
    if mode != "base":
        nontrivial(net.times.calibrators)
    measured = measure_model(net,histories(length))
    row = case("local_adapter_gate","local split-DT + adapter forward/backward",{
        k:dict(passed=v is not None and bool(torch.isfinite(v).all())) for k,v in measured.items()})
    row.update(mode=mode,input_length=length,kernel_length=(length+7)//8*8)
    return row


def padding(length, profile):
    source = kernel_inputs("MIMO",length=length,tied=True)
    records, checks = {}, {}
    for backend, official in (("official",True),("local",False)):
        original = clone_inputs(source)
        usual = kernel_measure(original,"MIMO",official=official)
        extra = kernel_measure(clone_inputs(source),"MIMO",official=official,extra_chunks=1)
        # Different kernel lengths use the pre-frozen reference profile, not a
        # bitwise assumption. Same padded shape retains structural tolerances.
        extended_checks = comparisons(extra,usual,profile)
        checks.update({backend+":extra_chunk:"+k:v for k,v in extended_checks.items()})
        manual = clone_inputs(source)
        padded, meta = prepare(manual)
        for key in AXES:
            padded[key].retain_grad()
        out = kernel_output(padded,"MIMO",official=official,native=official)
        out.retain_grad()
        scalar_loss(out[:,:length]).backward()
        seen = set()
        mapping = {"output":out[:,:length].detach(),"loss":scalar_loss(out[:,:length]).detach()}
        for key,value in manual.items():
            if id(value) not in seen:
                mapping["gradient:"+key] = value.grad
                seen.add(id(value))
        checks.update({backend+":crop_mapping:"+k:v for k,v in comparisons(mapping,usual).items()})
        dy_tail = out.grad[:,length:]
        checks[backend+":no_loss_on_tail"] = dict(passed=bool(torch.count_nonzero(dy_tail)==0))
        records[backend] = dict(**meta,extra_kernel_length=meta["kernel_length"]+8,
            original_output_shape=list(usual["output"].shape),
            padded_output_shape=list(out.shape),tail_output_gradient_nonzeros=int(torch.count_nonzero(dy_tail)),
            parameter_keys=[k for k in source if k not in AXES],
            parameter_gradient_check="All common parameter gradients compared in crop_mapping and extra_chunk checks",
            no_padding_identity=length%8==0)
        if length%8==0 and official:
            direct = kernel_measure(clone_inputs(source),"MIMO",official=True,native=True)
            checks.update({"native_identity:"+k:v for k,v in comparisons(usual,direct).items()})
    row = case("padding", "cropped loss/manual autograd mapping and extra neutral chunk",checks)
    row.update(input_length=length,kernel_length=(length+7)//8*8,backends=records,
        cross_length_profile=profile,same_shape_tolerance=dict(atol=1e-6,rtol=1e-5))
    return row
