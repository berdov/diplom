"""Attempt002: fixed diagnostics and candidate suites, with incremental evidence."""

import argparse
import importlib.metadata
import json
import os
import inspect
import re
from pathlib import Path
import traceback
import torch
from . import attempt, diagnostics, suites
from .backends import selected
from .evidence import create_record, update_record, case
from .fixtures import kernel_inputs, kernel_output, scalar_loss
from .initialization import initialization_suite, seed_all
from .provenance import CORE, PIN, require_submission, upstream, sha


def native_gate(arch, result, save):
    record = dict(status="RUNNING",forward="NOT_RUN",backward="NOT_RUN",backend="official_upstream",
        chunk_size=8 if arch == "MIMO" else 64, rank=4 if arch == "MIMO" else 1,
        batch=1,length=17 if arch == "MIMO" else 65,heads=2,d_state=128,headdim=64,
        rotary_dim_divisor=4,dtype="bfloat16",no_final_state_loss=True,
        kernel="mamba_mimo_bwd_bwd_kernel" if arch == "MIMO" else "mamba3_siso_bwd_kernel_dqkv",
        static_smem_bytes=None,dynamic_smem_bytes=None,
        smem_metadata_note="Native wrapper does not expose compiled launch metadata; preserve compiler stderr.")
    result["native_support"] = record
    save()
    try:
        values = kernel_inputs(arch,length=record["length"],tied=True)
        record["input_shapes"] = {k:list(v.shape) for k,v in values.items()}
        if arch == "MIMO":
            from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import mamba_mimo_bwd_combined
            parameters = inspect.signature(mamba_mimo_bwd_combined).parameters
            record["launch_defaults"] = {k:str(parameters[k].default) for k in
                ("bf_threads","bf_num_stages","bb_threads","bb_num_stages","states_dtype")}
            record["launch_flags"] = dict(hasZ=True,hasD=True,reduceO=True,
                packed_dout=False,fuse_pregate_headwise_rms_norm=False,groups=1)
        save()
        output = kernel_output(values,arch,official=True)
        torch.cuda.synchronize()
        record["forward"] = "PASS" if bool(torch.isfinite(output).all()) else "FAIL"
        save()
        loss = scalar_loss(output)
        record["backward"] = "RUNNING"
        save()
        loss.backward()
        torch.cuda.synchronize()
        checks = {"output":dict(passed=record["forward"] == "PASS")}
        checks.update({"gradient:"+k:dict(passed=v.grad is not None and bool(torch.isfinite(v.grad).all()))
                       for k,v in values.items()})
        row = case("I_native_official","genuine pinned official forward/backward",checks)
        record.update(status="PASS" if row["passed"] else "FAIL",
                      backward="PASS" if all(c["passed"] for c in checks.values()) else "FAIL")
        save()
        return row
    except Exception:
        record.update(status="HARDWARE_OR_KERNEL_BLOCKED",traceback=traceback.format_exc())
        requested = re.search(r"dynamic shared memory size to (\d+)",record["traceback"])
        if requested:
            record["dynamic_smem_bytes"] = int(requested.group(1))
        if record["forward"] == "NOT_RUN":
            record["forward"] = "FAIL"
        elif record["backward"] == "RUNNING":
            record["backward"] = "FAIL"
        save()
        raise


def expected_suite_names(arch, plan):
    names = [f"{suite}_{mode}_L{length}" for suite in ("A","B")
             for mode in ("eval","train") for length in (50,64)]
    names += ["C_leaf_gradient_sum","C_tied_calibrator_recovery",
              "reference_calibration_official_tied_L7","D_three_path_L7","E_causality_padding"]
    for length in plan["lengths"][arch]:
        names += [f"F_official_full_sequence_L{length}",
                  f"F_reference_calibration_official_tied_L{length}",f"F_D_three_path_L{length}"]
    names += ["G_synthetic_optimizer_dual","G_synthetic_optimizer_triple","H_state_dict_roundtrip"]
    return names


def localization_verdict(rows):
    models = [row for row in rows if row["name"].startswith("localize_model_")]
    reproduced = [r["name"] for r in models if not r["original_exact_zero"]]
    sources = []
    for row in models:
        variants = row["variants"]
        original = variants["upstream"]["backward_stages"]
        for index, stage in enumerate(original):
            t = stage["tensors"]
            angle = variants["stable_angle"]["backward_stages"][index]["tensors"]
            adt = variants["stable_adt"]["backward_stages"][index]["tensors"]
            angle_same = variants["stable_angle"]["same_intermediate_inputs_as_upstream"][index]["dTheta"]
            adt_same = variants["stable_adt"]["same_intermediate_inputs_as_upstream"][index]["grad_after_z"]
            if (angle_same and t["dTheta"]["exact_zero"] and
                any(not t[k]["exact_zero"] for k in ("dDT_phase","dAngles")) and
                all(angle[k]["exact_zero"] for k in ("dDT_phase","dAngles"))):
                sources.append(dict(case=row["name"],reverse_layer_order=index,source="angle_dt_bwd total-minus-prefix"))
            if adt_same and not t["dADT"]["exact_zero"] and adt["dADT"]["exact_zero"]:
                sources.append(dict(case=row["name"],reverse_layer_order=index,source="compute_dqkv ADT reduction"))
    return dict(status="LOCALIZED" if reproduced and sources else "INCONCLUSIVE",
                reproduced_original_cases=reproduced,measured_sources=sources,
                note="Layer traces identify affected derivatives; multiple sources may coexist. No claim of future data use from magnitude alone.")


def run(arch, path):
    manifest = require_submission()
    plan = json.loads(attempt.PLAN.read_text())
    result = dict(attempt_id="002",status="RUNNING",architecture=arch,
        execution_commit=os.environ["RUN_COMMIT"],parent_execution_commit=attempt.PARENT,
        job_id=os.environ["SLURM_JOB_ID"],source_hash=manifest["source_hash"],core_hash=CORE,pinned_commit=PIN,
        plan=plan,plan_sha256=sha(attempt.PLAN),backend_variant="stable_scan_candidate",
        chunk_size=8 if arch == "MIMO" else 64,cases=[],diagnostics=[],diagnostic_progress={},initialization=None,
        original_exact_zero_check="NOT_RUN",localization={"status":"NOT_RUN"},
        stable_replacement_checks="NOT_RUN",full_local_correctness="NOT_RUN",
        scientific_fits=0,TRAIN=0,VALID=0,TEST=0)
    create_record(path,result)
    def save():
        update_record(path,result)
    def record(row, group="cases"):
        result[group].append(row)
        save()
        print(json.dumps(dict(architecture=arch,group=group,case=row["name"],passed=row["passed"])),flush=True)
    def execute(label, factory, group="cases"):
        try:
            for row in factory():
                record(row,group)
        except Exception:
            record(dict(name=label+"_EXCEPTION",passed=False,failed_keys=["exception"],
                        traceback=traceback.format_exc()),group)
            traceback.print_exc()
    try:
        result["upstream"] = upstream()
        result["runtime"] = {n:importlib.metadata.version(n) for n in ("torch","mamba-ssm","triton","tilelang","recbole")}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.cuda.set_device(0)
        torch.empty(1,device="cuda").zero_()
        torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32 = False
        props = torch.cuda.get_device_properties(0)
        result.update(gpu=props.name,capability=list(torch.cuda.get_device_capability()),
            shared_memory_per_block=getattr(props,"shared_memory_per_block",None),
            shared_memory_per_block_optin=getattr(props,"shared_memory_per_block_optin",None))
        save()
        if "A100" not in props.name:
            raise RuntimeError("Only authorized A100")
        seed_all()
        try:
            native = native_gate(arch,result,save)
            record(native)
            if not native["passed"]:
                raise RuntimeError("Native numerical checks failed")
        except Exception:
            result["failure_origin"] = "PINNED_NATIVE_UPSTREAM"
            if arch == "MIMO":
                result["status"] = "HARDWARE_OR_KERNEL_BLOCKED"
            raise
        def initial_save(value):
            result["initialization"] = value
            save()
        initialization_suite(arch,"cuda",initial_save)
        if arch == "SISO":
            def progress(row):
                result["diagnostic_progress"][row["name"]] = row
                save()
            for label,factory in (("model",diagnostics.model_localization),
                                  ("kernel",diagnostics.kernel_localization),
                                  ("angle",diagnostics.isolated_angle)):
                execute(label,lambda factory=factory:factory(plan,progress),"diagnostics")
            result["localization"] = localization_verdict(result["diagnostics"])
            model_rows = [r for r in result["diagnostics"] if "original_exact_zero" in r]
            result["original_exact_zero_check"] = ("FAIL" if any(not r["original_exact_zero"] for r in model_rows)
                                                    else "PASS" if len(model_rows)==18 else "INCOMPLETE")
            save()
        execute("scan_regression",lambda:diagnostics.original_stable_regression(arch))
        from .gpu_checks import reference_rows
        with selected("stable_scan"):
            for label,factory in [
                ("A",lambda:suites.official_parity(arch)),("B",lambda:suites.dual_recovery(arch)),
                ("C",lambda:suites.triple_recovery(arch)),("D",lambda:reference_rows(arch,plan["reference"])),
                ("E",lambda:iter([suites.causal_padding(arch)])),
                ("F",lambda:suites.boundaries(arch,plan["lengths"][arch],plan["reference"])),
                ("G",lambda:suites.optimizer_steps(arch)),("H",lambda:iter([suites.roundtrip(arch)]))]:
                require_submission()
                execute(label,factory)
        expected = expected_suite_names(arch,plan) + ["I_native_official"] + [
            f"scan_regression_{m}_{t}_L{n}" for m in ("base","dual","triple")
            for t in ("eval","train") for n in (50,64)]
        actual = [r["name"] for r in result["cases"]]
        result["coverage"] = dict(expected=expected,missing=sorted(set(expected)-set(actual)),
                                  unexpected=sorted(set(actual)-set(expected)),duplicates=len(actual)!=len(set(actual)))
        complete = sorted(expected)==sorted(actual)
        local_pass = complete and all(r["passed"] for r in result["cases"])
        result["full_local_correctness"] = "PASS" if local_pass else "FAIL"
        diagnostic_pass = arch == "MIMO" or (len(result["diagnostics"])==36 and
            all(r["passed"] for r in result["diagnostics"]) and result["localization"]["status"] == "LOCALIZED")
        result["stable_replacement_checks"] = "PASS" if local_pass and diagnostic_pass else "FAIL"
        result["status"] = "PASS" if local_pass and diagnostic_pass else "FAIL"
        result["backend_promotion"] = "ELIGIBLE_FOR_REVIEW_NOT_APPLIED" if result["status"] == "PASS" else "NOT_AUTHORIZED"
    except Exception:
        if result["status"] != "HARDWARE_OR_KERNEL_BLOCKED":
            result["status"] = "FAIL"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        save()
    return result["status"] == "PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture",choices=("SISO","MIMO"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.resolve() != attempt.evidence(args.architecture).resolve():
        raise ValueError("Attempt002 evidence path required")
    raise SystemExit(0 if run(args.architecture,args.output) else 1)
