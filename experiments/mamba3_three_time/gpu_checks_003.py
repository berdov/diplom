"""Attempt003: ADT-only candidate, explicit coverage, separate MIMO gates."""

import argparse
import importlib.metadata
import json
import os
import re
from pathlib import Path
import traceback
import torch
from . import attempt_003 as attempt, diagnostics_003 as diagnostics, suites, mimo_checks_003 as mimo
from .backends import selected
from .evidence import create_record, update_record, case
from .initialization import initialization_suite
from .provenance import CORE, PIN, require_submission, upstream, sha
from .records_003 import Registry, identity
from .gpu_checks_002 import expected_suite_names


def specs(arch, plan):
    backend = "stable_adt" if arch == "SISO" else "upstream_with_adapter"
    rows = [(identity(arch,"native",backend="official",length=16 if arch=="MIMO" else 65),"native")]
    rows.append((identity(arch,"initialization"),"initialization"))
    if arch == "MIMO":
        rows += [(identity(arch,"wrapped_gate",backend="official",length=n),("wrapped",n)) for n in (17,50)]
        rows += [(identity(arch,"local_gate_"+m,backend=backend,length=n),("local",m,n))
                 for n in (17,50) for m in ("base","dual","triple")]
        rows += [(identity(arch,"padding",backend=backend,length=n),("padding",n)) for n in plan["lengths"][arch]]
    else:
        rows += [(identity(arch,"drift_"+m+"_"+t,backend=backend,length=n),("drift",m,t,n))
                 for m in ("base","dual","triple") for t in ("eval","train") for n in (50,64)]
        rows += [(identity(arch,"prefix_"+m,backend=backend,length=n,prefix=p,multiplier=x),("prefix",m,n,p,x))
                 for m in ("base","dual","triple") for n,p in plan["prefix_fixtures"] for x in plan["loss_multipliers"]]
        rows += [(identity(arch,"isolated_angle",backend="upstream_vs_stable_angle",length=n,prefix=p,multiplier=x),("angle",n,p,x))
                 for n,p in plan["prefix_fixtures"] for x in plan["loss_multipliers"]]
    rows += [(identity(arch,"paired_reference_"+kind,backend=backend,length=n),("reference",n,kind=="tied"))
             for n in sorted(set([7,*plan["lengths"][arch]])) for kind in ("tied","triple")]
    for name in expected_suite_names(arch,plan):
        match = re.search(r"_L(\d+)$",name)
        rows.append((identity(arch,name,backend=backend,length=int(match[1]) if match else "mixed"),("suite",name)))
    return rows


def suite_factories(arch, plan):
    from .gpu_checks import reference_rows
    factories = [lambda:suites.official_parity(arch),lambda:suites.dual_recovery(arch),
        lambda:suites.triple_recovery(arch),lambda:reference_rows(arch,plan["reference"]),
        lambda:iter([suites.causal_padding(arch)]),lambda:suites.boundaries(arch,plan["lengths"][arch],plan["reference"]),
        lambda:suites.optimizer_steps(arch),lambda:iter([suites.roundtrip(arch)])]
    names = expected_suite_names(arch,plan)
    groups = [names[:4],names[4:8],names[8:10],names[10:12],names[12:13],
              names[13:-3],names[-3:-1],names[-1:]]
    result = {}
    for factory, group in zip(factories, groups):
        # Instantiate lazily: model work must occur inside the candidate context.
        def lazy(factory=factory):
            yield from factory()
        iterator = lazy()
        for name in group:
            result[name] = lambda iterator=iterator:next(iterator)
    return result


def run(arch, path):
    manifest = require_submission(attempt_id="003")
    plan = json.loads(attempt.PLAN.read_text())
    result = dict(attempt_id="003",status="RUNNING",architecture=arch,
        execution_commit=os.environ["RUN_COMMIT"],parent_execution_commit=attempt.PARENT,
        job_id=os.environ["SLURM_JOB_ID"],source_hash=manifest["source_hash"],core_hash=CORE,pinned_commit=PIN,
        plan=plan,plan_sha256=sha(attempt.PLAN),source_manifest_sha256=sha(attempt.SOURCE_MANIFEST),
        candidate_backend="stable_adt" if arch=="SISO" else "upstream_with_adapter",
        default_backend="upstream",backend_promotion="NOT_AUTHORIZED",gates={},
        scientific_fits=0,TRAIN=0,VALID=0,TEST=0)
    create_record(path,result)
    def save():
        update_record(path,result)
    schedule = specs(arch,plan)
    registry = Registry(result,save,[key for key,_ in schedule])
    def gate_progress(key,value):
        result["gates"][key] = dict(value)
        save()
    def native_or_wrapped(key,length,native):
        try:
            return mimo.gate(arch,length,native=native,progress=lambda v:gate_progress(key,v))
        except Exception:
            state = result["gates"].get(key,{})
            for field in ("forward","backward"):
                if state.get(field)=="RUNNING":
                    state[field]="FAIL"
            state.update(status="FAIL",traceback=traceback.format_exc())
            gate_progress(key,state)
            raise
    def init():
        def persist(value):
            result["initialization"] = value
            save()
        value = initialization_suite(arch,"cuda",persist)
        return case("initialization","within architecture, saved before assertion",{"initialization":dict(passed=value["status"]=="PASS")})
    candidate = "stable_adt" if arch=="SISO" else "upstream"
    try:
        result["upstream"] = upstream()
        result["runtime"] = {n:importlib.metadata.version(n) for n in ("torch","mamba-ssm","triton","tilelang","recbole")}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.cuda.set_device(0)
        torch.empty(1,device="cuda").zero_()
        torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32 = False
        result.update(gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
        if "A100" not in result["gpu"]:
            raise RuntimeError("Only authorized A100")
        save()
        suite_functions = suite_factories(arch,plan)
        paired_calibration = {}
        for case_id, spec in schedule:
            result["stage"] = case_id
            save()
            gate = False
            if spec == "native":
                factory = lambda:native_or_wrapped(case_id,16 if arch=="MIMO" else 65,True)
                gate = True
            elif spec == "initialization":
                factory,gate = init,True
            elif spec[0] == "wrapped":
                factory = lambda:native_or_wrapped(case_id,spec[1],False)
                gate = True
            elif spec[0] == "local":
                factory,gate = lambda:mimo.local_gate(spec[1],spec[2]),True
            elif spec[0] == "padding":
                factory = lambda:mimo.padding(spec[1],plan["reference"])
            elif spec[0] == "drift":
                factory = lambda:diagnostics.drift(spec[1],spec[2]=="train",spec[3])
            elif spec[0] == "prefix":
                factory = lambda:diagnostics.prefix(*spec[1:])
            elif spec[0] == "angle":
                factory = lambda:diagnostics.angle(*spec[1:])
            elif spec[0] == "reference":
                def factory():
                    row = diagnostics.paired_reference(arch,spec[1],spec[2],plan["reference"])
                    if spec[2]:
                        paired_calibration[spec[1]] = row["passed"]
                    elif not paired_calibration.get(spec[1],False):
                        row.update(passed=False,status="INCONCLUSIVE",candidate_reference_confirmed=False,
                            reason="Prior official tied calibration failed; measurements retained, not confirmed")
                    return row
            else:
                def factory():
                    row = suite_functions[spec[1]]()
                    if row["name"] != spec[1]:
                        row.update(passed=False,status="INCONCLUSIVE",expected_name=spec[1])
                    return row
            with selected(candidate):
                def with_lengths():
                    value = factory()
                    if arch=="MIMO" and isinstance(spec,tuple) and spec[0]=="suite":
                        match = re.search(r"_L(\d+)$",spec[1])
                        lengths = ([int(match[1])] if match else [7] if spec[1]=="C_leaf_gradient_sum" else
                                   [1,4,17] if spec[1]=="E_causality_padding" else [17])
                        value["length_adapter_cases"] = [dict(input_length=n,kernel_length=(n+7)//8*8) for n in lengths]
                        value["official_comparator_scope"] = "official kernel + length adapter, not native unaligned support"
                    return value
                row = registry.run(case_id,with_lengths,gate=gate)
            print(json.dumps(dict(case_id=case_id,status=row["status"])),flush=True)
        result["status"] = "PASS" if registry.close() else "FAIL"
    except Exception:
        result.update(status="FAIL",failure_stage=result.get("stage"),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        registry.close()
        result["original_exact_zero_check"] = ("FAIL" if any(r.get("original_exact_zero") is False for r in result["cases"])
            else "PASS" if any("original_exact_zero" in r for r in result["cases"]) else "NOT_RUN")
        result["advisory_mismatches"] = [r["case_id"] for r in result["cases"] if r.get("measured_advisory_mismatch")]
        result["full_local_correctness"] = result["status"]
        def section(prefix):
            expected=[key for key,_ in schedule if key.startswith(prefix)]
            rows=[r for r in result["cases"] if r["case_id"] in expected]
            return dict(status="NOT_RUN" if not rows else "INCOMPLETE" if len(rows)!=len(expected) else
                "PASS" if all(r["passed"] for r in rows) else "FAIL",
                expected=len(expected),measured=len(rows),failed=[r["case_id"] for r in rows if not r["passed"]])
        result["technical_sections"] = dict(native_support=section("native/"),
            wrapped_support=section("wrapped_gate/"),adapter_matrix=section("padding/"),
            structural_drift=section("drift_"),candidate_prefix=section("prefix_"),
            paired_reference=section("paired_reference_"),
            evidence_complete=not result["coverage"]["missing"] and not result["coverage"]["duplicates"])
        save()
    return result["status"]=="PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture",choices=("SISO","MIMO"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.resolve() != attempt.evidence(args.architecture).resolve():
        raise ValueError("Attempt003 evidence path required")
    raise SystemExit(0 if run(args.architecture,args.output) else 1)
