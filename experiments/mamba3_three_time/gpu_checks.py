"""One architecture per process; owned, incremental evidence survives failure."""

import argparse
import json
import os
from pathlib import Path
import traceback
import torch
from .provenance import HERE, CORE, PIN, require_submission, upstream
from .evidence import create_record, update_record
from .initialization import initialization_suite, seed_all
from . import suites


def run(arch,path):
    manifest = require_submission()
    result = dict(status="RUNNING",architecture=arch,execution_commit=os.environ["RUN_COMMIT"],
        job_id=os.environ["SLURM_JOB_ID"],source_hash=manifest["source_hash"],core_hash=CORE,
        pinned_commit=PIN,cases=[],initialization=None,scientific_fits=0,TRAIN=0,VALID=0,TEST=0)
    create_record(path,result)
    def save():
        update_record(path,result)
    def record(row):
        result["cases"].append(row)
        save()
        print(json.dumps(dict(architecture=arch,case=row["name"],passed=row["passed"])),flush=True)
    try:
        result["upstream"] = upstream()
        if not torch.cuda.is_available():
            raise RuntimeError("GPU allocation lacks CUDA")
        torch.cuda.set_device(0)
        torch.empty(1,device="cuda").zero_()
        torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32 = False
        result.update(gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
            cuda_initialized_before_config=torch.cuda.is_initialized())
        save()
        if "A100" not in result["gpu"]:
            raise RuntimeError("Expected the authorized A100 allocation")
        # Native MIMO is attempted before local MIMO numerical suites.
        seed_all()
        try:
            native = suites.native(arch)
            record(native)
            if not native["passed"]:
                raise RuntimeError("Official native forward/backward nonfinite or missing gradients")
        except Exception:
            result["failure_origin"] = "PINNED_NATIVE_UPSTREAM"
            raise
        def initial_save(value):
            result["initialization"] = value
            save()
        initialization_suite(arch,"cuda",initial_save)
        plan = json.loads((HERE/"test_plan.json").read_text())
        factories = [
            ("A",lambda:suites.official_parity(arch)),
            ("B",lambda:suites.dual_recovery(arch)),
            ("C",lambda:suites.triple_recovery(arch)),
            ("D",lambda:reference_rows(arch,plan["reference"])),
            ("E",lambda:iter([suites.causal_padding(arch)])),
            ("F",lambda:suites.boundaries(arch,plan["lengths"][arch],plan["reference"])),
            ("G",lambda:suites.optimizer_steps(arch)),
            ("H",lambda:iter([suites.roundtrip(arch)]))]
        for name,factory in factories:
            require_submission()
            try:
                for row in factory():
                    record(row)
            except Exception:
                record(dict(name=name+"_EXCEPTION",oracle="technical suite",passed=False,
                    failed_keys=["exception"],checked_tensors=0,traceback=traceback.format_exc()))
                traceback.print_exc()
        result["status"] = "PASS" if all(row["passed"] for row in result["cases"]) else "FAIL"
    except Exception:
        result.update(status="FAIL",traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    return result["status"] == "PASS"


def reference_rows(arch,profile):
    calibration = suites.reference_check(arch,7,profile,independent=False)
    yield calibration
    if calibration["passed"]:
        yield suites.reference_check(arch,7,profile,independent=True)
    else:
        yield dict(name="D_INCONCLUSIVE",oracle="official tied reference calibration failed",
            passed=False,failed_keys=["calibration"],checked_tensors=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture",choices=("SISO","MIMO"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.architecture,args.output) else 1)
