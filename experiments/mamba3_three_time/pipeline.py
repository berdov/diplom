"""One Slurm allocation; separate architecture processes and durable summary."""

import json
import os
import subprocess
import sys
import traceback
from .evidence import create_record, update_record
from .provenance import HERE, require_submission, upstream


def main():
    manifest = require_submission()
    runtime, runs = HERE/"slurm_logs", HERE/"runs"
    runtime.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    identity = dict(job_id=os.environ["SLURM_JOB_ID"],execution_commit=os.environ["RUN_COMMIT"],
                    source_hash=manifest["source_hash"])
    if any((runs/name).exists() for name in ("siso_correctness_001.json","mimo_correctness_001.json","technical_summary.json")):
        raise FileExistsError("Existing evidence; no overwrite or automatic retry")
    create_record(runtime/"execution_001.lock",identity)
    state = dict(**identity,status="RUNNING",architectures={},scientific_fits=0,TRAIN=0,VALID=0,TEST=0)
    create_record(runtime/"pipeline_status.json",state)
    try:
        upstream()
        for arch in ("SISO","MIMO"):
            require_submission()
            state["stage"] = arch
            update_record(runtime/"pipeline_status.json",state)
            path = runs/f"{arch.lower()}_correctness_001.json"
            with (runtime/f"{arch.lower()}_stdout.log").open("x") as out, (runtime/f"{arch.lower()}_stderr.log").open("x") as err:
                process = subprocess.run([sys.executable,"-m","experiments.mamba3_three_time.gpu_checks",
                    "--architecture",arch,"--output",str(path)],stdout=out,stderr=err,check=False)
            evidence = json.loads(path.read_text()) if path.exists() else {"status":"MISSING"}
            state["architectures"][arch] = dict(exit_code=process.returncode,status=evidence["status"],
                evidence=str(path),failed_cases=[r["name"] for r in evidence.get("cases",[]) if not r["passed"]],
                failure_origin=evidence.get("failure_origin"))
            update_record(runtime/"pipeline_status.json",state)
        statuses = [v["status"] == "PASS" and v["exit_code"] == 0 for v in state["architectures"].values()]
        state["status"] = "PASS" if len(statuses) == 2 and all(statuses) else ("PARTIAL" if any(statuses) else "FAIL")
    except Exception:
        state.update(status="FAIL",traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        update_record(runtime/"pipeline_status.json",state)
        create_record(runs/"technical_summary.json",state)
    print(json.dumps(state,indent=2),flush=True)
    return 0 if state["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
