"""Attempt002, one allocation and two independent architecture subprocesses."""

import json
import os
import subprocess
import sys
import traceback
from . import attempt
from .evidence import create_record, update_record
from .provenance import require_submission, upstream, sha


def main():
    manifest = require_submission()
    attempt.require_unused()
    attempt.LOGS.mkdir(parents=True,exist_ok=True)
    attempt.RUNS.mkdir(exist_ok=True)
    identity = dict(attempt_id="002",job_id=os.environ["SLURM_JOB_ID"],
        execution_commit=os.environ["RUN_COMMIT"],parent_execution_commit=attempt.PARENT,
        source_hash=manifest["source_hash"],plan_sha256=sha(attempt.PLAN))
    reservation = json.loads(attempt.SUBMISSION.read_text())
    if reservation["execution_commit"] != identity["execution_commit"] or reservation["source_hash"] != identity["source_hash"]:
        raise ValueError("Attempt002 reservation mismatch")
    # Job may start before submit writes SUBMITTED; RESERVED is intentional.
    if reservation.get("job_id",identity["job_id"]) != identity["job_id"]:
        raise ValueError("Different Slurm job owns this attempt")
    create_record(attempt.LOCK,identity)
    state = dict(**identity,status="RUNNING",architectures={},scientific_fits=0,TRAIN=0,VALID=0,TEST=0)
    create_record(attempt.PIPELINE,state)
    try:
        upstream()
        for arch in ("SISO","MIMO"):
            require_submission()
            state["stage"] = arch
            update_record(attempt.PIPELINE,state)
            path = attempt.evidence(arch)
            with (attempt.LOGS/f"{arch.lower()}_stdout.log").open("x") as out, (attempt.LOGS/f"{arch.lower()}_stderr.log").open("x") as err:
                process = subprocess.run([sys.executable,"-m","experiments.mamba3_three_time.gpu_checks_002",
                    "--architecture",arch,"--output",str(path)],stdout=out,stderr=err,check=False)
            evidence = json.loads(path.read_text()) if path.exists() else {"status":"MISSING"}
            state["architectures"][arch] = dict(exit_code=process.returncode,status=evidence["status"],
                evidence=str(path),failed_cases=[r["name"] for r in evidence.get("cases",[]) if not r["passed"]],
                failed_diagnostics=[r["name"] for r in evidence.get("diagnostics",[]) if not r["passed"]],
                original_exact_zero_check=evidence.get("original_exact_zero_check"),
                localization=evidence.get("localization",{}).get("status"),
                native_support=evidence.get("native_support",{}).get("status"),
                full_local_correctness=evidence.get("full_local_correctness"))
            update_record(attempt.PIPELINE,state)
        passed = [v["status"] == "PASS" and v["exit_code"] == 0 for v in state["architectures"].values()]
        state["status"] = "PASS" if len(passed)==2 and all(passed) else "PARTIAL" if any(passed) else "FAIL"
    except Exception:
        state.update(status="FAIL",traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        update_record(attempt.PIPELINE,state)
        create_record(attempt.SUMMARY,state)
    print(json.dumps(state,indent=2),flush=True)
    return 0 if state["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
