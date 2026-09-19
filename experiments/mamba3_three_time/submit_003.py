"""Attempt003 reservation, exactly one sbatch; never retries or monitors."""

import argparse
import json
import os
import re
import subprocess
import traceback
from . import attempt_003 as attempt
from .evidence import create_record, update_record
from .provenance import ROOT, CORE, verify, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit",required=True)
    args = parser.parse_args()
    def git(*arguments):
        return subprocess.check_output(["git",*arguments],cwd=ROOT,text=True).strip()
    if str(ROOT) != "/home/daryumin/iberdov/diplom":
        raise ValueError("Canonical cluster checkout only")
    if git("rev-parse","HEAD") != args.commit or git("branch","--show-current") != "exp/mamba3-three-time":
        raise ValueError("Exact published branch/commit required")
    if git("diff","HEAD","--"):
        raise ValueError("Tracked modifications")
    manifest = verify()
    preflight = json.loads(attempt.PREFLIGHT.read_text())
    if preflight["status"] != "PASS" or preflight["source"] != manifest or preflight.get("attempt_id") != "003":
        raise ValueError("Current login preflight PASS required")
    attempt.require_unused()
    record = dict(attempt_id="003",status="RESERVED",execution_commit=args.commit,
        parent_execution_commit=attempt.PARENT,source_hash=manifest["source_hash"],core_hash=CORE,
        plan_sha256=sha(attempt.PLAN),source_manifest_sha256=sha(attempt.SOURCE_MANIFEST),branch="exp/mamba3-three-time",jobs_submitted=None,
        scientific_fits=0,TRAIN=0,VALID=0,TEST=0,git_status_before_submit=git("status","--short"))
    create_record(attempt.SUBMISSION,record)
    env = {**os.environ,"RUN_COMMIT":args.commit,"EXPECTED_STUDY_HASH":manifest["source_hash"],"EXPECTED_CORE_HASH":CORE}
    try:
        response = subprocess.run(["sbatch","--parsable","slurm/mamba3_three_time_correctness_003.sh"],
                                  cwd=ROOT,env=env,text=True,capture_output=True,check=False)
        record.update(stdout=response.stdout,stderr=response.stderr,returncode=response.returncode)
        match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?",response.stdout.strip())
        if response.returncode != 0 or match is None:
            record["status"] = "SUBMISSION_UNKNOWN_NO_RETRY"
            update_record(attempt.SUBMISSION,record)
            raise RuntimeError("Ambiguous/unsuccessful submission; reservation retained, do not retry")
        record.update(status="SUBMITTED",job_id=match.group(1),jobs_submitted=1)
        update_record(attempt.SUBMISSION,record)
        print(json.dumps(record,indent=2),flush=True)
    except Exception:
        if record["status"] == "RESERVED":
            record.update(status="SUBMISSION_UNKNOWN_NO_RETRY",traceback=traceback.format_exc())
            update_record(attempt.SUBMISSION,record)
        raise


if __name__ == "__main__":
    main()
