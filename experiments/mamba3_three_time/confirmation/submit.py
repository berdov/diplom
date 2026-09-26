"""Exclusive durable reservation; one sbatch; persist ID and return immediately."""
import argparse
import json
import os
import re
import subprocess
import traceback
from .config import ROOT, LOGS, HERE, SUBMISSION, POLICY_SHA, unused
from .provenance import verify, create_record, atomic_json, sha, now, CORE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--commit',required=True)
    commit=parser.parse_args().commit
    def git(*args):
        return subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT,text=True).strip()
    if str(ROOT)!='/home/daryumin/iberdov/diplom' or git('rev-parse','HEAD')!=commit or git('branch','--show-current')!='exp/mamba3-three-time':
        raise ValueError('Exact canonical checkout required')
    if git('diff','HEAD','--'):
        raise ValueError('Tracked checkout not clean')
    m=verify()
    for name in ('login_preflight.json','cpu_tests.json'):
        r=json.loads((LOGS/name).read_text())
        if r['status']!='PASS' or r['source_hash']!=m['source_hash']:
            raise ValueError('Same-source preflight/tests required')
    unused(include_submission=True)
    row=dict(status='RESERVED',reserved_at=now(),execution_commit=commit,source_hash=m['source_hash'],core_hash=CORE,
             policy_sha256=POLICY_SHA,plan_sha256=sha(HERE/'study_plan.json'),source_manifest_sha256=sha(HERE/'source_manifest.json'),
             jobs_submitted=None,max_scientific_fits=8,diagnostic_batches=1,max_diagnostic_optimizer_steps=2,
             scientific_fits_before_submit=0,TEST='NOT_RUN',test_evaluation_count=0,git_status_before_submit=git('status','--short'))
    create_record(SUBMISSION,row)
    try:
        response=subprocess.run(['sbatch','--parsable','slurm/mamba3_three_time_confirmation.sh'],cwd=ROOT,
            env=dict(os.environ,RUN_COMMIT=commit,EXPECTED_STUDY_HASH=m['source_hash'],EXPECTED_CORE_HASH=CORE),
            text=True,capture_output=True,check=False)
        row.update(stdout=response.stdout,stderr=response.stderr,returncode=response.returncode)
        match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',response.stdout.strip())
        if response.returncode!=0 or match is None:
            row['status']='SUBMISSION_UNKNOWN_NO_RETRY'
            atomic_json(SUBMISSION,row)
            raise RuntimeError('Ambiguous/failed sbatch; reservation retained; no retry')
        row.update(status='SUBMITTED',job_id=match.group(1),jobs_submitted=1)
        atomic_json(SUBMISSION,row)
        print(json.dumps(row,indent=2),flush=True)
    except BaseException:
        if row['status']=='RESERVED':
            row.update(status='SUBMISSION_UNKNOWN_NO_RETRY',traceback=traceback.format_exc())
            atomic_json(SUBMISSION,row)
        raise


if __name__=='__main__':
    main()
