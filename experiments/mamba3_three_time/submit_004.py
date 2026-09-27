"""Durable reservation and exactly one sbatch; never monitors/retries."""

import argparse
import json
import os
import re
import subprocess
import traceback
from . import attempt_004 as attempt
from .provenance_004 import verify, sha, CORE
from .evidence import create_record, update_record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--commit',required=True)
    a=p.parse_args()
    def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=attempt.ROOT,text=True).strip()
    if str(attempt.ROOT)!='/home/daryumin/iberdov/diplom':raise ValueError('Canonical cluster repo only')
    if git('rev-parse','HEAD')!=a.commit or git('branch','--show-current')!='exp/mamba3-three-time':raise ValueError('Wrong HEAD/branch')
    if git('diff','HEAD','--'):raise ValueError('Tracked changes')
    manifest=verify()
    preflight=json.loads(attempt.PREFLIGHT.read_text())
    if preflight['status']!='PASS' or preflight['source']!=manifest:raise ValueError('Current preflight PASS required')
    cpu=json.loads((attempt.LOGS/'cpu_tests_004.json').read_text())
    if cpu.get('status')!='PASS' or cpu.get('source_hash')!=manifest['source_hash']:raise ValueError('Current CPU tests required')
    attempt.require_unused()
    record=dict(attempt_id='004',status='RESERVED',execution_commit=a.commit,source_hash=manifest['source_hash'],
        branch='exp/mamba3-three-time',parent_execution_commit=attempt.PARENT,
        core_hash=CORE,plan_sha256=sha(attempt.PLAN),source_manifest_sha256=sha(attempt.SOURCE_MANIFEST),
        jobs_submitted=None,training_authorized=False,scientific_fits=0,TRAIN=0,VALID=0,TEST=0,
        git_status_before_submit=git('status','--short'))
    create_record(attempt.SUBMISSION,record)
    env={**os.environ,'RUN_COMMIT':a.commit,'EXPECTED_STUDY_HASH':manifest['source_hash'],'EXPECTED_CORE_HASH':CORE}
    try:
        response=subprocess.run(["sbatch",'--parsable','slurm/mamba3_three_time_diagnostics_004.sh'],
            cwd=attempt.ROOT,env=env,text=True,capture_output=True,check=False)
        record.update(stdout=response.stdout,stderr=response.stderr,returncode=response.returncode)
        match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',response.stdout.strip())
        if response.returncode!=0 or match is None:
            record['status']='SUBMISSION_UNKNOWN_NO_RETRY'
            update_record(attempt.SUBMISSION,record)
            raise RuntimeError('Ambiguous/failed submit: reservation retained; no retry')
        record.update(status='SUBMITTED',job_id=match.group(1),jobs_submitted=1)
        update_record(attempt.SUBMISSION,record)
        print(json.dumps(record,indent=2),flush=True)
    except BaseException:
        if record['status']=='RESERVED':
            record.update(status='SUBMISSION_UNKNOWN_NO_RETRY',traceback=traceback.format_exc())
            update_record(attempt.SUBMISSION,record)
        raise


if __name__=='__main__':main()
