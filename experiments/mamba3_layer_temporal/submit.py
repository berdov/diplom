"""One durable reservation before one sbatch; never poll or retry."""
import json
import os
import re
import secrets
import subprocess
import traceback
from . import config as c
from .provenance import login_verify,verify,bindings
from experiments.mamba3_mimo_time.records import read,create,sha,now


def reserve_and_submit(login,m):
    c.unused()
    create(c.LOGIN,login)
    token=secrets.token_hex(16)
    reservation=dict(**bindings(login['execution_commit'],m),reason="one shared versus layer-specific dual temporal pilot",status='RESERVED',token=token,login_sha256=sha(c.LOGIN),
                     max_scientific_fits=2,tasks=c.plan()['tasks'],scientific_fits_before_submit=0,jobs_requested=1,reserved_at=now())
    create(c.RESERVATION,reservation)
    operational=dict(token=token,job_id=None,status='SUBMISSION_UNKNOWN_NO_RETRY')
    create(c.SUBMISSION,operational)
    from experiments.mamba3_mimo_time.records import update
    try:
        proc=subprocess.run(['sbatch','--parsable',str(c.LAUNCHER.relative_to(c.ROOT))],cwd=c.ROOT,capture_output=True,text=True,
                            env=dict(os.environ,RUN_COMMIT=login['execution_commit'],EXPECTED_STUDY_HASH=m['source_hash'],RESERVATION_TOKEN=token))
        operational.update(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode)
        match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',proc.stdout.strip())
        if proc.returncode!=0 or match is None:raise RuntimeError('Ambiguous/failed sbatch; do not repeat')
        operational.update(job_id=match.group(1),status='SUBMITTED',jobs_submitted=1)
    except BaseException:
        operational['traceback']=traceback.format_exc();update(c.SUBMISSION,operational);raise
    update(c.SUBMISSION,operational)
    return operational


def main():
    if str(c.ROOT)!='/home/daryumin/iberdov/diplom':raise ValueError('Canonical cluster checkout required')
    login=login_verify();m=verify()
    for prefix in ('cpu_preflight_','no_git_preflight_'):
        record=read(c.LOGS/(prefix+login['execution_commit']+'.json'))
        if (record['status']!='PASS' or record['source_hash']!=m['source_hash'] or record['execution_commit']!=login['execution_commit']
            or record['execution_attempt']!=c.EXECUTION_ATTEMPT
            or record['cuda_initialized'] is not False or record['cpu_tests']['skipped']!=0):
            raise ValueError('Missing exact-source CPU/no-Git preflight')
    operational=reserve_and_submit(login,m)
    # Submission record is durable before revealing ID. No further cluster actions.
    print(json.dumps(dict(job_id=operational['job_id'],execution_attempt=c.EXECUTION_ATTEMPT,execution_commit=login['execution_commit'],source_hash=m['source_hash'],
                         reservation=str(c.RESERVATION),submission=str(c.SUBMISSION),jobs_submitted=1,
                         scientific_fits_before_submit=0,max_scientific_fits=2,TEST='NOT_RUN'),indent=2),flush=True)


if __name__=='__main__':main()
