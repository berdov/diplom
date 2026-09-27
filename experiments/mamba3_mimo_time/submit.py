"""Login only: durable immutable reservation, exactly one sbatch, no polling."""
import json
import os
import re
import secrets
import subprocess
import traceback
from . import config as c
from .records import read, create, sha, now
from .provenance import login_verify, bindings, verify
from .retry import verify_parent


def main():
    if str(c.ROOT)!='/home/daryumin/iberdov/diplom':
        raise ValueError('Canonical cluster checkout required')
    c.unused()
    verify_parent(live=True)
    login=login_verify()
    m=verify()
    for name in ('cpu_tests_001.json','login_preflight_001.json','no_git_preflight_001.json','resource_check_001.json'):
        r=read(c.LOGS/name)
        if r['status']!='PASS' or r['source_hash']!=m['source_hash'] or r['execution_commit']!=login['execution_commit']:
            raise ValueError('Missing exact-source pre-submit check: '+name)
    create(c.LOGIN,login)
    token=secrets.token_hex(16)
    reservation=dict(**bindings(login['execution_commit'],m),status='RESERVED',token=token,login_sha256=sha(c.LOGIN),
                     max_scientific_fits=3,scientific_fits_before_submit=0,reserved_at=now(),jobs_requested=1)
    create(c.RESERVATION,reservation)
    operational=dict(token=token,job_id=None,status='SUBMISSION_UNKNOWN_NO_RETRY')
    try:
        result=subprocess.run(['sbatch','--parsable','slurm/mamba3_mimo_time.sh'],cwd=c.ROOT,
                              env=dict(os.environ,RUN_COMMIT=login['execution_commit'],EXPECTED_STUDY_HASH=m['source_hash'],RESERVATION_TOKEN=token),
                              capture_output=True,text=True,check=False)
        operational.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
        match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',result.stdout.strip())
        if result.returncode!=0 or match is None:
            raise RuntimeError('Ambiguous/failed sbatch; do not repeat')
        operational.update(job_id=match.group(1),status='SUBMITTED',jobs_submitted=1)
    except BaseException:
        operational['traceback']=traceback.format_exc()
        create(c.SUBMISSION,operational)
        raise
    create(c.SUBMISSION,operational)
    print(json.dumps(dict(job_id=operational['job_id'],execution_commit=login['execution_commit'],source_hash=m['source_hash'],
                         policy_sha256=reservation['policy_sha256'],reservation=str(c.RESERVATION),jobs_submitted=1),indent=2),flush=True)


if __name__=='__main__':
    main()
