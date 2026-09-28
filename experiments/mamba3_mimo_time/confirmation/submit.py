"""Login only, durable reservation before exactly one sbatch; never poll/retry."""
import json
import os
import re
import secrets
import subprocess
import traceback
from . import config as c
from .records import read, create, sha, now
from .provenance import login_verify, bindings, verify


def main():
    if str(c.ROOT) != '/home/daryumin/iberdov/diplom':
        raise ValueError('Canonical cluster checkout required')
    c.unused()
    login = login_verify()
    m = verify()
    for name in ('cpu_tests_001.json','login_preflight_001.json','no_git_preflight_001.json','ownership_001.json'):
        r = read(c.LOGS/name)
        if r['status'] != 'PASS' or r['source_hash'] != m['source_hash'] or r['execution_commit'] != login['execution_commit']:
            raise ValueError('Missing exact-source pre-submit check: '+name)
    create(c.LOGIN,login)
    token = secrets.token_hex(16)
    reservation = dict(**bindings(login['execution_commit'],m),status='RESERVED',token=token,login_sha256=sha(c.LOGIN),
                       max_scientific_fits=12,tasks=c.tasks(),scientific_fits_before_submit=0,jobs_requested=1,reserved_at=now())
    create(c.RESERVATION,reservation)
    operational = dict(token=token,job_id=None,status='SUBMISSION_UNKNOWN_NO_RETRY')
    try:
        proc = subprocess.run(['sbatch','--parsable',str(c.LAUNCHER.relative_to(c.ROOT))],cwd=c.ROOT,capture_output=True,text=True,
                              env=dict(os.environ,RUN_COMMIT=login['execution_commit'],EXPECTED_STUDY_HASH=m['source_hash'],RESERVATION_TOKEN=token))
        operational.update(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode)
        match = re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',proc.stdout.strip())
        if proc.returncode != 0 or match is None:
            raise RuntimeError('Ambiguous/failed sbatch: do not repeat')
        operational.update(job_id=match.group(1),status='SUBMITTED',jobs_submitted=1)
    except BaseException:
        operational['traceback'] = traceback.format_exc()
        create(c.SUBMISSION,operational)
        raise
    create(c.SUBMISSION,operational)
    print(json.dumps(dict(job_id=operational['job_id'],execution_commit=login['execution_commit'],source_hash=m['source_hash'],
                         admission_sha256=c.ADMISSION_SHA,smoke_sha256=c.SMOKE_SHA,reservation=str(c.RESERVATION),jobs_submitted=1),indent=2),flush=True)


if __name__ == '__main__':
    main()
