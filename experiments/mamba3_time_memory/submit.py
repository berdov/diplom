"""Durable one-shot reservation; a second allocation needs verified pre-fit failure."""
import json
import os
import re
import secrets
import subprocess
import traceback
from . import config as c
from .provenance import login_verify,verify,bindings,coverage_verify
from experiments.mamba3_mimo_time.records import read,create,update,sha,now


def retry_review():
    if c.EXECUTION_ATTEMPT=='001':
        if (c.HERE/'slurm_logs/attempt_002/reservation.json').exists():raise ValueError('Second attempt already reserved')
        return None
    # Written only after preserving the actual failed allocation and reviewing its logs.
    path=c.HERE/'runtime/retry_review.json';r=read(path)
    if (r.get('status')!='APPROVED_PRE_FIT_TECHNICAL_RETRY' or r.get('scientific_fits_started')!=0
        or r.get('unknown_scientific_starts')!=0 or r.get('method_unchanged') is not True
        or r.get('tolerances_unchanged') is not True or r.get('regression_passed') is not True
        or not r.get('infrastructure_error') or r.get('old_job_terminal') is not True):raise ValueError('Retry not within authorized budget')
    old=c.HERE/'slurm_logs/attempt_001';submission=read(old/'submission.json')
    if submission.get('job_id')!=r['old_job_id']:raise ValueError('Retry parent job mismatch')
    scheduler=r['scheduler']
    if scheduler['JobIDRaw']!=r['old_job_id'] or scheduler['State'] not in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','COMPLETED'):raise ValueError('No terminal parent proof')
    preserved=c.ROOT/r['preservation_manifest_path']
    if sha(preserved)!=r['preservation_manifest_sha256']:raise ValueError('Failure preservation changed')
    for v in c.MODES:
        record=c.HERE/'runs/attempt_001'/(c.paths(v)['run_id']+'.json')
        lock=old/c.paths(v)['run_id']/'run.lock'
        if not record.exists():
            if lock.exists():raise ValueError('Unknown parent scientific start')
            continue
        if read(record).get('scientific_fit_started') is not False:raise ValueError('A scientific fit already started')
    return sha(path)


def reserve_and_submit(login,m):
    c.unused();review=retry_review();create(c.LOGIN,login);token=secrets.token_hex(16)
    reservation=dict(**bindings(login['execution_commit'],m),status='RESERVED',token=token,login_sha256=sha(c.LOGIN),coverage_sha256=coverage_verify(login['execution_commit']),max_scientific_fits=3,tasks=c.plan()['tasks'],scientific_fits_before_submit=0,jobs_requested=1,retry_review_sha256=review,reserved_at=now())
    create(c.RESERVATION,reservation);operational=dict(token=token,job_id=None,status='SUBMISSION_UNKNOWN_NO_RETRY');create(c.SUBMISSION,operational)
    try:
        limit='06:00:00' if c.EXECUTION_ATTEMPT=='001' else '04:00:00'
        command=['sbatch','--parsable','--time='+limit,'--output='+str((c.LOGS/'m3-time-memory-%j.out').relative_to(c.ROOT)),'--error='+str((c.LOGS/'m3-time-memory-%j.err').relative_to(c.ROOT)),str(c.LAUNCHER.relative_to(c.ROOT))]
        proc=subprocess.run(command,cwd=c.ROOT,capture_output=True,text=True,env=dict(os.environ,TIME_MEMORY_ATTEMPT=c.EXECUTION_ATTEMPT,RUN_COMMIT=login['execution_commit'],EXPECTED_STUDY_HASH=m['source_hash'],RESERVATION_TOKEN=token))
        operational.update(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode)
        match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',proc.stdout.strip())
        if proc.returncode!=0 or match is None:raise RuntimeError('Ambiguous/failed sbatch; do not repeat')
        operational.update(job_id=match.group(1),status='SUBMITTED',jobs_submitted=int(c.EXECUTION_ATTEMPT))
    except BaseException:
        operational['traceback']=traceback.format_exc();update(c.SUBMISSION,operational);raise
    update(c.SUBMISSION,operational);return operational


def main():
    if str(c.ROOT)!='/home/daryumin/iberdov/diplom':raise ValueError('Canonical cluster checkout required')
    login=login_verify();m=verify()
    for prefix in ('cpu_preflight_','no_git_preflight_'):
        r=read(c.LOGS/(prefix+login['execution_commit']+'.json'))
        if (r['status']!='PASS' or r['source_hash']!=m['source_hash'] or r['execution_commit']!=login['execution_commit'] or r['execution_attempt']!=c.EXECUTION_ATTEMPT or r['cuda_initialized'] is not False or any(r['cpu_tests'][k] for k in ('skipped','errors','failures'))):raise ValueError('Missing exact CPU/no-Git proof')
    result=reserve_and_submit(login,m)
    print(json.dumps(dict(job_id=result['job_id'],execution_attempt=c.EXECUTION_ATTEMPT,execution_commit=login['execution_commit'],source_hash=m['source_hash'],reservation=str(c.RESERVATION),submission=str(c.SUBMISSION),scientific_fits_before_submit=0,max_scientific_fits=3,TEST='NOT_RUN'),indent=2),flush=True)


if __name__=='__main__':main()
