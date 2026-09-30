"""Durable submit intent before sbatch; at most8h plus one guarded4h continuation."""
import json
import os
import re
import secrets
import subprocess
import traceback
from . import config as c
from .provenance import login_verify,verify,bindings
from .handoff import save
from experiments.mamba3_mimo_time.records import read,create,update,sha,now


def reserve_and_submit(login,m,attempt='001',continuation=None):
    a=c.allocation(attempt);c.unused(attempt)
    if attempt=='001':
        if c.allocation('002')['reservation'].exists():raise ValueError('Another allocation already reserved')
    elif not continuation:raise ValueError('Continuation audit required')
    create(a['login'],login)
    token=secrets.token_hex(16)
    scheduled=[{k:e[k] for k in ('variant','seed','run_id')} for e in c.index(attempt)['entries'] if e['attempt']==attempt]
    reservation=dict(**bindings(login['execution_commit'],m,attempt),status='RESERVED',token=token,login_sha256=sha(a['login']),
                     max_scientific_fits=12,tasks=scheduled,planned_tasks=c.tasks(),jobs_requested=1,reserved_at=now(),
                     allocation_seconds=c.plan()['allocations'][attempt],total_requested_seconds=28800 if attempt=='001' else 43200,
                     continuation=continuation,retry_reason=None if continuation is None else continuation['reason_class'])
    create(a['reservation'],reservation)
    operational=dict(token=token,job_id=None,status='SUBMISSION_UNKNOWN_NO_RETRY',submitted_at=now())
    create(a['submission'],operational)
    try:
        command=['sbatch','--parsable','--time='+('08:00:00' if attempt=='001' else '04:00:00'),
                 '--output='+str(a['logs']/'%x-%j.out'),'--error='+str(a['logs']/'%x-%j.err'),str(c.LAUNCHER.relative_to(c.ROOT))]
        proc=subprocess.run(command,cwd=c.ROOT,capture_output=True,text=True,
                            env=dict(os.environ,RUN_COMMIT=login['execution_commit'],EXPECTED_STUDY_HASH=m['source_hash'],
                                     RESERVATION_TOKEN=token,CONFIRMATION_ATTEMPT=attempt))
        operational.update(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode)
        match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',proc.stdout.strip())
        if proc.returncode!=0 or match is None:raise RuntimeError('Ambiguous/failed sbatch; recover existing submission, never repeat blindly')
        operational.update(job_id=match.group(1),status='SUBMITTED',jobs_submitted=1)
    except BaseException:
        operational['traceback']=traceback.format_exc();update(a['submission'],operational);raise
    update(a['submission'],operational)
    save('submitted',branch=c.BRANCH,execution_commit=login['execution_commit'],source_hash=m['source_hash'],
         job_ids=[operational['job_id']],reservation_path=str(a['reservation']),execution_attempt=attempt,
         submissions=int(attempt),requested_gpu_seconds=reservation['total_requested_seconds'],
         first_submission_at=read(c.allocation('001')['submission'])['submitted_at'],
         next_safe_step='Monitor no faster than every600s; terminal read-only audit, no duplicate submit')
    return operational


def main(attempt='001'):
    if str(c.ROOT)!='/home/daryumin/iberdov/diplom':raise ValueError('Canonical cluster checkout required')
    login=login_verify(attempt);m=verify(attempt);a=c.allocation(attempt)
    for prefix in ('cpu_preflight_','no_git_preflight_'):
        r=read(a['logs']/(prefix+login['execution_commit']+'.json'))
        if (r['status']!='PASS' or r['source_hash']!=m['source_hash'] or r['execution_commit']!=login['execution_commit']
            or r['cuda_initialized'] is not False or r['cpu_tests']['skipped']!=0):raise ValueError('Exact-source CPU/no-Git preflight missing')
    continuation=None
    if attempt=='002':
        from .continuation import verify_authorization
        continuation=verify_authorization()
    r=reserve_and_submit(login,m,attempt,continuation)
    print(json.dumps(dict(job_id=r['job_id'],execution_commit=login['execution_commit'],source_hash=m['source_hash'],
                         execution_attempt=attempt,submission=str(a['submission']),max_scientific_fits=12,TEST='NOT_RUN'),indent=2),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--attempt',choices=['001','002'],default='001')
    main(p.parse_args().attempt)
