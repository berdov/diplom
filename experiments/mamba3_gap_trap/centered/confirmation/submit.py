"""Durable reservation before one sbatch; ambiguous submission is never retried."""
import argparse
import json
import os
import re
import secrets
import subprocess
import traceback
from . import config as c
from .provenance import login_verify, verify, bindings
from experiments.mamba3_mimo_time.records import read, create, update, sha, now


def tasks_for_attempt(attempt,execution,source):
    if attempt=='001':return c.tasks(),None
    from .continuation import validate
    path=c.HERE/'runtime/continuation_admission.json'
    return validate(read(path),execution,source),sha(path)


def reserve_and_submit(login,m,attempt):
    tasks,admission=tasks_for_attempt(attempt,login['execution_commit'],m['source_hash'])
    c.unused(attempt,tasks);a=c.allocation(attempt);create(a['login'],login);token=secrets.token_hex(16)
    create(a['reservation'],dict(**bindings(login['execution_commit'],m,attempt),status='RESERVED',token=token,
        login_sha256=sha(a['login']),max_scientific_fits=8,tasks=tasks,jobs_requested=1,reserved_at=now(),
        scientific_fits_before_submit=8-len(tasks),continuation_admission_sha256=admission))
    operational=dict(token=token,job_id=None,status='SUBMISSION_UNKNOWN_NO_RETRY');create(a['submission'],operational)
    try:
        proc=subprocess.run(['sbatch','--parsable','--output='+str(a['logs']/'%x-%j.out'),'--error='+str(a['logs']/'%x-%j.err'),
            str(c.LAUNCHER.relative_to(c.ROOT)),attempt],cwd=c.ROOT,capture_output=True,text=True,
            env=dict(os.environ,RUN_COMMIT=login['execution_commit'],EXPECTED_STUDY_HASH=m['source_hash'],RESERVATION_TOKEN=token))
        operational.update(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode)
        match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',proc.stdout.strip())
        if proc.returncode!=0 or match is None:raise RuntimeError('Ambiguous/failed sbatch; do not repeat')
        operational.update(job_id=match.group(1),status='SUBMITTED',jobs_submitted=1)
    except BaseException:
        operational['traceback']=traceback.format_exc();update(a['submission'],operational);raise
    update(a['submission'],operational);return operational


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',choices=['001','002'],required=True);args=parser.parse_args()
    if str(c.ROOT)!='/home/daryumin/iberdov/diplom':raise ValueError('Canonical cluster checkout required')
    login=login_verify(args.attempt);m=verify();a=c.allocation(args.attempt)
    for prefix in ('cpu_preflight_','no_git_preflight_'):
        r=read(a['logs']/(prefix+login['execution_commit']+'.json'))
        if (r['status']!='PASS' or r['execution_commit']!=login['execution_commit'] or r['source_hash']!=m['source_hash']
            or r['cuda_initialized'] is not False or r['cpu_tests']['skipped']!=0 or r['cpu_tests']['run']<33):raise ValueError('Exact-source CPU/no-Git preflight required')
    # This checkout must have no active Slurm allocation. Other users' projects
    # in the shared account are inspected, never cancelled or modified.
    active=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%A'],text=True).split()
    jobs={}
    for job in sorted(set(active)):
        detail=subprocess.check_output(['scontrol','show','job',job],text=True)
        jobs[job]=detail
        if str(c.ROOT) in detail:raise ValueError('Active job references canonical checkout: '+job)
    login['active_job_checkout_check']=dict(status='PASS',jobs_checked=list(jobs))
    operational=reserve_and_submit(login,m,args.attempt)
    print(json.dumps(dict(job_id=operational['job_id'],execution_attempt=args.attempt,execution_commit=login['execution_commit'],
        source_hash=m['source_hash'],max_scientific_fits=8,TEST='NOT_RUN',test_evaluation_count=0),indent=2),flush=True)


if __name__=='__main__':main()
