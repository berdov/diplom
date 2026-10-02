"""Own allocation identity; unchanged pilot admission inherited by content."""
import hashlib
import importlib.metadata
import os
import re
import sys
from pathlib import Path
from . import config as c
from experiments.mamba3_mimo_time.records import read,sha,digest,create,now,accepted_cases
from experiments.mamba3_mimo_time.confirmation.provenance import check_files,inherited as inherited_kernel


def inherited():
    from experiments.mamba3_head_timescales.provenance import validate_smoke
    from experiments.mamba3_time_mechanisms.provenance import fingerprint
    manifest=read(c.PILOT_ROOT/'source_manifest_002.json')
    if manifest['source_hash']!=c.PILOT_SOURCE:raise ValueError('Pilot source identity')
    check_files(c.ROOT,manifest)
    if fingerprint()!=c.CORE or sha(c.POLICY)!=c.POLICY_SHA:raise ValueError('Mathematical core/policy drift')
    kernel=inherited_kernel()
    if sha(c.GATE)!=c.GATE_SHA or sha(c.SMOKE)!=c.SMOKE_SHA:raise ValueError('Pilot admission SHA')
    gate,smoke=read(c.GATE),read(c.SMOKE)
    spec=read(c.PILOT_ROOT/'study_plan.json')['required_cases']
    expected=dict(execution_commit=c.PILOT_COMMIT,source_hash=c.PILOT_SOURCE,job_id='4362620',execution_attempt='002',
                  core_hash=c.CORE,pinned_commit=c.PIN,policy_sha256=c.POLICY_SHA,TEST='NOT_RUN',test_evaluation_count=0)
    for r in (gate,smoke,*[read(c.pilot_path(v)) for v in c.MODES]):
        if r.get('status')!='PASS' or any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Pilot evidence lineage')
    if len(spec)!=9 or sum(len(x['required_keys']) for x in spec)!=228 or gate['required_cases']!=spec or not accepted_cases(gate['cases'],spec):
        raise ValueError('Incomplete targeted registry/leaves')
    validate_smoke(smoke)
    if smoke['targeted_gate_sha256']!=c.GATE_SHA:raise ValueError('Smoke-to-gate lineage')
    if c.plan()['diagnostic_grid']!=read(c.PILOT_ROOT/'study_plan.json')['diagnostic_grid']:
        raise ValueError('Unchanged trainer diagnostic grid dependency')
    return dict(status='INHERITED_PASS',original_job_id='4362620',original_execution_commit=c.PILOT_COMMIT,
                original_source_hash=c.PILOT_SOURCE,targeted_gate_sha256=c.GATE_SHA,smoke_sha256=c.SMOKE_SHA,
                targeted_cases=9,targeted_checks=228,smoke_synthetic_steps=9,repeated_gpu_checks=0,
                mathematical_dependencies_unchanged=True,kernel=kernel)


def freeze(attempt='001'):
    a=c.allocation(attempt);c.unused(attempt);inherited()
    files=dict(read(c.PILOT_ROOT/'source_manifest_002.json')['files'])
    sources=[*c.HERE.glob('*.py'),*(c.HERE/'tests').glob('*.py'),c.HERE/'README.md',c.HERE/'study_plan.json',
             a['source_index'],c.LAUNCHER,c.PILOT_ROOT/'source_manifest_002.json',c.GATE,c.SMOKE]
    sources += [c.pilot_path(v) for v in c.MODES]
    if attempt=='002':sources += [c.allocation('001')['manifest'],c.allocation('001')['source_index']]
    for p in sources:files[str(p.relative_to(c.ROOT))]=sha(p)
    value=dict(files=files,source_hash=digest(files),publication_commit=c.PUBLICATION,pilot_execution=c.PILOT_COMMIT,
               pilot_source_hash=c.PILOT_SOURCE,core_hash=c.CORE,execution_attempt=attempt)
    create(a['manifest'],value)
    return value


def verify(attempt='001'):
    a=c.allocation(attempt);m=read(a['manifest'])
    if m['execution_attempt']!=attempt:raise ValueError('Manifest allocation mismatch')
    check_files(c.ROOT,m);c.plan();c.index(attempt);inherited()
    return m


def imported_sources(attempt='001'):
    manifest=verify(attempt);result={}
    for name,module in tuple(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if not name.startswith('experiments.') or not path:continue
        p=Path(path).resolve()
        if not p.is_relative_to(c.ROOT):raise ValueError('Import outside canonical checkout')
        rel=str(p.relative_to(c.ROOT))
        if rel not in manifest['files'] or sha(p)!=manifest['files'][rel]:raise ValueError('Unmanifested import: '+rel)
        result[name]=rel
    return result


def runtime(cuda=False,attempt='001'):
    import torch
    from experiments.mamba3_three_time.provenance import upstream
    expected=read(c.pilot_path('fixed'))['runtime']
    names=dict(torch='torch',recbole='recbole',mamba_ssm='mamba-ssm',triton='triton',numpy='numpy',tilelang='tilelang',apache_tvm_ffi='apache-tvm-ffi')
    actual={k:importlib.metadata.version(v) for k,v in names.items()}
    pin=upstream()
    if any(actual[k]!=expected[k] for k in names) or torch.version.cuda!=expected['cuda']:raise ValueError('Environment drift')
    if pin['pinned_commit']!=c.PIN or pin['manifest_sha256']!=expected['upstream_manifest_sha256']:raise ValueError('Upstream drift')
    if cuda and (not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name()):
        raise ValueError('Exactly one visible A100 required')
    return dict(**actual,pinned_commit=c.PIN,upstream_manifest_sha256=pin['manifest_sha256'],cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if cuda else None,imported_sources=imported_sources(attempt))


def bindings(commit,m,attempt='001'):
    a=c.allocation(attempt)
    return dict(study_id=c.STUDY,execution_attempt=attempt,execution_commit=commit,source_hash=m['source_hash'],
                source_manifest_sha256=sha(a['manifest']),source_index_sha256=sha(a['source_index']),
                plan_sha256=sha(c.HERE/'study_plan.json'),core_hash=c.CORE,pinned_commit=c.PIN,
                policy_version='mimo_numeric_acceptance_v1',policy_sha256=c.POLICY_SHA,
                inherited_gate_sha256=c.GATE_SHA,inherited_smoke_sha256=c.SMOKE_SHA,
                pilot_execution_commit=c.PILOT_COMMIT,pilot_job_id='4362620',
                architecture='MIMO',backend='upstream',rank=4,chunk=8,TEST='NOT_RUN',test_evaluation_count=0)


def login_verify(attempt='001'):
    import subprocess
    def git(*args):return subprocess.check_output(['git',*args],cwd=c.ROOT,text=True).strip()
    m=verify(attempt);commit=git('rev-parse','HEAD')
    if git('branch','--show-current')!=c.BRANCH or git('status','--porcelain','--untracked-files=no'):
        raise ValueError('Expected clean tracked confirmation branch')
    if git('rev-parse','origin/'+c.BRANCH)!=commit:raise ValueError('Exact commit not published')
    subprocess.run(['git','merge-base','--is-ancestor',c.PUBLICATION,commit],cwd=c.ROOT,check=True)
    def blobs(manifest,execution):
        for path,expected in manifest['files'].items():
            if hashlib.sha256(subprocess.check_output(['git','show',execution+':'+path],cwd=c.ROOT)).hexdigest()!=expected:
                raise ValueError('Execution blob mismatch: '+path)
    blobs(m,commit)
    for path,execution in [
        ('experiments/mamba3_head_timescales/source_manifest_002.json',c.PILOT_COMMIT),
        ('experiments/mamba3_head_timescales/source_manifest.json','5eac7e22c05230f04dd87f3336dc426e093f6bd8'),
        ('experiments/mamba3_mimo_time/confirmation/source_manifest.json','5670e898ed04924a929756e52f39d1d00eb79c5a'),
        ('experiments/mamba3_mimo_time/source_manifest.json','c1dd31eec7907c67348769b6a1811b89aa4011c0')]:
        blobs(read(c.ROOT/path),execution)
    return dict(**bindings(commit,m,attempt),status='PASS',tracked_clean=True,published_commit=commit,
                source_blobs_verified=True,historical_manifests_verified=True,verified_at=now(),runtime=runtime(False,attempt))


def validate_ownership(login,reservation,login_sha,expected,job,operational=None):
    attempt=expected['execution_attempt']
    if not re.fullmatch('[0-9]+',job):raise ValueError('Actual Slurm ID required')
    if any(any(r.get(k)!=v for k,v in expected.items()) for r in (login,reservation)):raise ValueError('Execution binding mismatch')
    tasks=[{k:e[k] for k in ('variant','seed','run_id')} for e in c.index(attempt)['entries'] if e['attempt']==attempt]
    if (login.get('status')!='PASS' or not login.get('tracked_clean') or not login.get('source_blobs_verified')
        or login.get('published_commit')!=expected['execution_commit'] or reservation.get('login_sha256')!=login_sha
        or reservation.get('max_scientific_fits')!=12 or reservation.get('tasks')!=tasks
        or reservation.get('planned_tasks')!=c.tasks() or reservation.get('jobs_requested')!=1
        or reservation.get('allocation_seconds')!=c.plan()['allocations'][attempt]
        or reservation.get('status')!='RESERVED' or not re.fullmatch('[0-9a-f]{32}',reservation.get('token',''))):
        raise ValueError('Invalid immutable reservation limits')
    # sbatch may start the allocation before the login process records the returned ID.
    if operational is not None:
        if operational.get('token')!=reservation['token']:raise ValueError('Wrong submission owner')
        if operational.get('job_id') not in (None,job):raise ValueError('Wrong submitted job')
        if operational.get('job_id') is None and operational.get('status')!='SUBMISSION_UNKNOWN_NO_RETRY':
            raise ValueError('Invalid submission intent')


def identity(attempt='001'):
    a=c.allocation(attempt);m=verify(attempt);login,reservation=read(a['login']),read(a['reservation'])
    commit=os.environ.get('RUN_COMMIT','')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact execution commit required')
    if os.environ.get('EXPECTED_STUDY_HASH')!=m['source_hash'] or os.environ.get('RESERVATION_TOKEN')!=reservation['token']:
        raise ValueError('Environment owner mismatch')
    if os.environ.get('CONFIRMATION_ATTEMPT')!=attempt:raise ValueError('Environment allocation mismatch')
    expected=bindings(commit,m,attempt);job=os.environ.get('SLURM_JOB_ID','')
    validate_ownership(login,reservation,sha(a['login']),expected,job,read(a['submission']))
    if a['lock'].exists():
        lock=read(a['lock'])
        if lock.get('job_id')!=job or lock.get('reservation_token')!=reservation['token']:raise ValueError('Another allocation owns pipeline')
    return dict(**expected,job_id=job,reservation_token=reservation['token'],reservation_sha256=sha(a['reservation']),login_verification_sha256=sha(a['login']))


def require_inherited(base,attempt):
    p=c.allocation(attempt)['inherited'];r=read(p)
    if r.get('status')!='PASS' or any(r.get(k)!=v for k,v in base.items()) or r.get('inherited')!=inherited():
        raise ValueError('Missing/foreign inherited admission for this allocation')
    return sha(p)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--attempt',default='001',choices=['001','002'])
    args=p.parse_args();m=freeze(args.attempt) if args.freeze else verify(args.attempt)
    print(json.dumps(dict(source_hash=m['source_hash'],files=len(m['files']))))
