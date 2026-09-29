"""Content checks on compute; Git publication proofs only on login/local."""
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from pathlib import Path
from . import config as c
from experiments.mamba3_mimo_time.records import read,sha,digest,create,now,accepted_cases
from experiments.mamba3_mimo_time.confirmation.provenance import inherited,check_files


def freeze():
    c.unused()
    inherited()
    old=c.ROOT/'experiments/mamba3_mimo_time/confirmation/source_manifest.json'
    files=dict(read(old)['files'])
    files={p:h for p,h in files.items() if not p.startswith('reports/') and p not in ('README.md','experiments/results.csv')}
    for p in [*c.HERE.glob('*.py'),*(c.HERE/'tests').glob('*.py'),c.HERE/'DESIGN.md',c.HERE/'study_plan.json',c.LAUNCHER,old]:
        files[str(p.relative_to(c.ROOT))]=sha(p)
    value=dict(files=files,source_hash=digest(files),publication_commit=c.PUBLICATION,
               inherited_execution=c.PILOT_COMMIT,core_hash=c.CORE)
    create(c.HERE/'source_manifest.json',value)
    return value


def verify():
    value=read(c.HERE/'source_manifest.json')
    check_files(c.ROOT,value)
    c.plan()
    from experiments.mamba3_time_mechanisms.provenance import fingerprint
    if fingerprint()!=c.CORE:
        raise ValueError('Frozen mathematical core drift')
    inherited()
    return value


def imported_sources():
    m=verify(); result={}
    for name,module in tuple(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if not name.startswith('experiments.') or not path:
            continue
        p=Path(path).resolve()
        if not p.is_relative_to(c.ROOT):
            raise ValueError('Imported outside canonical checkout')
        rel=str(p.relative_to(c.ROOT))
        if rel not in m['files'] or sha(p)!=m['files'][rel]:
            raise ValueError('Unmanifested import: '+rel)
        result[name]=rel
    return result


def runtime(cuda=False):
    import torch
    from experiments.mamba3_three_time.provenance import upstream
    expected=read(c.PILOT)['runtime']
    names=dict(torch='torch',recbole='recbole',mamba_ssm='mamba-ssm',triton='triton',numpy='numpy',tilelang='tilelang',apache_tvm_ffi='apache-tvm-ffi')
    actual={k:importlib.metadata.version(v) for k,v in names.items()}
    pin=upstream()
    if any(actual[k]!=expected[k] for k in names) or torch.version.cuda!=expected['cuda']:
        raise ValueError('Existing environment drift')
    if pin['pinned_commit']!=c.PIN or pin['manifest_sha256']!=expected['upstream_manifest_sha256']:
        raise ValueError('Upstream source drift')
    if cuda and (not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name()):
        raise ValueError('Exactly one visible A100 required')
    return dict(**actual,pinned_commit=c.PIN,upstream_manifest_sha256=pin['manifest_sha256'],cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if cuda else None,imported_sources=imported_sources())


def bindings(commit,m):
    return dict(study_id=c.STUDY,execution_commit=commit,source_hash=m['source_hash'],
                source_manifest_sha256=sha(c.HERE/'source_manifest.json'),plan_sha256=sha(c.HERE/'study_plan.json'),
                core_hash=c.CORE,pinned_commit=c.PIN,policy_version='mimo_numeric_acceptance_v1',policy_sha256=c.POLICY_SHA,
                inherited_admission_sha256=c.ADMISSION_SHA,inherited_smoke_sha256=c.SMOKE_SHA,
                pilot_execution_commit=c.PILOT_COMMIT,pilot_job_id='4358147',architecture='MIMO',backend='upstream',
                rank=4,chunk=8,TEST='NOT_RUN',test_evaluation_count=0)


def login_verify():
    import subprocess
    def git(*args):return subprocess.check_output(['git',*args],cwd=c.ROOT,text=True).strip()
    m=verify(); commit=git('rev-parse','HEAD')
    if git('branch','--show-current')!=c.BRANCH or git('status','--porcelain','--untracked-files=no'):
        raise ValueError('Expected clean tracked study branch')
    if git('rev-parse','origin/'+c.BRANCH)!=commit:
        raise ValueError('Exact commit not published')
    subprocess.run(['git','merge-base','--is-ancestor',c.PUBLICATION,commit],cwd=c.ROOT,check=True)
    for name,expected in m['files'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=c.ROOT)).hexdigest()!=expected:
            raise ValueError('Published blob mismatch: '+name)
    for name,execution in [('experiments/mamba3_mimo_time/source_manifest.json',c.PILOT_COMMIT),
                           ('experiments/mamba3_mimo_time/confirmation/source_manifest.json','5670e898ed04924a929756e52f39d1d00eb79c5a')]:
        for p,h in read(c.ROOT/name)['files'].items():
            if hashlib.sha256(subprocess.check_output(['git','show',execution+':'+p],cwd=c.ROOT)).hexdigest()!=h:
                raise ValueError('Historical execution blob mismatch: '+p)
    return dict(**bindings(commit,m),status='PASS',tracked_clean=True,published_commit=commit,
                source_blobs_verified=True,verified_at=now(),runtime=runtime(False))


def validate_ownership(login,reservation,login_sha,expected,job,operational=None):
    if not re.fullmatch('[0-9]+',job):
        raise ValueError('Actual Slurm ID required')
    if any(any(r.get(k)!=v for k,v in expected.items()) for r in (login,reservation)):
        raise ValueError('Execution binding mismatch')
    if (login.get('status')!='PASS' or not login.get('tracked_clean') or not login.get('source_blobs_verified')
        or login.get('published_commit')!=expected['execution_commit'] or reservation.get('login_sha256')!=login_sha
        or reservation.get('max_scientific_fits')!=3 or reservation.get('tasks')!=c.plan()['tasks']
        or reservation.get('jobs_requested')!=1 or reservation.get('status')!='RESERVED'
        or not re.fullmatch('[0-9a-f]{32}',reservation.get('token',''))):
        raise ValueError('Invalid immutable reservation')
    if operational is not None and (operational.get('job_id')!=job or operational.get('token')!=reservation['token']):
        raise ValueError('Wrong submitted job owner')


def identity():
    m=verify(); login,reservation=read(c.LOGIN),read(c.RESERVATION)
    commit=os.environ.get('RUN_COMMIT','')
    if not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Exact execution commit required')
    if os.environ.get('EXPECTED_STUDY_HASH')!=m['source_hash'] or os.environ.get('RESERVATION_TOKEN')!=reservation['token']:
        raise ValueError('Environment owner mismatch')
    expected=bindings(commit,m); job=os.environ.get('SLURM_JOB_ID','')
    validate_ownership(login,reservation,sha(c.LOGIN),expected,job,read(c.SUBMISSION) if c.SUBMISSION.exists() else None)
    if (c.LOGS/'pipeline.lock').exists():
        owner=read(c.LOGS/'pipeline.lock')
        if owner.get('job_id')!=job or owner.get('reservation_token')!=reservation['token']:
            raise ValueError('Another allocation owns this pipeline')
    return dict(**expected,job_id=job,reservation_token=reservation['token'],reservation_sha256=sha(c.RESERVATION),login_verification_sha256=sha(c.LOGIN))


def require_stage(path,base):
    r=read(path)
    if r.get('status')!='PASS' or any(r.get(k)!=v for k,v in base.items()):
        raise ValueError('Missing/failed/foreign stage: '+str(path))
    if path==c.GATE and not accepted_cases(r.get('cases',[]),c.plan()['required_cases']):
        raise ValueError('Missing/failed targeted leaves')
    if path==c.SMOKE:
        validate_smoke(r)
        if r['targeted_gate_sha256']!=sha(c.GATE):
            raise ValueError('Wrong smoke gate')
    return sha(path)


def validate_smoke(r):
    if (r.get('batch'),r.get('history_length'),r.get('kernel_length'),r.get('steps_per_mode'))!=(2048,50,56,3):
        raise ValueError('Smoke shape/steps drift')
    if [x.get('time_scale_mode') for x in r.get('rows',[])]!=list(c.MODES):
        raise ValueError('Missing/duplicate smoke variant')
    for row in r['rows']:
        variant=row['time_scale_mode']
        alpha=set() if variant=='fixed' else set(c.plan()['alpha_keys'])
        names=set(c.plan()['common_parameter_keys'])|alpha
        if row.get('status')!='PASS' or len(row.get('steps',[]))!=3:
            raise ValueError('Incomplete smoke')
        for i,step in enumerate(row['steps']):
            norms=step.get('gradient_norms',{})
            if (step.get('step')!=i or step.get('finite_loss') is not True or not isinstance(step.get('loss'),(int,float))
                or not math.isfinite(step['loss']) or set(norms)!=names
                or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in norms.values())):
                raise ValueError('Missing/nonfinite smoke leaf')
        if any(not any(s['gradient_norms'][n]>0 for s in row['steps'][1:]) for n in names if n.startswith('times.')):
            raise ValueError('Calibrator learning not demonstrated after zero-init')
        if set(row.get('alpha_updated',{}))!=alpha or any(v is not True for v in row['alpha_updated'].values()):
            raise ValueError('Alpha update missing/failed')
        if (row.get('roundtrip_passed') is not True or row.get('roundtrip')!='weights_only=True'
            or not all(isinstance(row.get(k),int) and row[k]>0 for k in ('peak_allocated_bytes','peak_reserved_bytes'))):
            raise ValueError('Smoke roundtrip/memory evidence missing')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    m=freeze() if args.freeze else verify()
    print(json.dumps(dict(source_hash=m['source_hash'],files=len(m['files']))))
