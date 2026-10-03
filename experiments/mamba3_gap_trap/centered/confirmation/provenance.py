"""Exact pilot admission inheritance; two separately owned allocations at most."""
import hashlib
import os
import re
import subprocess
from . import config as c
from .. import config as pilot, provenance as pilot_p
from ... import provenance as parent
from ..reuse import bind
from experiments.mamba3_mimo_time.records import read, sha, digest, create, accepted_cases, now


def inherited():
    m=pilot_p.verify()
    if m['source_hash']!=c.PARENT_SOURCE:raise ValueError('Published centered implementation drift')
    gate,smoke=read(pilot.GATE),read(pilot.SMOKE)
    expected=dict(status='PASS',job_id='4371876',execution_commit=c.PARENT_EXECUTION,
                  source_hash=c.PARENT_SOURCE,execution_attempt='002',TEST='NOT_RUN',test_evaluation_count=0)
    if any(any(r.get(k)!=v for k,v in expected.items()) for r in (gate,smoke)):raise ValueError('Centered pilot admission identity')
    spec=pilot.plan()['required_cases']
    if not accepted_cases(gate['cases'],spec):raise ValueError('Centered pilot gate leaves')
    pilot_p.validate_smoke(smoke)
    if smoke['targeted_gate_sha256']!=sha(pilot.GATE):raise ValueError('Centered smoke/gate binding')
    return dict(status='INHERITED_PASS',previous=pilot_p.inherited(),centered=dict(
        job_id='4371876',execution_commit=c.PARENT_EXECUTION,source_hash=c.PARENT_SOURCE,
        manifest_sha256=sha(pilot.MANIFEST),gate_sha256=sha(pilot.GATE),smoke_sha256=sha(pilot.SMOKE),
        cases=len(spec),checks=sum(len(x['required_keys']) for x in spec),repeated_gpu_checks=0))


def freeze():
    inherited();c.plan();c.index()
    files=dict(read(pilot.MANIFEST)['files'])
    sources=[*c.HERE.glob('*.py'),*(c.HERE/'tests').glob('*.py'),c.HERE/'DESIGN.md',c.PLAN,c.INDEX,c.LAUNCHER,
             pilot.MANIFEST,pilot.GATE,pilot.SMOKE,*[c.pilot_path(v) for v in c.MODES],*[c.historical(s) for s in c.SEEDS]]
    for path in sources:files[str(path.relative_to(c.ROOT))]=sha(path)
    value=dict(files=files,source_hash=digest(files),publication_commit=c.PUBLICATION,core_hash=c.CORE,
               parent_execution=c.PARENT_EXECUTION,parent_manifest_sha256=sha(pilot.MANIFEST))
    create(c.MANIFEST,value)
    return value


_engine=bind(parent,dict(c=c,inherited=inherited),__package__)
verify,runtime,imported_sources=(_engine[k] for k in ('verify','runtime','imported_sources'))


def bindings(commit,manifest,attempt='001'):
    c.allocation(attempt)
    value=_base_bindings(commit,manifest)
    value.update(execution_attempt=attempt,pilot_execution_commit=c.PARENT_EXECUTION,pilot_job_id='4371876',
                 inherited_admission_sha256=sha(pilot.GATE),inherited_smoke_sha256=sha(pilot.SMOKE),source_index_sha256=sha(c.INDEX))
    return value


_base_bindings=_engine['bindings']
_engine.update(bindings=bindings)


def login_verify(attempt):
    value=_engine['login_verify']()
    value['execution_attempt']=attempt
    for name,expected in read(pilot.MANIFEST)['files'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',c.PARENT_EXECUTION+':'+name],cwd=c.ROOT)).hexdigest()!=expected:
            raise ValueError('Centered execution blob mismatch: '+name)
    c.index()
    value['centered_execution_blobs_verified']=True
    return value


def validate_ownership(login,reservation,login_sha,expected,job,operational=None):
    if not re.fullmatch('[0-9]+',job):raise ValueError('Actual Slurm ID required')
    if any(any(r.get(k)!=v for k,v in expected.items()) for r in (login,reservation)):raise ValueError('Execution binding mismatch')
    tasks=reservation.get('tasks')
    if (login.get('status')!='PASS' or login.get('tracked_clean') is not True or login.get('source_blobs_verified') is not True
        or login.get('centered_execution_blobs_verified') is not True or login.get('published_commit')!=expected['execution_commit']
        or reservation.get('login_sha256')!=login_sha or reservation.get('max_scientific_fits')!=8
        or reservation.get('jobs_requested')!=1 or reservation.get('status')!='RESERVED'
        or not re.fullmatch('[0-9a-f]{32}',reservation.get('token','')) or not tasks):raise ValueError('Invalid reservation')
    planned=c.tasks()
    if expected['execution_attempt']=='001':
        if tasks!=planned or reservation.get('continuation_admission_sha256') is not None:raise ValueError('Primary reservation scope')
    else:
        admission=read(c.HERE/'runtime/continuation_admission.json')
        if (reservation.get('continuation_admission_sha256')!=sha(c.HERE/'runtime/continuation_admission.json')
            or tasks!=admission['remaining_tasks'] or admission['execution_commit']!=expected['execution_commit']
            or admission['source_hash']!=expected['source_hash']):raise ValueError('Continuation binding')
    if tasks!=planned[-len(tasks):]:raise ValueError('Only fixed-order suffix may be reserved')
    if operational is not None:
        if operational.get('token')!=reservation['token'] or operational.get('job_id') not in (None,job):raise ValueError('Wrong job owner')
        if operational.get('job_id') is None and operational.get('status')!='SUBMISSION_UNKNOWN_NO_RETRY':raise ValueError('Invalid submission intent')


def identity(attempt):
    m=verify();a=c.allocation(attempt);login,reservation=read(a['login']),read(a['reservation'])
    commit=os.environ.get('RUN_COMMIT','');job=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact execution commit required')
    if os.environ.get('EXPECTED_STUDY_HASH')!=m['source_hash'] or os.environ.get('RESERVATION_TOKEN')!=reservation['token']:
        raise ValueError('Environment owner mismatch')
    expected=bindings(commit,m,attempt)
    validate_ownership(login,reservation,sha(a['login']),expected,job,read(a['submission']))
    if a['lock'].exists():
        owner=read(a['lock'])
        if owner.get('job_id')!=job or owner.get('reservation_token')!=reservation['token']:raise ValueError('Pipeline already owned')
    return dict(**expected,job_id=job,reservation_token=reservation['token'],reservation_sha256=sha(a['reservation']),login_verification_sha256=sha(a['login']))


def validate_record_owner(record):
    a=c.allocation(record['execution_attempt']);m=verify()
    expected=bindings(record['execution_commit'],m,record['execution_attempt'])
    login,reservation,submission=(read(a[k]) for k in ('login','reservation','submission'))
    validate_ownership(login,reservation,sha(a['login']),expected,record['job_id'],submission)
    full=dict(**expected,job_id=record['job_id'],reservation_token=reservation['token'],reservation_sha256=sha(a['reservation']),login_verification_sha256=sha(a['login']))
    if any(record.get(k)!=v for k,v in full.items()):raise ValueError('Foreign scientific record')
    task=dict(variant=record['gap_trap_mode'],seed=record['seed'],run_id=record['run_id'])
    if task not in reservation['tasks']:raise ValueError('Unreserved fit')


def require_inherited(base):
    path=c.allocation(base['execution_attempt'])['inherited'];r=read(path)
    if r.get('status')!='PASS' or any(r.get(k)!=v for k,v in base.items()) or r.get('inherited')!=inherited():
        raise ValueError('Missing or foreign inherited admission')
    return sha(path)


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    m=freeze() if args.freeze else verify();print(json.dumps(dict(files=len(m['files']),source_hash=m['source_hash'])))
