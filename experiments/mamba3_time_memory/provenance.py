"""Frozen source, inherited kernel evidence, and durable allocation ownership."""
import hashlib
import math
import os
import re
import subprocess
from . import config as c
from experiments.mamba3_gap_trap import provenance as parent
from experiments.mamba3_gap_trap.centered.reuse import bind
from experiments.mamba3_mimo_time.confirmation.provenance import inherited
from experiments.mamba3_mimo_time.records import read,sha,digest,create,now,accepted_cases


def freeze():
    c.unused();inherited();c.plan()
    files=dict(read(c.PARENT_MANIFEST)['files'])
    for path in [*c.HERE.glob('*.py'),*(c.HERE/'tests').glob('*.py'),c.HERE/'DESIGN.md',c.HERE/'study_plan.json',c.LAUNCHER,c.PARENT_MANIFEST,c.PILOT]:
        files[str(path.relative_to(c.ROOT))]=sha(path)
    value=dict(files=files,source_hash=digest(files),publication_commit=c.PUBLICATION,core_hash=c.CORE)
    create(c.MANIFEST,value);return value


_engine=bind(parent,dict(c=c,inherited=inherited),__package__)
verify,runtime,imported_sources=(_engine[k] for k in ('verify','runtime','imported_sources'))


def bindings(commit,m):
    value=_engine['bindings'](commit,m)
    value.update(execution_attempt=c.EXECUTION_ATTEMPT)
    return value


def coverage_verify(commit=None):
    m=verify();r=read(c.COVERAGE)
    expected=bindings(commit or os.environ.get('RUN_COMMIT',''),m)
    if any(r.get(k)!=v for k,v in expected.items()) or r.get('status')!='PASS':raise ValueError('TRAIN coverage not admitted for exact source')
    scope=dict(split='TRAIN',sampled_examples=10000,train_population=1062567,anchors=[1,4,16,32],reference_ms=838393,max_history=50,model_forward_count=0,model_instances_created=0,train_loaders_created=0,valid_loaders_created=0,test_loaders_created=0,target_fields_read=False,scientific_fits_started=0,test_evaluation_count=0,cuda_initialized_before=False,cuda_initialized_after=False)
    if any(r.get(k)!=v for k,v in scope.items()):raise ValueError('TRAIN-only coverage scope')
    expected_indices=[i*(1062567-1)//(10000-1) for i in range(10000)]
    if r.get('selected_train_indices')!=expected_indices or r.get('selected_indices_sha256')!=digest(expected_indices):raise ValueError('Coverage sample drift')
    old=read(c.PILOT)
    if r['protocol']!=old['protocol'] or r['manifest_sha256']!=old['manifest_sha256'] or r['frozen_train_time_stats_sha256']!=old['train_time_stats_sha256']:raise ValueError('Coverage data identity')
    if not 0<=r['selected_sets_equal_count']<10000 or r['selected_sets_equal_fraction']!=r['selected_sets_equal_count']/10000:raise ValueError('Uninformative or invalid coverage')
    return sha(c.COVERAGE)


def validate_ownership(login,reservation,login_sha,expected,job,operational=None):
    if not re.fullmatch('[0-9]+',job):raise ValueError('Actual Slurm ID required')
    if any(any(r.get(k)!=v for k,v in expected.items()) for r in (login,reservation)):raise ValueError('Execution binding mismatch')
    if (login.get('status')!='PASS' or not login.get('tracked_clean') or not login.get('source_blobs_verified')
        or login.get('published_commit')!=expected['execution_commit'] or reservation.get('login_sha256')!=login_sha
        or reservation.get('max_scientific_fits')!=3 or reservation.get('tasks')!=c.plan()['tasks']
        or reservation.get('jobs_requested')!=1 or reservation.get('status')!='RESERVED'
        or not re.fullmatch('[0-9a-f]{32}',reservation.get('token',''))):raise ValueError('Invalid immutable reservation')
    if not isinstance(login.get('coverage_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',login.get('coverage_sha256','')) or login['coverage_sha256']!=reservation.get('coverage_sha256'):raise ValueError('Reserved TRAIN coverage binding mismatch')
    if operational is not None and (operational.get('token')!=reservation['token'] or operational.get('job_id') not in (None,job)
        or (operational.get('job_id') is None and operational.get('status')!='SUBMISSION_UNKNOWN_NO_RETRY')):raise ValueError('Wrong submitted job owner')


def identity():
    m=verify();login,reservation=read(c.LOGIN),read(c.RESERVATION);commit=os.environ.get('RUN_COMMIT','')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact execution commit required')
    if os.environ.get('EXPECTED_STUDY_HASH')!=m['source_hash'] or os.environ.get('RESERVATION_TOKEN')!=reservation['token']:raise ValueError('Environment owner mismatch')
    expected=bindings(commit,m);job=os.environ.get('SLURM_JOB_ID','')
    validate_ownership(login,reservation,sha(c.LOGIN),expected,job,read(c.SUBMISSION) if c.SUBMISSION.exists() else None)
    if (c.LOGS/'pipeline.lock').exists():
        lock=read(c.LOGS/'pipeline.lock')
        if lock.get('job_id')!=job or lock.get('reservation_token')!=reservation['token']:raise ValueError('Pipeline already owned')
    coverage=coverage_verify(commit)
    if coverage!=login['coverage_sha256']:raise ValueError('TRAIN coverage changed after reservation')
    return dict(**expected,job_id=job,reservation_token=reservation['token'],reservation_sha256=sha(c.RESERVATION),login_verification_sha256=sha(c.LOGIN),coverage_sha256=coverage)


def login_verify():
    def git(*args):return subprocess.check_output(['git',*args],cwd=c.ROOT,text=True).strip()
    m=verify();commit=git('rev-parse','HEAD')
    if git('branch','--show-current')!=c.BRANCH or git('status','--porcelain','--untracked-files=no'):raise ValueError('Clean canonical study branch required')
    if git('rev-parse','origin/'+c.BRANCH)!=commit:raise ValueError('Exact execution not published')
    subprocess.run(['git','merge-base','--is-ancestor',c.PUBLICATION,commit],cwd=c.ROOT,check=True)
    for path,expected in m['files'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path],cwd=c.ROOT)).hexdigest()!=expected:raise ValueError('Published blob '+path)
    return dict(**bindings(commit,m),status='PASS',tracked_clean=True,published_commit=commit,source_blobs_verified=True,verified_at=now(),runtime=runtime(False),coverage_sha256=coverage_verify(commit))


def validate_smoke(r):
    if (r.get('batch'),r.get('history_length'),r.get('kernel_length'),r.get('steps_per_mode'))!=(2048,50,56,3):raise ValueError('Smoke shape/steps')
    if [x.get('memory_mode') for x in r.get('rows',[])]!=list(c.MODES):raise ValueError('Smoke variants')
    for row in r['rows']:
        names=set(c.parameter_keys(row['memory_mode']))
        if row['status']!='PASS' or len(row['steps'])!=3:raise ValueError('Smoke incomplete')
        for i,step in enumerate(row['steps']):
            g=step['gradient_norms']
            if step['step']!=i or not step['finite_loss'] or not math.isfinite(step['loss']) or set(g)!=names or any(not math.isfinite(v) or v<0 for v in g.values()):raise ValueError('Smoke loss/gradients')
        if row['memory_mode']!='no_memory':
            for step in row['steps']:
                if step.get('memory_shape')!=[2048,50,4,64] or not all(step.get(k) is True for k in ('selected_causal','finite_weights','invalid_weights_zero','weight_sums','selected_value_gradient_finite','hook_removed')):raise ValueError('Smoke memory shape/masks')
                if not all(isinstance(step.get(k),(int,float)) and math.isfinite(step[k]) for k in ('beta_before','beta_after','lambda_before','lambda_after','selected_value_gradient_l2')) or step['selected_value_gradient_l2']<0:raise ValueError('Smoke memory numeric evidence')
        if not row['roundtrip_passed'] or row['roundtrip']!='weights_only=True':raise ValueError('Smoke roundtrip')


def require_stage(path,base):
    r=read(path)
    if r.get('status')!='PASS' or any(r.get(k)!=v for k,v in base.items()):raise ValueError('Missing/failed/foreign stage '+str(path))
    if path==c.GATE and not accepted_cases(r.get('cases',[]),c.plan()['required_cases']):raise ValueError('Targeted leaves incomplete')
    if path==c.SMOKE:
        validate_smoke(r)
        if r['targeted_gate_sha256']!=sha(c.GATE):raise ValueError('Wrong smoke gate')
    return sha(path)


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    result=freeze() if args.freeze else verify();print(json.dumps(dict(source_hash=result['source_hash'],files=len(result['files']))))
