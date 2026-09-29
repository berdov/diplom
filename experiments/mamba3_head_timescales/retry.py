"""One explicitly authorized retry of the exact empty-progress failure."""
from . import config as c
from experiments.mamba3_mimo_time.records import read,sha,accepted_cases

PARENT_JOB='4361071'
PARENT_COMMIT='5eac7e22c05230f04dd87f3336dc426e093f6bd8'
PARENT_SOURCE='a7ef84b343f159c475ccb8a300c025318361b39e65bf3eab1f36d78a85760905'
PRESERVATION_SHA='845c256d29e94a1406eb4250c72f301b0f0e9e3c0e9434e835ae93161337f1c5'
FAILURE_SHA='3e387d38937c6a1339b235b74df35fa06c31a9c3f111f52d597628ae43729b53'
REASON='empty_progress_passed_to_final_case_constructor'
# Only orchestration may differ from the parent. New files are separately hashed.
CHANGED_PARENT_FILES={
    'experiments/mamba3_head_timescales/config.py',
    'experiments/mamba3_head_timescales/gate.py',
    'experiments/mamba3_head_timescales/provenance.py',
    'experiments/mamba3_head_timescales/preflight.py',
    'experiments/mamba3_head_timescales/submit.py',
    'slurm/mamba3_head_timescales.sh',
}


def retry_bindings():
    return dict(execution_attempt=c.EXECUTION_ATTEMPT,retry_of_job=PARENT_JOB,
                parent_execution_commit=PARENT_COMMIT,parent_source_hash=PARENT_SOURCE,
                parent_failure_evidence_sha256=FAILURE_SHA,parent_preservation_manifest_sha256=PRESERVATION_SHA,
                retry_reason=REASON,parent_scientific_fits=0)


def verify_parent(originals=False):
    if sha(c.PRESERVATION)!=PRESERVATION_SHA:raise ValueError('Preservation manifest changed')
    archive=c.PRESERVATION.parent;manifest=read(c.PRESERVATION)
    if (manifest['parent_job_id']!=PARENT_JOB or manifest['parent_execution_commit']!=PARENT_COMMIT
        or manifest['parent_source_hash']!=PARENT_SOURCE or manifest['reason']!=REASON
        or manifest['parent_scientific_fits']!=0 or manifest['parent_synthetic_optimizer_steps']!=0
        or manifest['parent_state']['State']!='FAILED' or manifest['parent_state']['ExitCode']!='1:0'):
        raise ValueError('Unapproved retry parent')
    if sha(archive/'scheduler_snapshot.json')!=manifest['scheduler_sha256']:
        raise ValueError('Parent scheduler evidence changed')
    for name,row in manifest['files'].items():
        p=archive/row['preserved']
        if not p.resolve().is_relative_to(archive.resolve()) or sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:
            raise ValueError('Archived parent changed: '+name)
        if originals and (sha(c.HERE/name)!=row['sha256'] or (c.HERE/name).stat().st_size!=row['bytes']):
            raise ValueError('Original parent changed: '+name)
    parent=archive/'files'
    gate=read(parent/'runs/targeted_gate_001.json');summary=read(parent/'runs/pilot_summary.json')
    if sha(parent/'runs/targeted_gate_001.json')!=FAILURE_SHA:
        raise ValueError('Wrong parent failure')
    specs=read(parent/'study_plan.json')['required_cases'];cases=gate['cases']
    if (gate['status']!='FAIL' or gate['job_id']!=PARENT_JOB or gate['execution_commit']!=PARENT_COMMIT
        or gate['source_hash']!=PARENT_SOURCE or gate['required_cases']!=specs or len(cases)!=7
        or not accepted_cases(cases[:6],specs[:6]) or sum(len(x['checks']) for x in cases[:6])!=198
        or cases[-1]['case_id']!='causality_fixed' or cases[-1]['status']!='FAIL' or cases[-1]['checks']!={}
        or cases[-1]['missing_keys']!=specs[6]['required_keys']
        or 'gradient_forward_started' not in cases[-1]['traceback'] or 'ValueError: Empty required checks' not in cases[-1]['traceback']):
        raise ValueError('Parent is not the authorized empty-progress failure')
    if summary['scientific_fits_started']!=0 or summary['scientific_fits_completed']!=0 or summary['status']!='INCOMPLETE':
        raise ValueError('Parent fit already started')
    for mode in c.MODES:
        name=f'mamba3_headtime_{mode}_seed2026_001'
        r=read(parent/'runs'/(name+'.json'))
        if (r['status']!='NOT_RUN' or r['scientific_fit_started'] is not False or r['history']!=[] or r['actual_epochs']!=0
            or r['job_id']!=PARENT_JOB or r['execution_commit']!=PARENT_COMMIT or r['source_hash']!=PARENT_SOURCE
            or r['time_scale_mode']!=mode or r['mode']!='dual' or r['seed']!=2026 or r['TEST']!='NOT_RUN' or r['test_evaluation_count']!=0):
            raise ValueError('Parent scientific run was used')
        if originals and (c.HERE/'slurm_logs'/name).exists():
            raise ValueError('Parent scientific runtime/weights/lock exists')
    if originals and any(p.exists() for p in (c.HERE/'runs/smoke_001.json',c.HERE/'slurm_logs/smoke')):
        raise ValueError('Unexpected parent smoke')
    old=read(parent/'source_manifest.json')
    if old['source_hash']!=PARENT_SOURCE:raise ValueError('Parent source identity changed')
    changed=[]
    for name,h in old['files'].items():
        if sha(c.ROOT/name)!=h:
            if name not in CHANGED_PARENT_FILES:raise ValueError('Scientific/undeclared source drift: '+name)
            changed.append(name)
    return dict(status='PASS',**retry_bindings(),originals_checked=originals,changed_orchestration=sorted(changed),
                parent_numerical_checks_preserved=198,parent_causality_checks_not_measured=30)


def check_attempt_namespaces(allow_current=False):
    for kind in ('runs','slurm_logs'):
        for p in (c.HERE/kind).glob('attempt_*'):
            if p.name!='attempt_002':raise ValueError('Unknown execution attempt: '+str(p))
    known={c.HERE/'slurm_logs/reservation_001.json'}
    if allow_current:known.add(c.RESERVATION)
    for p in (c.HERE/'slurm_logs').rglob('*'):
        if p.is_file() and p.name.startswith(('reservation','submission')):
            allowed=known|{c.HERE/'slurm_logs/submission_001.json'}
            if allow_current:allowed.add(c.SUBMISSION)
            if p not in allowed:raise ValueError('Unknown or occupied attempt ownership: '+str(p))
    if not allow_current:c.unused()
