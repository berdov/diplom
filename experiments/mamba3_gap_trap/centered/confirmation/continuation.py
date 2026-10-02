"""Read-only admission for one deadline-only continuation, never a fit retry."""
from . import config as c, report
from experiments.mamba3_mimo_time.records import read, sha


def check_preserved(admission,execution,source):
    from .provenance import verify, bindings, validate_ownership
    a=c.allocation('001');m=verify();expected=bindings(execution,m,'001')
    if m['source_hash']!=source:raise ValueError('Continuation manifest drift')
    login,reservation,submission=(read(a[k]) for k in ('login','reservation','submission'))
    validate_ownership(login,reservation,sha(a['login']),expected,submission['job_id'],submission)
    full=dict(**expected,job_id=submission['job_id'],reservation_token=reservation['token'],reservation_sha256=sha(a['reservation']),login_verification_sha256=sha(a['login']))
    pipeline,terminal=(read(a[k]) for k in ('pipeline','terminal'))
    if any(any(r.get(k)!=v for k,v in full.items()) for r in (pipeline,terminal)):raise ValueError('Prior allocation identity')
    if terminal['pipeline_sha256']!=sha(a['pipeline']):raise ValueError('Prior terminal/pipeline binding')
    for key in ('scientific_fits_started','scientific_fits_completed'):
        if terminal[key]!=pipeline[key]:raise ValueError('Prior terminal counts')
    if pipeline.get('unknown_scientific_starts',0):raise ValueError('Unknown prior starts')
    for name,expected_sha in admission['preserved_files'].items():
        if sha(c.ROOT/name)!=expected_sha:raise ValueError('Preserved continuation source changed: '+name)


def validate(admission,execution,source):
    a=c.allocation('001');pipeline=read(a['pipeline']);submission=read(a['submission']);terminal=read(a['terminal'])
    scheduler=admission['scheduler']
    if (admission.get('execution_commit')!=execution or admission.get('source_hash')!=source
        or scheduler.get('job_id')!=submission['job_id'] or scheduler.get('state')!='COMPLETED' or scheduler.get('exit_code')!='0:0'
        or pipeline.get('status')!='PAUSED_DEADLINE' or terminal['status']!='PAUSED_DEADLINE'
        or any(k in pipeline for k in ('error','traceback','report_traceback'))):raise ValueError('Only clean terminal deadline pause permits continuation')
    check_preserved(admission,execution,source)
    if pipeline['execution_commit']!=execution or pipeline['source_hash']!=source:raise ValueError('Continuation source changed')
    guard=pipeline['deadline_guard'];budget=c.plan()
    if (guard['min_remaining_seconds']!=budget['min_remaining_to_start_fit_seconds']
        or guard['remaining_seconds']>=guard['min_remaining_seconds']
        or abs(guard['remaining_seconds']-(pipeline['deadline_unix']-guard['checked_unix']))>1e-6):raise ValueError('Deadline guard proof')
    records=report.completed_prefix()
    if not 0<=len(records)<8 or pipeline['scientific_fits_started']!=len(records) or pipeline['scientific_fits_completed']!=len(records):
        raise ValueError('Any started incomplete fit blocks continuation')
    remaining=c.tasks()[len(records):]
    if guard['next_task']!=remaining[0] or admission['remaining_tasks']!=remaining:raise ValueError('Continuation suffix drift')
    required=[a[k] for k in ('pipeline','terminal','reservation','submission','login','summary')]
    for record in records:
        p=c.paths(record['gap_trap_mode'],record['seed']);required.extend(p[k] for k in ('result','metadata','lock'))
        if record['gap_trap_mode']=='centered_gap_trap':
            pair=c.RUNS/f'pair_seed{record["seed"]}.json';required.append(pair);r=read(pair)
            if r['status']!='PASS' or any(r[label+'_result_sha256']!=sha(c.paths(v,record['seed'])['result']) for label,v in zip(('fixed','centered'),c.MODES)):
                raise ValueError('Pair audit binding')
    declared=admission['preserved_files']
    for path in required:
        if declared.get(str(path.relative_to(c.ROOT)))!=sha(path):raise ValueError('Continuation preservation binding')
    # Include process logs and all compact outputs in the admission manifest.
    actual={str(p.relative_to(c.ROOT)):sha(p) for p in c.LOGS.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.out','.err','.lock') and 'tilelang_cache' not in p.parts and 'attempt_002' not in p.parts}
    if any(declared.get(k)!=v for k,v in actual.items()):raise ValueError('Missing/changed prior process logs')
    return remaining
