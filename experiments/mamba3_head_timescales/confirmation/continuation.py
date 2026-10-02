"""One conservative continuation of never-started runs; never a metric retry."""
import re
from . import config as c
from experiments.mamba3_mimo_time.records import read,sha


def eligible_records(records):
    remaining=[]
    forbidden=('nonfinite','gradient','parity','config drift','initialization','pair mismatch','outofmemory','nan/inf')
    for task,r in zip(c.tasks(),records,strict=True):
        if r.get('status')=='PASS':continue
        if r.get('scientific_fit_started') is not False or r.get('history') or r.get('first_train_batch_sha256') or r.get('checkpoint_sha256'):
            raise ValueError('Started or uncertain incomplete fit blocks all autonomous continuation')
        if r.get('status') not in ('NOT_RUN','FAIL','INCOMPLETE'):raise ValueError('Unclassified run state')
        error=str(r.get('error','')).lower()
        if r.get('status')!='NOT_RUN' and not error:raise ValueError('Failure cause not recorded')
        if any(word in error for word in forbidden):raise ValueError('Scientific/numerical failure is not retryable')
        remaining.append(task['run_id'])
    if not remaining:raise ValueError('No unstarted logical runs remain')
    return remaining


def verify_authorization():
    """A terminal read-only audit creates this narrow proof before a second submit."""
    proof=read(c.RUNTIME/'continuation_authorization.json')
    if proof.get('reason_class') not in ('deadline_before_start','infrastructure_before_fit','serialization_before_fit'):
        raise ValueError('Continuation reason not allowed')
    a=c.allocation('001');submission=read(a['submission'])
    if not re.fullmatch('[0-9]+',str(submission.get('job_id',''))):raise ValueError('Unambiguous submitted parent job required')
    if proof.get('parent_job_id')!=submission.get('job_id'):raise ValueError('Wrong terminal parent')
    terminal=read(c.ROOT/proof['scheduler_snapshot'])
    if sha(c.ROOT/proof['scheduler_snapshot'])!=proof['scheduler_snapshot_sha256']:raise ValueError('Terminal snapshot changed')
    if terminal.get('job_id')!=submission['job_id'] or terminal.get('state') not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):
        raise ValueError('Parent allocation not proven terminal')
    preserve=read(c.ROOT/proof['preservation_manifest'])
    if sha(c.ROOT/proof['preservation_manifest'])!=proof['preservation_manifest_sha256']:raise ValueError('Preservation proof changed')
    if not preserve.get('files'):raise ValueError('Missing original evidence preservation')
    for original,entry in preserve['files'].items():
        if sha(c.ROOT/original)!=entry['sha256'] or sha(c.ROOT/entry['copy'])!=entry['sha256']:
            raise ValueError('Original/archived evidence changed')
    records=[read(c.paths(t['variant'],t['seed'],'001')['result']) for t in c.tasks()]
    from .report import validate_record,validate_checkpoint
    for task,record in zip(c.tasks(),records,strict=True):
        if record.get('status')=='PASS':
            validate_record(record,task)
            validate_checkpoint(record,dict(task,attempt='001'))
            if record.get('job_id')!=submission['job_id'] or record.get('reservation_sha256')!=sha(a['reservation']):
                raise ValueError('Successful parent belongs to another allocation')
    remaining=eligible_records(records)
    expected={str(c.paths(t['variant'],t['seed'],'001')['result'].relative_to(c.ROOT)) for t in c.tasks()}
    expected|={str(a[k].relative_to(c.ROOT)) for k in ('reservation','submission','pipeline','inherited','summary','lock')}
    expected|={str(a[k].relative_to(c.ROOT)) for k in ('login','manifest','source_index')}
    expected.add(str(a['summary'].with_suffix('.md').relative_to(c.ROOT)))
    for path in a['logs'].rglob('*'):
        if path.is_file() and not any(x in path.parts for x in ('tilelang_cache','log_tensorboard')) and path.suffix in ('.json','.log','.out','.err','.lock'):
            expected.add(str(path.relative_to(c.ROOT)))
    if not expected.issubset(preserve['files']):raise ValueError('Incomplete preservation coverage')
    if proof.get('remaining_run_ids')!=remaining or proof.get('regression_status')!='PASS' or not proof.get('regression_evidence'):
        raise ValueError('Missing exact-failure regression')
    regression=read(c.ROOT/proof['regression_evidence'])
    if regression.get('status')!='PASS' or regression.get('source_hash')!=read(c.allocation('002')['manifest'])['source_hash']:
        raise ValueError('Regression source mismatch')
    if sha(c.ROOT/proof['regression_evidence'])!=proof['regression_evidence_sha256'] or not proof.get('failure_path'):
        raise ValueError('Concrete failure/regression binding missing')
    if sha(c.ROOT/proof['failure_path'])!=proof['failure_sha256']:raise ValueError('Original failure evidence changed')
    before,after=read(a['manifest']),read(c.allocation('002')['manifest'])
    if proof.get('parent_source_hash')!=before['source_hash'] or proof.get('parent_execution_commit')!=read(a['reservation'])['execution_commit']:
        raise ValueError('Parent execution/source binding missing')
    prefix=str(c.HERE.relative_to(c.ROOT))+'/'
    allowed={prefix+name for name in ('pipeline.py','report.py','submit.py','provenance.py','continuation.py','handoff.py','preflight.py')}
    allowed.add(str(c.LAUNCHER.relative_to(c.ROOT)))
    changed={p for p in set(before['files'])|set(after['files']) if before['files'].get(p)!=after['files'].get(p)}
    extra={prefix+'source_manifest.json',prefix+'source_index_002.json'}
    if any(p not in allowed|extra and not p.startswith(prefix+'tests/') for p in changed):
        raise ValueError('Continuation changed scientific/config/state/runner dependencies: '+repr(sorted(changed)))
    if proof.get('reviewed_changed_files')!=sorted(changed):raise ValueError('Changes not explicitly reviewed')
    selected=c.index('002')['entries']
    if [e['run_id'] for e in selected if e['attempt']=='002']!=remaining:raise ValueError('Continuation source index repeats/skips tasks')
    if sum(r.get('scientific_fit_started') is True for r in records)+len(remaining)>12:raise ValueError('Scientific budget exceeded')
    return dict(parent_job_id=submission['job_id'],remaining_run_ids=remaining,
                preservation_manifest_sha256=proof['preservation_manifest_sha256'],reason_class=proof['reason_class'])
