"""Compact scientific status for monitoring without rereading epoch histories."""
from experiments.mamba3_mimo_time.records import create,read,update

OWNER_FIELDS=('execution_attempt','study_id','execution_commit','source_hash','job_id','run_id','memory_mode','seed')


def write(record,paths):
    if any(record.get(k) is None for k in OWNER_FIELDS):raise ValueError('Incomplete progress owner')
    value={k:record.get(k) for k in ('execution_attempt','study_id','execution_commit','source_hash','job_id','run_id','memory_mode','seed','status','stage','scientific_fit_started','actual_epochs','started_at','finished_at','error')}
    value['last_epoch']=record['history'][-1]['epoch'] if record.get('history') else None
    path=paths['runtime']/'progress.json'
    if path.is_symlink():raise ValueError('Progress path must not be a symlink')
    if path.exists():
        owner=read(path)
        if any(owner.get(k)!=value[k] for k in OWNER_FIELDS):raise ValueError('Progress belongs to another run or execution')
        update(path,value)
    else:
        create(path,value)
