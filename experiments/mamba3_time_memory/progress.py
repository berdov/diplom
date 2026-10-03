"""Compact scientific status for monitoring without rereading epoch histories."""
from experiments.mamba3_mimo_time.records import update


def write(record,paths):
    value={k:record.get(k) for k in ('execution_attempt','study_id','execution_commit','source_hash','job_id','run_id','memory_mode','seed','status','stage','scientific_fit_started','actual_epochs','started_at','finished_at','error')}
    value['last_epoch']=record['history'][-1]['epoch'] if record.get('history') else None
    update(paths['runtime']/'progress.json',value)
