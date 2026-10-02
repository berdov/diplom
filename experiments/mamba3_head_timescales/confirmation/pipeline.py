"""Inherited admission, twelve fresh processes, durable partial summary."""
import os
import signal
import subprocess
import sys
import time
import traceback
from . import config as c
from .provenance import identity,inherited,runtime,require_inherited
from .handoff import save
from experiments.mamba3_head_timescales.process_env import child_environment
from experiments.mamba3_mimo_time.process_env import visibility
from experiments.mamba3_mimo_time.records import create,update,read,now


class StartDeadline(TimeoutError):
    pass


def child(variant,seed,attempt,deadline,directory):
    directory.mkdir(parents=True,exist_ok=True)
    environment=child_environment(variant,os.environ)
    create(directory/'child_environment.json',dict(stage=variant,seed=seed,
           parent_cuda_visibility=visibility(os.environ),child_cuda_visibility=visibility(environment)))
    command=[sys.executable,'-B','-m','experiments.mamba3_head_timescales.confirmation.runner',
             '--variant',variant,'--seed',str(seed),'--attempt',attempt]
    with (directory/'stdout.log').open('xb') as out,(directory/'stderr.log').open('xb') as err:
        proc=subprocess.Popen(command,cwd=c.ROOT,env=environment,stdout=out,stderr=err,start_new_session=True)
        try:code=proc.wait(timeout=max(1,deadline-time.time()))
        except BaseException:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    if code:raise RuntimeError(f'{variant} seed{seed} exit {code}; see {directory}/stderr.log')


def main(attempt='001'):
    a=c.allocation(attempt);base=identity(attempt);create(a['lock'],base)
    result=dict(**base,status='RUNNING',stages=[],started_at=now(),scientific_fits_started=0,scientific_fits_completed=0,
                allocation_cuda_visibility=visibility(os.environ))
    create(a['pipeline'],result)
    budget=c.plan()
    deadline=min(float(os.environ.get('SLURM_JOB_END_TIME','inf')),time.time()+budget['allocations'][attempt])-budget['save_margin_seconds']
    os.environ['PIPELINE_DEADLINE']=str(deadline)
    result['deadline_unix']=deadline
    def terminate(signum,frame):raise TimeoutError(f'Pipeline interrupted: {signum}')
    signal.signal(signal.SIGTERM,terminate)
    try:
        create(a['inherited'],dict(**base,status='PASS',inherited=inherited(),runtime=runtime(True,attempt),verified_at=now()))
        require_inherited(base,attempt)
        from .report import source_record,validate_record,validate_checkpoint
        for entry in c.index(attempt)['entries']:
            if entry['attempt']!=attempt:
                r=source_record(entry,base);validate_record(r,entry);validate_checkpoint(r,entry)
                continue
            if deadline-time.time()<budget['min_remaining_to_start_fit_seconds']:
                raise StartDeadline('Deadline does not permit starting another unchanged full fit')
            variant,seed=entry['variant'],entry['seed']
            row=dict(run_id=entry['run_id'],variant=variant,seed=seed,status='RUNNING',started_at=now())
            result['stages'].append(row);update(a['pipeline'],result)
            child(variant,seed,attempt,deadline,c.paths(variant,seed,attempt)['runtime']/'process')
            r=source_record(entry,base)
            if r['status']!='PASS':raise ValueError('Incomplete scientific result')
            validate_record(r,entry);validate_checkpoint(r,entry)
            result['scientific_fits_completed']+=1
            result['scientific_fits_started']+=1
            row.update(status='PASS',finished_at=now());update(a['pipeline'],result)
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='INCOMPLETE' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'FAIL',
                      error=repr(exc),traceback=traceback.format_exc(),
                      failure_class='deadline_before_start' if isinstance(exc,StartDeadline) else 'requires_terminal_evidence_audit')
        if result['stages'] and result['stages'][-1]['status']=='RUNNING':result['stages'][-1]['status']=result['status']
    finally:
        from .report import write
        try:
            summary=write(base,attempt,result.get('error'))
            result.update(summary_status=summary['status'],scientific_fits_started=summary['scientific_fits_started'],
                          scientific_fits_start_unknown=summary['scientific_fits_start_unknown'],scientific_fits_completed=summary['scientific_fits_completed'])
            if summary['status']!='PASS' and result['status']=='PASS':result['status']='INCOMPLETE'
        except BaseException:result.update(status='FAIL',report_traceback=traceback.format_exc())
        result['finished_at']=now();update(a['pipeline'],result)
        save('pipeline_finished',execution_commit=base['execution_commit'],source_hash=base['source_hash'],
             job_ids=[base['job_id']],execution_attempt=attempt,pipeline_status=result['status'],
             new_fits_started=result['scientific_fits_started'],new_fits_completed=result['scientific_fits_completed'],
             next_safe_step='Wait for terminal scheduler status within poll budget, then audit saved artifacts')
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--attempt',choices=['001','002'],required=True)
    raise SystemExit(main(p.parse_args().attempt))
