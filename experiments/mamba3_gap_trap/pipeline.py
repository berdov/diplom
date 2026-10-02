"""One allocation: evidence, targeted gate, smoke, two fresh fits, summary."""
import os
import signal
import subprocess
import sys
import time
import traceback
from . import config as c
from .process_env import child_environment
from .provenance import identity,inherited,runtime,require_stage
from experiments.mamba3_mimo_time.records import create,update,read,now
from experiments.mamba3_mimo_time.process_env import visibility


def child(stage,module,args,deadline,directory):
    directory.mkdir(parents=True,exist_ok=True)
    environment=child_environment(stage,os.environ)
    create(directory/'child_environment.json',dict(stage=stage,parent_cuda_visibility=visibility(os.environ),child_cuda_visibility=visibility(environment)))
    with (directory/'stdout.log').open('xb') as out,(directory/'stderr.log').open('xb') as err:
        proc=subprocess.Popen([sys.executable,'-B','-m',module,*args],cwd=c.ROOT,env=environment,
                              stdout=out,stderr=err,start_new_session=True)
        try:
            code=proc.wait(timeout=max(1,deadline-time.time()))
        except BaseException:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    if code:raise RuntimeError(f'{stage} exit {code}; see {directory}/stderr.log')


def main():
    base=identity();create(c.LOGS/'pipeline.lock',base)
    result=dict(**base,status='RUNNING',stages=[],started_at=now(),scientific_fits_started=0,scientific_fits_completed=0,
                allocation_cuda_visibility=visibility(os.environ))
    create(c.PIPELINE,result)
    budget=c.plan()['budget']
    deadline=min(float(os.environ.get('SLURM_JOB_END_TIME','inf')),time.time()+budget['allocation_seconds'])-budget['save_margin_seconds']
    os.environ['PIPELINE_DEADLINE']=str(deadline)
    result['deadline_unix']=deadline
    def terminate(signum,frame):raise TimeoutError(f'Pipeline interrupted: {signum}')
    signal.signal(signal.SIGTERM,terminate)
    try:
        create(c.INHERITED,dict(**base,status='PASS',inherited=inherited(),runtime=runtime(True),verified_at=now()))
        require_stage(c.INHERITED,base)
        for stage in ('gate','smoke',*c.MODES):
            fit=stage in c.MODES
            if fit and deadline-time.time()<budget['min_remaining_to_start_fit_seconds']:
                raise TimeoutError('Insufficient time for another unchanged full fit; no retry')
            row=dict(stage=stage,status='RUNNING',started_at=now())
            result['stages'].append(row);update(c.PIPELINE,result)
            module='experiments.mamba3_gap_trap.'+('runner' if fit else stage)
            directory=c.paths(stage)['runtime']/'process' if fit else c.LOGS/stage
            child(stage,module,['--variant',stage] if fit else [],deadline,directory)
            if fit:
                r=read(c.paths(stage)['result'])
                if r['status']!='PASS' or any(r.get(k)!=v for k,v in base.items()):
                    raise ValueError('Incomplete/foreign scientific result')
                from .report import validate_record,validate_checkpoint
                validate_record(r,stage);validate_checkpoint(r,stage)
                result['scientific_fits_completed']+=1
                result['scientific_fits_started']=result['scientific_fits_completed']
            else:require_stage(c.GATE if stage=='gate' else c.SMOKE,base)
            row.update(status='PASS',finished_at=now());update(c.PIPELINE,result)
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='INCOMPLETE' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'FAIL',error=repr(exc),traceback=traceback.format_exc())
        if result['stages'] and result['stages'][-1]['status']=='RUNNING':result['stages'][-1]['status']=result['status']
    finally:
        from .report import write
        try:
            summary=write(base,result.get('error'))
            result.update(summary_status=summary['status'],scientific_fits_started=summary['scientific_fits_started'],scientific_fits_completed=summary['scientific_fits_completed'])
            if summary['status']!='PASS' and result['status']=='PASS':result['status']='INCOMPLETE'
        except BaseException:result.update(status='FAIL',report_traceback=traceback.format_exc())
        result['finished_at']=now();update(c.PIPELINE,result)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
