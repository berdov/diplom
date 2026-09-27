"""Finite sequential subprocess pipeline, fail-stop, no retries/nested sbatch."""
import os
import signal
import subprocess
import sys
import time
import traceback
from . import config as c
from .records import create, update, now, read
from .provenance import identity, require_gate
from .process_env import child_environment, visibility


def child(stage, module, args, deadline):
    directory=c.LOGS/stage
    directory.mkdir(parents=True,exist_ok=True)
    environment = child_environment(stage, os.environ)
    create(directory/'child_environment.json', dict(stage=stage,
           parent_cuda_visibility=visibility(os.environ), child_cuda_visibility=visibility(environment)))
    with (directory/'stdout.log').open('xb') as out, (directory/'stderr.log').open('xb') as err:
        proc=subprocess.Popen([sys.executable,'-B','-m',module,*args],cwd=c.ROOT,
                              stdout=out,stderr=err,start_new_session=True,env=environment)
        try:
            code=proc.wait(timeout=max(1,deadline-time.time()))
        except BaseException:
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); proc.wait()
            raise
    if code:
        raise RuntimeError(f'{stage} exit {code}; see {directory}/stderr.log')


def main():
    base=identity()
    create(c.LOGS/'pipeline.lock',base)
    result=dict(**base,status='RUNNING',stages=[],started_at=now(),scientific_fits=0,
                allocation_cuda_visibility=visibility(os.environ))
    create(c.PIPELINE,result)
    end=min(float(os.environ.get('SLURM_JOB_END_TIME','inf')),time.time()+8*3600)
    deadline=end-c.plan()['budget']['save_margin_seconds']
    os.environ['PIPELINE_DEADLINE']=str(deadline)
    result.update(deadline_unix=deadline,min_remaining_to_start_fit_seconds=c.plan()['budget']['min_remaining_to_start_fit_seconds'])
    def terminate(signum,frame):
        raise TimeoutError(f'Pipeline signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    try:
        for stage in ('preflight','admission','smoke',*c.MODES):
            if stage in c.MODES:
                require_gate(base)
                smoke=read(c.SMOKE)
                if smoke['status']!='PASS' or any(smoke.get(k)!=v for k,v in base.items()):
                    raise ValueError('Scientific fit blocked by smoke')
                if deadline-time.time()<c.plan()['budget']['min_remaining_to_start_fit_seconds']:
                    raise TimeoutError('Insufficient remaining budget to start another fit; settings unchanged')
            row=dict(stage=stage,status='RUNNING',started_at=now())
            result['stages'].append(row);update(c.PIPELINE,result)
            module='runner' if stage in c.MODES else stage
            args = ['--mode',stage] if stage in c.MODES else []
            if stage == 'preflight':
                args = ['--evidence', str(c.LOGS/'preflight/evidence.json')]
            child(stage,'experiments.mamba3_mimo_time.'+module,args,deadline)
            if stage in c.MODES:
                run=read(c.paths(stage)['result'])
                if run['status']!='PASS':
                    raise ValueError('Child returned without successful result')
                result['scientific_fits_completed'] = result.get('scientific_fits_completed', 0) + 1
            row.update(status='PASS',finished_at=now());update(c.PIPELINE,result)
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAIL',error=repr(exc),traceback=traceback.format_exc())
        if result['stages'] and result['stages'][-1]['status']=='RUNNING':
            result['stages'][-1]['status']='FAIL'
    finally:
        from .report import write
        try:
            summary = write(base)
            result.update(summary_status=summary['status'], scientific_fits=summary['scientific_fits_started'],
                          scientific_fits_completed=summary['scientific_fits_completed'])
        except BaseException:
            result.update(status='FAIL',report_traceback=traceback.format_exc())
        result['finished_at']=now();update(c.PIPELINE,result)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
