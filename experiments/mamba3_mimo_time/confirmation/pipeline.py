"""One allocation, twelve fresh sequential processes, durable partial summary."""
import os
import signal
import subprocess
import sys
import time
import traceback
from . import config as c
from .records import create, update, now, read
from .provenance import identity, inherited, runtime, require_inherited
from experiments.mamba3_mimo_time.process_env import child_environment, visibility


def child(stage, module, args, deadline, directory):
    directory.mkdir(parents=True,exist_ok=True)
    environment = child_environment(stage,os.environ)
    create(directory/'child_environment.json',dict(stage=stage,parent_cuda_visibility=visibility(os.environ),child_cuda_visibility=visibility(environment)))
    with (directory/'stdout.log').open('xb') as out, (directory/'stderr.log').open('xb') as err:
        proc = subprocess.Popen([sys.executable,'-B','-m',module,*args],cwd=c.ROOT,stdout=out,stderr=err,start_new_session=True,env=environment)
        try:
            code = proc.wait(timeout=max(1,deadline-time.time()))
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            raise
    if code:
        raise RuntimeError(f'{stage} exit {code}; see {directory}/stderr.log')


def main():
    base = identity()
    create(c.LOGS/'pipeline.lock',base)
    result = dict(**base,status='RUNNING',stages=[],started_at=now(),scientific_fits_started=0,scientific_fits_completed=0,
                  allocation_cuda_visibility=visibility(os.environ))
    create(c.PIPELINE,result)
    budget = c.plan()['budget']
    deadline = min(float(os.environ.get('SLURM_JOB_END_TIME','inf')),time.time()+budget['allocation_seconds'])-budget['save_margin_seconds']
    os.environ['PIPELINE_DEADLINE'] = str(deadline)
    result.update(deadline_unix=deadline,min_remaining_to_start_fit_seconds=budget['min_remaining_to_start_fit_seconds'])
    def terminate(signum,frame):
        raise TimeoutError(f'Pipeline interrupted: {signum}')
    signal.signal(signal.SIGTERM,terminate)
    try:
        row = dict(stage='runtime_inherited_admission',status='RUNNING',started_at=now())
        result['stages'].append(row)
        update(c.PIPELINE,result)
        create(c.INHERITED,dict(**base,status='PASS',inherited=inherited(),runtime=runtime(True),verified_at=now()))
        require_inherited(base)
        row.update(status='PASS',finished_at=now())
        update(c.PIPELINE,result)
        for task in c.tasks():
            if deadline-time.time() < budget['min_remaining_to_start_fit_seconds']:
                raise TimeoutError('Insufficient budget for another unchanged full fit; no retry')
            row = dict(task,status='RUNNING',started_at=now())
            result['stages'].append(row)
            update(c.PIPELINE,result)
            directory = c.LOGS/task['run_id']/'process'
            child(task['mode'],'experiments.mamba3_mimo_time.confirmation.runner',
                  ['--mode',task['mode'],'--seed',str(task['seed'])],deadline,directory)
            run = read(c.paths(task['mode'],task['seed'])['result'])
            if run['status'] != 'PASS' or any(run.get(k) != v for k,v in base.items()):
                raise ValueError('Child did not complete a valid owned result')
            result['scientific_fits_completed'] += 1
            result['scientific_fits_started'] = result['scientific_fits_completed']
            row.update(status='PASS',finished_at=now())
            update(c.PIPELINE,result)
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='INCOMPLETE' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'FAIL',error=repr(exc),traceback=traceback.format_exc())
        if result['stages'] and result['stages'][-1]['status']=='RUNNING':
            result['stages'][-1]['status'] = result['status']
    finally:
        from .report import write
        try:
            summary = write(base)
            result.update(summary_status=summary['status'],scientific_fits_started=summary['scientific_fits_started'],
                          scientific_fits_completed=summary['scientific_fits_completed'])
            if summary['status'] != 'PASS' and result['status']=='PASS':
                result['status'] = 'INCOMPLETE'
        except BaseException:
            result.update(status='FAIL',report_traceback=traceback.format_exc())
        result['finished_at'] = now()
        update(c.PIPELINE,result)
    return 0 if result['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
