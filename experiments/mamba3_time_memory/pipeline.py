"""One reserved allocation: admission, smoke, and three ordered fresh fits."""
import os
import re
import signal
import subprocess
import time
import traceback
from . import config as c
from .process_env import child_environment
from .provenance import identity,inherited,runtime,require_stage
from experiments.mamba3_gap_trap import pipeline as parent
from experiments.mamba3_gap_trap.centered.reuse import bind
from experiments.mamba3_mimo_time.records import read,create,update,sha,now
from experiments.mamba3_mimo_time.process_env import visibility
child=bind(parent,dict(c=c,child_environment=child_environment))['child']


def fit_counters():
    started=completed=unknown=0
    for v in c.MODES:
        paths=c.paths(v)
        if paths['result'].exists():
            try:r=read(paths['result']);started+=r.get('scientific_fit_started') is True;completed+=r.get('status')=='PASS'
            except (ValueError,OSError):unknown+=1
        elif paths['lock'].exists():unknown+=1
    return dict(scientific_fits_started=started,scientific_fits_completed=completed,unknown_scientific_starts=unknown)


def run():
    base=identity();create(c.LOGS/'pipeline.lock',base)
    result=dict(**base,status='RUNNING',stages=[],started_at=now(),**fit_counters(),allocation_cuda_visibility=visibility(os.environ))
    create(c.PIPELINE,result)
    try:
        budget=c.plan()['budget'];allocation=budget['allocation_seconds'] if c.EXECUTION_ATTEMPT=='001' else budget['retry_allocation_seconds']
        deadline=min(float(os.environ.get('SLURM_JOB_END_TIME','inf')),time.time()+allocation)-budget['save_margin_seconds']
        os.environ['PIPELINE_DEADLINE']=str(deadline);result['deadline_unix']=deadline
        def terminate(signum,frame):raise TimeoutError(f'Pipeline interrupted: {signum}')
        signal.signal(signal.SIGTERM,terminate)
        # CPU proofs belong to this exact source/commit; runtime validates compute dependencies.
        for prefix in ('cpu_preflight_','no_git_preflight_'):
            cpu=read(c.LOGS/(prefix+base['execution_commit']+'.json'))
            if cpu['status']!='PASS' or cpu['execution_commit']!=base['execution_commit'] or cpu['source_hash']!=base['source_hash'] or cpu['cuda_initialized'] is not False:raise ValueError('CPU preflight proof')
        create(c.INHERITED,dict(**base,status='PASS',inherited=inherited(),runtime=runtime(True),verified_at=now()))
        require_stage(c.INHERITED,base)
        for stage in ('gate','smoke',*c.MODES):
            fit=stage in c.MODES
            if fit and deadline-time.time()<budget['min_remaining_to_start_fit_seconds']:raise TimeoutError('Insufficient allocation; no continuation/refit')
            row=dict(stage=stage,status='RUNNING',started_at=now());result['stages'].append(row);update(c.PIPELINE,result)
            directory=c.paths(stage)['runtime']/'process' if fit else c.LOGS/stage
            child(stage,'experiments.mamba3_time_memory.'+('runner' if fit else stage),['--variant',stage] if fit else [],deadline,directory)
            if fit:
                from .report import validate_record,validate_checkpoint
                r=read(c.paths(stage)['result'])
                if r['status']!='PASS' or any(r.get(k)!=v for k,v in base.items()):raise ValueError('Incomplete/foreign scientific result')
                validate_record(r,stage);validate_checkpoint(r,stage)
            else:require_stage(c.GATE if stage=='gate' else c.SMOKE,base)
            result.update(fit_counters());row.update(status='PASS',finished_at=now());update(c.PIPELINE,result)
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='INCOMPLETE' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'FAIL',error=repr(exc),traceback=traceback.format_exc())
        if result['stages'] and result['stages'][-1]['status']=='RUNNING':result['stages'][-1]['status']=result['status']
    finally:
        from .report import write
        try:
            summary=write(base,result.get('error'));result.update(summary_status=summary['status'])
            if summary['status']!='PASS' and result['status']=='PASS':result['status']='INCOMPLETE'
        except BaseException:result.update(status='FAIL',report_traceback=traceback.format_exc())
        result.update(fit_counters(),finished_at=now());update(c.PIPELINE,result)
        checkpoints={v:dict(path=str(c.paths(v)['checkpoint']),bytes=c.paths(v)['checkpoint'].stat().st_size,sha256=sha(c.paths(v)['checkpoint'])) for v in c.MODES if c.paths(v)['checkpoint'].exists()}
        terminal={k:v for k,v in result.items() if k!='stages'}
        terminal.update(pipeline_sha256=sha(c.PIPELINE),checkpoints=checkpoints,checkpoint_loading=False)
        create(c.RUNS/'terminal_metadata.json',terminal)
    return 0 if result['status']=='PASS' else 1


def main():
    try:return run()
    except BaseException as exc:
        job=os.environ.get('SLURM_JOB_ID','');label=job if re.fullmatch('[0-9]+',job) else 'unknown'
        path=c.LOGS/('startup_failure_'+label+'.json')
        failure=dict(status='FAIL',stage='STARTUP',identity_verified=False,job_id=job,**fit_counters(),TEST='NOT_RUN',test_evaluation_count=0,error=repr(exc),traceback=traceback.format_exc(),finished_at=now())
        if not path.exists():create(path,failure)
        return 1


if __name__=='__main__':raise SystemExit(main())
