"""Fixed ordered fresh processes; safe deadline pause is distinct from failure."""
import argparse
import os
import signal
import time
import traceback
from . import config as c, report
from .provenance import identity, inherited, runtime, require_inherited
from ..process_env import child_environment
from ..reuse import bind
from ... import pipeline as parent
from experiments.mamba3_mimo_time.records import create, update, read, sha, now
from experiments.mamba3_mimo_time.process_env import visibility

child=bind(parent,dict(c=c,child_environment=child_environment))['child']


def may_start(deadline,clock):
    return deadline-clock>=c.plan()['min_remaining_to_start_fit_seconds']


def run(attempt):
    base=identity(attempt);a=c.allocation(attempt);create(a['lock'],base)
    result=dict(**base,status='RUNNING',stages=[],started_at=now(),scientific_fits_started=0,scientific_fits_completed=0,
                allocation_cuda_visibility=visibility(os.environ))
    create(a['pipeline'],result)
    try:
        budget=c.plan();start=time.time()
        hard_deadline=min(float(os.environ.get('SLURM_JOB_END_TIME','inf')),start+budget['allocations'][attempt])
        deadline=hard_deadline-budget['save_margin_seconds'];os.environ['PIPELINE_DEADLINE']=str(deadline)
        result.update(hard_deadline_unix=hard_deadline,deadline_unix=deadline)
        def terminate(signum,frame):raise TimeoutError(f'Pipeline interrupted: {signum}')
        signal.signal(signal.SIGTERM,terminate)
        records=report.completed_prefix(base)
        if attempt=='002':
            from .continuation import check_preserved
            check_preserved(read(c.HERE/'runtime/continuation_admission.json'),base['execution_commit'],base['source_hash'])
        remaining=c.tasks()[len(records):]
        if read(a['reservation'])['tasks']!=remaining:raise ValueError('Reserved suffix changed before pipeline')
        result.update(scientific_fits_started=len(records),scientific_fits_completed=len(records))
        create(a['inherited'],dict(**base,status='PASS',inherited=inherited(),runtime=runtime(True),verified_at=now()))
        require_inherited(base)
        for task in remaining:
            clock=time.time()
            if not may_start(deadline,clock):
                result.update(status='PAUSED_DEADLINE',deadline_guard=dict(checked_unix=clock,remaining_seconds=deadline-clock,
                    min_remaining_seconds=budget['min_remaining_to_start_fit_seconds'],next_task=task),
                    reason='Safe allocation budget exhausted between fits; never-started suffix only')
                break
            report.assert_next_task(task['variant'],task['seed'],base)
            row=dict(**task,status='RUNNING',started_at=now());result['stages'].append(row);update(a['pipeline'],result)
            p=c.paths(task['variant'],task['seed'])
            child(task['variant'],__package__+'.runner',['--variant',task['variant'],'--seed',str(task['seed']),'--attempt',attempt],deadline,p['runtime']/'process')
            record=report.successful(task,base)
            if task['variant']=='centered_gap_trap':
                from .state import paired_records
                fixed=report.previous_records(task['variant'],task['seed'],base)[0]
                paired_records(fixed,record,first_batch=True)
                create(c.RUNS/f'pair_seed{task["seed"]}.json',dict(status='PASS',seed=task['seed'],TEST='NOT_RUN',test_evaluation_count=0,
                    fixed_run_id=fixed['run_id'],centered_run_id=record['run_id'],
                    fixed_result_sha256=sha(c.paths('fixed_replay',task['seed'])['result']),centered_result_sha256=sha(p['result']),
                    shared={k:record[k] for k in ('initial_backbone_sha256','initial_common_calibrator_hashes','rng_components',
                        'first_train_batch_sha256','protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings')},
                    historical_fixed_replay=report.replay_check(fixed)))
            row.update(status='PASS',finished_at=now());result['scientific_fits_completed']+=1
            result['scientific_fits_started']=result['scientific_fits_completed'];update(a['pipeline'],result)
        else:result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
        if result['stages'] and result['stages'][-1]['status']=='RUNNING':result['stages'][-1]['status']='FAIL'
    finally:
        try:
            summary=report.write(base,result.get('error') or result.get('reason'))
            result.update(summary_status=summary['status'],scientific_fits_started=summary['scientific_fits_started'],unknown_scientific_starts=summary['unknown_scientific_starts'],scientific_fits_completed=summary['scientific_fits_completed'],complete_pairs=summary['complete_pairs'])
            if summary['validation_errors'] or summary['unknown_scientific_starts']:raise ValueError('Invalid or unknown-start record; no continuation')
            if result['status']=='PASS' and summary['status']!='PASS':raise ValueError('Full pipeline requires 8/8 valid fits')
        except BaseException:result.update(status='FAIL',report_traceback=traceback.format_exc())
        result['finished_at']=now();update(a['pipeline'],result)
        checkpoints={}
        for task in c.tasks():
            path=c.paths(task['variant'],task['seed'])['checkpoint']
            if path.exists():checkpoints[task['run_id']]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
        create(a['terminal'],dict(**base,status=result['status'],pipeline_sha256=sha(a['pipeline']),
            scientific_fits_started=result['scientific_fits_started'],scientific_fits_completed=result['scientific_fits_completed'],
            checkpoints=checkpoints,checkpoint_loading=False,finished_at=now()))
    return 0 if result['status'] in ('PASS','PAUSED_DEADLINE') else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',choices=['001','002'],required=True);args=p.parse_args()
    try:return run(args.attempt)
    except BaseException as exc:
        job=os.environ.get('SLURM_JOB_ID','unknown')
        create(c.allocation(args.attempt)['logs']/('startup_failure_'+job+'.json'),dict(status='FAIL',stage='STARTUP',
            job_id=job,identity_verified=False,error=repr(exc),traceback=traceback.format_exc(),TEST='NOT_RUN',test_evaluation_count=0))
        return 1


if __name__=='__main__':raise SystemExit(main())
