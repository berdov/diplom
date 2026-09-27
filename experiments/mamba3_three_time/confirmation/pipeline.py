"""One finite allocation, fresh children, stop-on-failure and partial reporting."""
import json
import os
import signal
import time
import traceback
from experiments.mamba3_context_time.pipeline import run_child
from experiments.mamba3_three_time.records_003 import interrupt_file
from .config import LOGS, BATCH, INIT, SUMMARY, SUBMISSION, LOCK, paths, plan, unused
from .provenance import identity, create_record, atomic_json, now, gates, require_evidence


def reservation(base):
    r=json.loads(SUBMISSION.read_text())
    if any(r.get(k)!=base[k] for k in ('execution_commit','source_hash','core_hash')):
        raise ValueError('Reservation source mismatch')
    if r['status'] not in ('RESERVED','SUBMITTED') or r.get('job_id') not in (None,base['job_id']):
        raise ValueError('Allocation does not own reservation')


def main():
    base=identity()
    reservation(base)
    unused()
    create_record(LOCK,base)
    state=dict(**base,status='RUNNING',stages=[],scientific_fits=0,started_at=now())
    output=LOGS/'pipeline_status.json'
    create_record(output,state)
    deadline=time.time()+plan()['budget']['pipeline_seconds']
    os.environ['PIPELINE_DEADLINE']=str(deadline)
    failed=False
    def terminate(signum,frame):
        raise TimeoutError(f'Allocation interrupted: {signum}')
    signal.signal(signal.SIGTERM,terminate)
    try:
        for stage in ('preflight','one_batch','initialization'):
            state['stage']=stage
            atomic_json(output,state)
            run_child(__package__+'.'+stage, ['--allocation'] if stage=='preflight' else [], LOGS/stage,
                      min(deadline,time.time()+plan()['budget']['setup_stage_seconds']))
            if stage=='one_batch':
                require_evidence(BATCH,base)
            if stage=='initialization':
                gates(base)
            state['stages'].append(dict(stage=stage,status='PASS'))
        for t in plan()['tasks']:
            if deadline-time.time()<plan()['budget']['minimum_start_remaining_seconds']:
                raise TimeoutError('Insufficient time to start next fit; scientific settings unchanged')
            gates(base)
            state['stage']=t['run_id']
            atomic_json(output,state)
            p=paths(t['mode'],t['seed'])
            run_child(__package__+'.runner',['--mode',t['mode'],'--seed',str(t['seed'])],p['runtime'],deadline)
            if json.loads(p['result'].read_text())['status']!='PASS':
                raise ValueError('Child exited without completed scientific PASS')
            state['stages'].append(dict(stage=t['run_id'],status='PASS'))
    except BaseException as exc:
        failed=True
        state.update(status='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAIL',error=repr(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        for p in (BATCH,INIT):
            interrupt_file(p)
        for t in plan()['tasks']:
            p=paths(t['mode'],t['seed'])['result']
            if not p.exists():
                create_record(p,dict(**base,**t,status='NOT_RUN',scientific_fit_started=False,
                                     error=state.get('error','Earlier stage incomplete')))
            else:
                r=json.loads(p.read_text())
                if r['status']=='RUNNING':
                    r.update(status='INCOMPLETE',error='Child interrupted before finalization',finished_at=now())
                    atomic_json(p,r)
        state['scientific_fits']=sum(json.loads(paths(t['mode'],t['seed'])['result'].read_text()).get('scientific_fit_started',False) for t in plan()['tasks'])
        try:
            run_child(__package__+'.report',[],LOGS/'report',max(deadline,time.time()+180))
            state['stages'].append(dict(stage='report',status='PASS'))
        except BaseException as exc:
            failed=True
            state.update(reporting_error=repr(exc),reporting_traceback=traceback.format_exc())
            if not SUMMARY.exists():
                create_record(SUMMARY,dict(**base,status='INCOMPLETE',error=repr(exc)))
        state.update(status=state['status'] if failed and state['status']!='RUNNING' else 'FAIL' if failed else 'PASS',finished_at=now())
        atomic_json(output,state)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
