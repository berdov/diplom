"""One allocation; inherited gates, seven fresh children, immutable parent runs."""
import json
import os
import signal
import time
import traceback
from experiments.mamba3_context_time.pipeline import run_child
from . import config as c, resume_config as r, resume_provenance as q
from .provenance import create_record, atomic_json, now


def main():
    base = q.identity()
    r.unused()
    create_record(r.LOCK, base)
    state = dict(**base, status='RUNNING', stages=[], scientific_fits_this_attempt=0,
                 previous_completed_confirmation_fits=1, total_completed_confirmation_fits=1,
                 new_diagnostic_batches=0, new_diagnostic_optimizer_steps=0, started_at=now())
    output = r.LOGS/'pipeline_status.json'
    create_record(output, state)
    deadline = time.time()+c.plan()['budget']['pipeline_seconds']
    os.environ['PIPELINE_DEADLINE'] = str(deadline)
    failed = False
    def terminate(signum, frame):
        raise TimeoutError(f'Allocation interrupted: {signum}')
    signal.signal(signal.SIGTERM, terminate)
    try:
        state['stage'] = 'resume_preflight'
        atomic_json(output, state)
        run_child(__package__+'.resume_preflight', ['--allocation'], r.LOGS/'preflight',
                  min(deadline, time.time()+c.plan()['budget']['setup_stage_seconds']))
        state['stages'].append(dict(stage='inherited_gates', status='PASS', evidence=q.parent_ready()))
        for t in r.plan()['tasks']:
            if deadline-time.time() < c.plan()['budget']['minimum_start_remaining_seconds']:
                raise TimeoutError('Insufficient remaining time; no config changes')
            q.inherited_gates(base)
            state['stage'] = t['run_id']
            atomic_json(output, state)
            paths = r.paths(t['mode'], t['seed'])
            run_child(__package__+'.runner', ['--resume', '--mode', t['mode'], '--seed', str(t['seed'])], paths['runtime'], deadline)
            if q.resolve(t['mode'], t['seed'], base)['status'] != 'PASS':
                raise ValueError('Child exited without completed PASS')
            state['stages'].append(dict(stage=t['run_id'], status='PASS'))
    except BaseException as exc:
        failed = True
        state.update(status='INCOMPLETE' if isinstance(exc, TimeoutError) else 'FAIL', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        rows = []
        for t in r.plan()['tasks']:
            path = r.paths(t['mode'], t['seed'])['result']
            if not path.exists():
                create_record(path, dict(**base, **t, status='NOT_RUN', scientific_fit_started=False,
                                         error=state.get('error','Earlier stage incomplete')))
            row = json.loads(path.read_text())
            if row['status'] == 'RUNNING':
                row.update(status='INCOMPLETE', error='Child interrupted', finished_at=now())
                atomic_json(path, row)
            rows.append(row)
        state['scientific_fits_this_attempt'] = sum(x.get('scientific_fit_started',False) for x in rows)
        state['total_completed_confirmation_fits'] = 1+sum(x['status']=='PASS' for x in rows)
        try:
            run_child(__package__+'.resume_report', [], r.LOGS/'report', max(deadline,time.time()+180))
            state['stages'].append(dict(stage='report', status='PASS'))
        except BaseException as exc:
            failed = True
            state.update(reporting_error=repr(exc), reporting_traceback=traceback.format_exc())
            if not r.SUMMARY.exists():
                create_record(r.SUMMARY, dict(**base, status='INCOMPLETE', error=repr(exc)))
        state.update(status=state['status'] if failed and state['status']!='RUNNING' else 'FAIL' if failed else 'PASS', finished_at=now())
        atomic_json(output,state)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
