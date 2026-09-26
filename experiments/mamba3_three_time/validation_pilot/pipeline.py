"""One finite allocation, no retries or scheduler calls, partial evidence on failure."""
import json
import os
import signal
import time
import traceback
from experiments.mamba3_context_time.pipeline import run_child
from experiments.mamba3_three_time.records_003 import interrupt_file
from .config import MODES, LOGS, GATE, SMOKE, SUMMARY, SUBMISSION, LOCK, paths, plan, unused
from .provenance import identity, create_record, atomic_json, require_gate, now


def require_reservation(base):
    saved = json.loads(SUBMISSION.read_text())
    for key in ('execution_commit', 'source_hash', 'core_hash'):
        if saved.get(key) != base[key]:
            raise ValueError('Submission reservation identity mismatch')
    # A job may start before sbatch returns to its caller; exclusive reservation is already durable.
    if saved['status'] not in ('RESERVED', 'SUBMITTED'):
        raise ValueError('Submission is not reserved')
    if saved.get('job_id') not in (None, base['job_id']):
        raise ValueError('Allocation does not own reservation')


def main():
    base = identity()
    require_reservation(base)
    unused()
    state = dict(**base, status='RUNNING', started_at=now(), stages=[], scientific_fits=0)
    create_record(LOCK, base)
    output = LOGS / 'pipeline_status.json'
    create_record(output, state)
    deadline = time.time() + plan()['budget']['pipeline_seconds']
    os.environ['PIPELINE_DEADLINE'] = str(deadline)
    failed = False
    def terminate(signum, frame):
        raise TimeoutError(f'Allocation interrupted: {signum}')
    signal.signal(signal.SIGTERM, terminate)
    try:
        for stage, seconds in (('gate', 'gate_seconds'), ('smoke', 'smoke_seconds')):
            state['stage'] = stage
            atomic_json(output, state)
            run_child(__package__ + '.' + stage, [], LOGS / stage,
                      min(deadline, time.time() + plan()['budget'][seconds]))
            state['stages'].append(dict(stage=stage, status='PASS'))
        for mode in MODES:
            if deadline-time.time() < plan()['budget']['minimum_start_remaining_seconds']:
                raise TimeoutError('Insufficient remaining budget; epochs are not shortened')
            require_gate()
            state['stage'] = mode
            atomic_json(output, state)
            run_child(__package__ + '.runner', ['--mode', mode], paths(mode)['runtime'], deadline)
            row = json.loads(paths(mode)['result'].read_text())
            if row['status'] != 'PASS':
                raise ValueError('Scientific child exited without complete PASS')
            state['stages'].append(dict(stage=mode, status='PASS'))
    except BaseException as exc:
        failed = True
        state.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        interrupt_file(GATE)
        interrupt_file(SMOKE)
        for task in plan()['tasks']:
            path = paths(task['mode'])['result']
            if not path.exists():
                create_record(path, dict(**base, **task, status='NOT_RUN', scientific_fit_started=False,
                                        error='Mandatory earlier stage failed or budget exhausted'))
            else:
                row = json.loads(path.read_text())
                if row['status'] == 'RUNNING':
                    row.update(status='INCOMPLETE', finished_at=now(), error='Child interrupted before finalization')
                    atomic_json(path, row)
        state['scientific_fits'] = sum(json.loads(paths(m)['result'].read_text()).get('scientific_fit_started', False) for m in MODES)
        try:
            run_child(__package__ + '.report', [], LOGS / 'report', max(deadline, time.time()+180))
            state['stages'].append(dict(stage='report', status='PASS'))
        except BaseException as exc:
            failed = True
            state.update(reporting_error=repr(exc), reporting_traceback=traceback.format_exc())
            if not SUMMARY.exists():
                create_record(SUMMARY, dict(**base, status='INCOMPLETE', error=repr(exc)))
            if not SUMMARY.with_suffix('.md').exists():
                with SUMMARY.with_suffix('.md').open('x') as stream:
                    stream.write('# Неполный SISO pilot\n\nОшибка формирования сводки. '
                                 'См. slurm_logs/pipeline_status.json. Победитель не выбран. TEST=0.\n')
        state.update(status='FAIL' if failed else 'PASS', finished_at=now())
        atomic_json(output, state)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
