"""Finite allocation: gates, four sequential fits, report; no scheduler calls/retries."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback

from .config import CORE, GPU_EVIDENCE, HERE, ROOT, RUNTIME, STUDY_ID, SUMMARY, paths, plan
from .provenance import atomic_json, exclusive, now, require_submission


def child_env(stage_dir):
    cache = stage_dir / 'cache'
    env = dict(os.environ)
    for key, suffix in dict(TRITON_CACHE_DIR='triton', TORCHINDUCTOR_CACHE_DIR='inductor',
            XDG_CACHE_HOME='xdg', TORCH_EXTENSIONS_DIR='extensions', CUDA_CACHE_PATH='cuda',
            MPLCONFIGDIR='matplotlib', TMPDIR='tmp').items():
        target = cache / suffix
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    return env


def run_child(module, args, stage_dir, deadline):
    stage_dir.mkdir(parents=True, exist_ok=True)
    with (stage_dir / 'stdout.log').open('x') as out, (stage_dir / 'stderr.log').open('x') as err:
        process = subprocess.Popen([sys.executable, '-m', module, *args], cwd=stage_dir,
            env=child_env(stage_dir), stdout=out, stderr=err, start_new_session=True)
        try:
            code = process.wait(timeout=max(1, deadline-time.time()))
            if code:
                raise RuntimeError(f'{module} exited {code}; see {stage_dir}/stderr.log')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def main():
    manifest = require_submission()
    identity = dict(study_id=STUDY_ID, job_id=os.environ['SLURM_JOB_ID'], execution_commit=os.environ['RUN_COMMIT'],
                    source_hash=manifest['source_hash'], status='RUNNING', started_at=now(), stages=[])
    if GPU_EVIDENCE.exists() or SUMMARY.exists():
        raise FileExistsError('Prior GPU evidence/summary exists')
    for task in plan()['tasks']:
        p = paths(task)
        if any(p[k].exists() for k in ('result', 'lock', 'checkpoint', 'metadata')):
            raise FileExistsError('Prior scientific artifact/lock exists')
    exclusive(RUNTIME / 'pipeline.lock', identity)
    output = RUNTIME / 'pipeline_status.json'
    deadline = time.time() + plan()['budget']['pipeline_seconds']
    os.environ['PIPELINE_DEADLINE'] = str(deadline)
    active_task = None
    failed = False
    def terminate(signum, frame):
        raise TimeoutError(f'Allocation interrupted: {signum}')
    signal.signal(signal.SIGTERM, terminate)
    try:
        for stage in ('preflight', 'gpu_checks'):
            require_submission()
            identity['stage'] = stage
            atomic_json(output, identity)
            run_child('experiments.mamba3_input_time.' + stage, [], RUNTIME / stage, deadline)
            identity['stages'].append(dict(stage=stage, status='PASS'))
        for task in plan()['tasks']:
            if deadline-time.time() < plan()['budget']['minimum_start_remaining_seconds']:
                raise TimeoutError('Insufficient remaining allocation budget to start another fit')
            active_task = task
            require_submission()
            identity['stage'] = task['mode']
            atomic_json(output, identity)
            run_child('experiments.mamba3_input_time.run', ['--mode', task['mode']], paths(task)['runtime'], deadline)
            row = json.loads(paths(task)['result'].read_text())
            if row.get('status') != 'PASS':
                raise ValueError('Child exited without successful completed result')
            identity['stages'].append(dict(stage=task['mode'], status='PASS'))
            active_task = None
    except BaseException as exc:
        failed = True
        identity.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        if identity.get('stage') == 'gpu_checks' and GPU_EVIDENCE.exists():
            evidence = json.loads(GPU_EVIDENCE.read_text())
            if evidence.get('status') == 'RUNNING':
                evidence.update(status='GPU_GATE_FAIL', pipeline_error=repr(exc), finished_at=now())
                atomic_json(GPU_EVIDENCE, evidence)
        if active_task is not None:
            p = paths(active_task)['result']
            row = json.loads(p.read_text()) if p.exists() else dict(**active_task, study_id=STUDY_ID,
                execution_commit=identity['execution_commit'], job_id=identity['job_id'],
                source_hash=manifest['source_hash'], core_hash=CORE, test_status='NOT_RUN', test_evaluation_count=0)
            if row.get('status') != 'PASS':
                row.update(status='FAIL', pipeline_error=repr(exc), stage=row.get('stage', 'STARTUP'), finished_at=now())
                atomic_json(p, row)
    finally:
        for task in plan()['tasks']:
            p = paths(task)['result']
            if not p.exists():
                atomic_json(p, dict(**task, study_id=STUDY_ID, status='NOT_RUN', stage='NOT_RUN',
                                    execution_commit=identity['execution_commit'], job_id=identity['job_id'],
                                    source_hash=manifest['source_hash'], core_hash=CORE, test_status='NOT_RUN', test_evaluation_count=0,
                                    error='Previous mandatory stage failed or budget exhausted'))
        try:
            # Reserved margin permits partial reporting after the scientific deadline.
            run_child('experiments.mamba3_input_time.report', [], RUNTIME / 'report', max(deadline, time.time()+180))
            identity['stages'].append(dict(stage='report', status='PASS'))
        except BaseException as exc:
            failed = True
            identity.update(status='FAIL', reporting_error=repr(exc), reporting_traceback=traceback.format_exc())
            partial = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else dict(study_id=STUDY_ID)
            partial.update(status='INCOMPLETE', reporting_error=repr(exc), test_evaluation_count=0)
            atomic_json(SUMMARY, partial)
            md = HERE / 'runs/pilot_summary.md'
            md.write_text('# Неполный input-time pilot\n\nОшибка формирования отчёта; победитель не выбран.\n\n'
                          + f'`{exc!r}`\n\nДиагностика: `../slurm_logs/pipeline_status.json`. TEST=0.\n')
        identity.update(status='FAIL' if failed else 'PASS', finished_at=now(), test_evaluation_count=0)
        atomic_json(output, identity)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
