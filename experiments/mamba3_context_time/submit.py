"""Login-only durable exactly-once submission. Never poll or retry."""
import json
import os
import re
import subprocess

from .config import CORE, GPU_EVIDENCE, ROOT, RUNTIME, SUMMARY, paths, plan
from .provenance import atomic_json, exclusive, now, verify


def submit_once(execute=subprocess.run):
    manifest = verify()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'exp/mamba3-context-time' or commit != os.environ.get('RUN_COMMIT'):
        raise ValueError('Wrong branch/exact commit')
    if subprocess.check_output(['git', 'status', '--porcelain', '-uno'], cwd=ROOT, text=True):
        raise ValueError('Tracked changes; no submission')
    if (os.environ.get('EXPECTED_CORE_HASH'), os.environ.get('EXPECTED_STUDY_HASH')) != (CORE, manifest['source_hash']):
        raise ValueError('Expected source hashes required')
    for task in plan()['tasks']:
        p = paths(task)
        if any(p[k].exists() for k in ('result', 'lock', 'checkpoint', 'metadata')):
            raise FileExistsError('Prior run evidence/lock; no submission')
    if any(p.exists() for p in (GPU_EVIDENCE, SUMMARY, RUNTIME/'pipeline.lock', RUNTIME/'gpu_gate/gate.lock')):
        raise FileExistsError('Prior pipeline evidence')
    record = RUNTIME / 'submission_001.json'
    command = ['sbatch', '--parsable', 'slurm/mamba3_context_time.sh']
    payload = dict(status='PREPARED', commit=commit, source_hash=manifest['source_hash'], core_hash=CORE,
                   tasks=plan()['tasks'], prepared_at=now(), command=command, automatic_retry=False)
    exclusive(record, payload)
    try:
        answer = execute(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
        payload.update(returncode=answer.returncode, stdout=answer.stdout, stderr=answer.stderr)
        match = re.fullmatch(r'(\d+)(?:;[A-Za-z0-9_.-]+)?\s*', answer.stdout)
        if answer.returncode != 0 or match is None:
            raise RuntimeError('Submission failed or ambiguous; do not retry')
        payload.update(status='SUBMITTED', job_id=match.group(1))
    except BaseException as exc:
        payload.update(status='SUBMISSION_UNKNOWN', error=repr(exc))
        raise
    finally:
        atomic_json(record, payload)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    submit_once()
