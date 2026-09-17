"""Login-node one-shot array submission, durable evidence, no status polling."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone

from .config import CORE, HERE, ROOT, paths, plan
from .provenance import atomic_json, verify


def submit_once(run_command=subprocess.run):
    manifest=verify()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if os.environ.get('RUN_COMMIT')!=commit or not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Exact login-node commit mismatch')
    if os.environ.get('EXPECTED_STUDY_HASH')!=manifest['source_hash'] or os.environ.get('EXPECTED_CORE_HASH')!=CORE:
        raise ValueError('Submission hashes differ')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():
        raise ValueError('Tracked checkout is dirty')
    for task in plan()['tasks']:
        p=paths(task)
        if any(p[k].exists() for k in ('result','lock','smoke_checkpoint')):
            raise FileExistsError('Existing result/lock: no duplicate series')
        if p['checkpoints'].exists() and any(p['checkpoints'].iterdir()):
            raise FileExistsError('Existing checkpoints: no duplicate series')
    directory=HERE/'slurm_logs'
    directory.mkdir(parents=True,exist_ok=True)
    record_path=directory/'submission_001.json'
    record=dict(status='PREPARED',prepared_at=datetime.now(timezone.utc).isoformat(),
                commit=commit,core_fingerprint=CORE,study_source_hash=manifest['source_hash'],
                tasks=plan()['tasks'],command=['sbatch','--parsable','slurm/mamba3_time_confirmation.sh'],
                test_jobs_submitted=0,automatic_retry=False)
    with record_path.open('x') as handle:
        json.dump(record,handle,indent=2)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    # Exactly one scheduler mutation; even ambiguous output leaves a durable no-retry record.
    try:
        completed=run_command(record['command'],cwd=ROOT,text=True,capture_output=True,check=False)
        record.update(returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
        output=completed.stdout.strip()
        if completed.returncode==0 and re.fullmatch(r'\d+(;[\w.-]+)?',output):
            record.update(status='SUBMITTED',array_job_id=output.split(';')[0])
        else:
            record['status']='FAILED_OR_AMBIGUOUS_DO_NOT_RETRY'
        atomic_json(record_path,record)
    except BaseException as exc:
        record.update(status='EXCEPTION_OR_AMBIGUOUS_DO_NOT_RETRY',error=repr(exc))
        atomic_json(record_path,record)
        raise
    print(json.dumps(dict(record=record,record_path=str(record_path)),indent=2))
    if record['status']!='SUBMITTED':
        raise SystemExit(1)
    # No scheduler queries, log reads or source changes after receiving the ID.


if __name__=='__main__':
    submit_once()
