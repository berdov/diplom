"""Durable exclusive reservation followed by exactly one sbatch, never polling."""
import argparse
import json
import os
import re
import subprocess
import traceback
from .config import ROOT, LOGS, SUBMISSION, HERE, unused
from .provenance import verify, create_record, atomic_json, sha, now, CORE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    commit = parser.parse_args().commit
    def git(*args):
        return subprocess.check_output(['git', '--no-optional-locks', *args], cwd=ROOT, text=True).strip()
    if str(ROOT) != '/home/daryumin/iberdov/diplom':
        raise ValueError('Canonical cluster checkout only')
    if git('rev-parse', 'HEAD') != commit or git('branch', '--show-current') != 'exp/mamba3-three-time':
        raise ValueError('Unexpected exact checkout')
    if git('diff', 'HEAD', '--'):
        raise ValueError('Tracked checkout not clean')
    manifest = verify()
    for filename in ('login_preflight.json', 'cpu_tests.json'):
        value = json.loads((LOGS / filename).read_text())
        if value.get('status') != 'PASS' or value.get('source_hash') != manifest['source_hash']:
            raise ValueError('Current checks required: ' + filename)
    unused(include_submission=True)
    record = dict(status='RESERVED', reserved_at=now(), execution_commit=commit, source_hash=manifest['source_hash'],
                  core_hash=CORE, policy_sha256=sha(HERE / 'numeric_acceptance_v1.json'),
                  plan_sha256=sha(HERE / 'study_plan.json'), source_manifest_sha256=sha(HERE / 'source_manifest.json'),
                  jobs_submitted=None, max_scientific_fits=2, scientific_fits_before_submit=0, TEST='NOT_RUN',
                  test_evaluation_count=0, git_status_before_submit=git('status', '--short'))
    create_record(SUBMISSION, record)
    env = dict(os.environ, RUN_COMMIT=commit, EXPECTED_STUDY_HASH=manifest['source_hash'], EXPECTED_CORE_HASH=CORE)
    try:
        response = subprocess.run(['sbatch', '--parsable', 'slurm/mamba3_three_time_validation_pilot.sh'],
                                  cwd=ROOT, env=env, text=True, capture_output=True, check=False)
        record.update(stdout=response.stdout, stderr=response.stderr, returncode=response.returncode)
        match = re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?', response.stdout.strip())
        if response.returncode != 0 or match is None:
            record['status'] = 'SUBMISSION_UNKNOWN_NO_RETRY'
            atomic_json(SUBMISSION, record)
            raise RuntimeError('Ambiguous/failed response; reservation retained; no retry')
        record.update(status='SUBMITTED', job_id=match.group(1), jobs_submitted=1)
        atomic_json(SUBMISSION, record)
        print(json.dumps(record, indent=2), flush=True)
    except BaseException:
        if record['status'] == 'RESERVED':
            record.update(status='SUBMISSION_UNKNOWN_NO_RETRY', traceback=traceback.format_exc())
            atomic_json(SUBMISSION, record)
        raise


if __name__ == '__main__':
    main()
