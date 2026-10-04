"""Durable reservations; bounded pilot, conditional confirmation, one pre-fit retry."""
import json
import os
import re
import secrets
import subprocess
import traceback
from . import config as c
from .provenance import login_verify, verify, bindings, coverage_verify, confirmation_authorization
from experiments.mamba3_mimo_time.records import read, create, update, sha, now


def retry_review():
    if c.EXECUTION_ATTEMPT == '001':
        if (c.HERE/'slurm_logs'/c.STAGE/'attempt_002/reservation.json').exists():
            raise ValueError('Retry already reserved; initial phase cannot restart')
        return None
    review_path = c.HERE/'runtime'/('retry_review_'+c.STAGE+'.json')
    r = read(review_path)
    if (r.get('status') != 'APPROVED_PRE_FIT_TECHNICAL_RETRY' or r.get('scientific_fits_started') != 0
        or r.get('unknown_scientific_starts') != 0 or r.get('method_unchanged') is not True
        or r.get('tolerances_unchanged') is not True or r.get('regression_passed') is not True
        or not r.get('infrastructure_error') or r.get('old_job_terminal') is not True):
        raise ValueError('Retry outside authorized conditions')
    old = c.HERE/'slurm_logs'/c.STAGE/'attempt_001'
    if read(old/'submission.json').get('job_id') != r['old_job_id']:
        raise ValueError('Wrong retry parent job')
    scheduler = r['scheduler']
    if scheduler['JobIDRaw'] != r['old_job_id'] or scheduler['State'] not in (
        'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'COMPLETED'):
        raise ValueError('Missing terminal parent proof')
    if sha(c.ROOT/r['preservation_manifest_path']) != r['preservation_manifest_sha256']:
        raise ValueError('Preserved failure changed')
    for task in c.tasks():
        name = task['run_id']
        path = c.HERE/'runs'/c.STAGE/'attempt_001'/(name+'.json')
        owner = old/name
        if not path.exists():
            if any((owner/x).exists() for x in ('run.lock', 'progress.json', 'checkpoints')):
                raise ValueError('Unknown prior scientific start')
        else:
            record = read(path)
            if record.get('scientific_fit_started') is not False or record.get('history') or record.get('actual_epochs', 0):
                raise ValueError('Scientific fit already started; retry forbidden')
    return sha(review_path)


def budget_review():
    previous = [read(p) for p in (c.HERE/'slurm_logs').glob('*/*/reservation.json')]
    for r in previous:
        stage, attempt = r.get('study_phase'), r.get('execution_attempt')
        if r.get('study_id') != c.STUDY or stage not in ('pilot', 'confirmation') or attempt not in ('001', '002'):
            raise ValueError('Malformed global reservation ledger')
        expected = 14400 if attempt == '002' else 21600 if stage == 'pilot' else 28800
        if (type(r.get('requested_seconds')) is not int or r['requested_seconds'] != expected
            or r.get('max_scientific_fits') != (3 if stage == 'pilot' else 12)):
            raise ValueError('Malformed reserved walltime/fit budget')
    if len(previous) >= 3 or sum(r['requested_seconds'] for r in previous)+c.allocation_seconds() > 64800:
        raise ValueError('Global submission/walltime ceiling exhausted')
    if c.EXECUTION_ATTEMPT == '002' and any(r['execution_attempt'] == '002' for r in previous):
        raise ValueError('Only one technical retry for the entire study')
    if any(r['study_phase'] == c.STAGE and r['execution_attempt'] == c.EXECUTION_ATTEMPT for r in previous):
        raise ValueError('Phase attempt already reserved')
    return dict(previous_submissions=len(previous), requested_seconds_before=sum(r['requested_seconds'] for r in previous),
                requested_seconds_after=sum(r['requested_seconds'] for r in previous)+c.allocation_seconds())


def reserve_and_submit(login, manifest):
    c.unused()
    review, budget, decision = retry_review(), budget_review(), confirmation_authorization()
    create(c.LOGIN, login)
    token = secrets.token_hex(16)
    reservation = dict(**bindings(login['execution_commit'], manifest), status='RESERVED', token=token,
        login_sha256=sha(c.LOGIN), coverage_sha256=coverage_verify(login['execution_commit']),
        confirmation_decision_sha256=decision, max_scientific_fits=len(c.tasks()), tasks=c.tasks(),
        scientific_fits_before_submit=0, jobs_requested=1, requested_seconds=c.allocation_seconds(),
        global_budget=budget, retry_review_sha256=review, reserved_at=now())
    create(c.RESERVATION, reservation)
    operational = dict(token=token, job_id=None, status='SUBMISSION_UNKNOWN_NO_RETRY')
    create(c.SUBMISSION, operational)
    try:
        limit = f'{c.allocation_seconds()//3600:02d}:00:00'
        command = ['sbatch', '--parsable', '--time='+limit,
            '--output='+str((c.LOGS/'m3-absolute-phase-%j.out').relative_to(c.ROOT)),
            '--error='+str((c.LOGS/'m3-absolute-phase-%j.err').relative_to(c.ROOT)),
            str(c.LAUNCHER.relative_to(c.ROOT))]
        env = dict(os.environ, ABS_PHASE_STAGE=c.STAGE, ABS_PHASE_ATTEMPT=c.EXECUTION_ATTEMPT,
                   RUN_COMMIT=login['execution_commit'], EXPECTED_STUDY_HASH=manifest['source_hash'], RESERVATION_TOKEN=token)
        proc = subprocess.run(command, cwd=c.ROOT, capture_output=True, text=True, env=env)
        operational.update(stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
        match = re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?', proc.stdout.strip())
        if proc.returncode != 0 or match is None:
            raise RuntimeError('Ambiguous/failed sbatch; do not repeat')
        operational.update(job_id=match.group(1), status='SUBMITTED', jobs_submitted=budget['previous_submissions']+1)
    except BaseException:
        operational['traceback'] = traceback.format_exc()
        update(c.SUBMISSION, operational)
        raise
    update(c.SUBMISSION, operational)
    return operational


def main():
    if str(c.ROOT) != '/home/daryumin/iberdov/diplom':
        raise ValueError('Canonical cluster checkout required')
    login, manifest = login_verify(), verify()
    for prefix in ('cpu_preflight_', 'no_git_preflight_'):
        r = read(c.LOGS/(prefix+login['execution_commit']+'.json'))
        if (r['status'] != 'PASS' or r['source_hash'] != manifest['source_hash']
            or r['execution_commit'] != login['execution_commit'] or r['execution_attempt'] != c.EXECUTION_ATTEMPT
            or r['study_phase'] != c.STAGE or r['cuda_initialized'] is not False
            or any(r['cpu_tests'][k] for k in ('skipped', 'errors', 'failures'))):
            raise ValueError('Missing exact CPU/no-Git proof')
    result = reserve_and_submit(login, manifest)
    print(json.dumps(dict(job_id=result['job_id'], study_phase=c.STAGE, execution_attempt=c.EXECUTION_ATTEMPT,
        execution_commit=login['execution_commit'], source_hash=manifest['source_hash'],
        reservation=str(c.RESERVATION), submission=str(c.SUBMISSION), scientific_fits_before_submit=0,
        max_scientific_fits=len(c.tasks()), TEST='NOT_RUN'), indent=2), flush=True)


if __name__ == '__main__':
    main()
