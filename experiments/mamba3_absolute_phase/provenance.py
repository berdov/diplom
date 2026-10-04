"""Frozen dependencies, coverage, conditional progression and allocation ownership."""
import hashlib
import math
import os
import re
import subprocess
from . import config as c
from experiments.mamba3_gap_trap import provenance as parent
from experiments.mamba3_gap_trap.centered.reuse import bind
from experiments.mamba3_mimo_time.confirmation.provenance import inherited
from experiments.mamba3_mimo_time.records import read, sha, digest, create, now, accepted_cases


def freeze():
    c.unused()
    inherited()
    c.plan()
    files = dict(read(c.PARENT_MANIFEST)['files'])
    paths = [*c.HERE.glob('*.py'), *(c.HERE/'tests').glob('*.py'), c.HERE/'DESIGN.md',
             c.HERE/'NEW_PLAN.md', c.HERE/'study_plan.json', c.LAUNCHER, c.PARENT_MANIFEST,
             *[c.historical(s) for s in (2026, 2027, 2028, 2029, 2030)]]
    for p in paths:
        files[str(p.relative_to(c.ROOT))] = sha(p)
    value = dict(files=files, source_hash=digest(files), publication_commit=c.PUBLICATION, core_hash=c.CORE)
    create(c.MANIFEST, value)
    return value


_engine = bind(parent, dict(c=c, inherited=inherited), __package__)
verify, runtime, imported_sources = (_engine[k] for k in ('verify', 'runtime', 'imported_sources'))


def bindings(commit, manifest):
    value = _engine['bindings'](commit, manifest)
    value.update(execution_attempt=c.EXECUTION_ATTEMPT, study_phase=c.STAGE)
    return value


def coverage_verify(commit=None):
    manifest = verify()
    r = read(c.COVERAGE)
    expected = bindings(commit or os.environ.get('RUN_COMMIT', ''), manifest)
    if any(r.get(k) != v for k, v in expected.items()) or r.get('status') != 'PASS':
        raise ValueError('TRAIN coverage missing or not bound to exact execution')
    for key, value in dict(split='TRAIN', model_forward_count=0, model_instances_created=0,
                           train_loaders_created=0, valid_loaders_created=0, test_loaders_created=0,
                           target_fields_read=False, scientific_fits_started=0,
                           test_evaluation_count=0, cuda_initialized_before=False,
                           cuda_initialized_after=False).items():
        if r.get(key) != value:
            raise ValueError('Coverage scope ' + key)
    old = read(c.PILOT)
    for key in ('protocol', 'manifest_sha256'):
        if r[key] != old[key]:
            raise ValueError('Coverage data identity ' + key)
    if r['frozen_train_time_stats_sha256'] != old['train_time_stats_sha256']:
        raise ValueError('TRAIN reference identity')
    return sha(c.COVERAGE)


def confirmation_authorization():
    if c.STAGE == 'pilot':
        return None
    path = c.HERE/'runtime/confirmation_decision.json'
    decision = read(path)
    if decision.get('status') != 'AUTHORIZED_BY_FROZEN_RULE':
        raise ValueError('Confirmation not authorized by pilot rule')
    if decision.get('source_hash') != verify()['source_hash'] or decision.get('plan_sha256') != sha(c.HERE/'study_plan.json'):
        raise ValueError('Confirmation scientific source/plan changed')
    audit_path = c.ROOT/decision['audit_path']
    audit = read(audit_path)
    if sha(audit_path) != decision['audit_sha256'] or audit.get('status') != 'PASS':
        raise ValueError('Pilot independent audit missing or changed')
    expected_audit = dict(study_phase='pilot', execution_commit=decision['execution_commit'],
        source_hash=decision['source_hash'], plan_sha256=decision['plan_sha256'],
        scientific_fits_started=3, scientific_fits_completed=3, unknown_scientific_starts=0,
        TEST='NOT_RUN', test_evaluation_count=0, pairing_verified=True)
    if any(audit.get(k) != v for k, v in expected_audit.items()):
        raise ValueError('Pilot audit is not bound to completed paired phase')
    scores = {}
    results = {}
    for mode in c.MODES:
        entry = decision['pilot_results'][mode]
        p = c.ROOT/entry['path']
        r = read(p)
        if (sha(p) != entry['sha256'] or audit.get('pilot_result_sha256', {}).get(mode) != entry['sha256']
            or r.get('status') != 'PASS' or r.get('seed') != 2026 or r.get('phase_mode') != mode
            or r.get('scientific_fit_started') is not True
            or any(r.get(k) != expected_audit[k] for k in ('study_phase', 'execution_commit', 'source_hash', 'plan_sha256', 'TEST', 'test_evaluation_count'))):
            raise ValueError('Pilot result identity')
        scores[mode] = r['best_valid_metrics']['ndcg@10']
        results[mode] = r
    from .report import PAIRING
    for mode in c.MODES[1:]:
        if any(k not in results[mode] or k not in results['baseline_dual']
               or results[mode][k] != results['baseline_dual'][k] for k in PAIRING):
            raise ValueError('Actual pilot pairing mismatch')
    if not all(scores['absolute_phase'] >= scores[v] for v in ('baseline_dual', 'relative_phase')):
        raise ValueError('Frozen confirmation metric condition not met')
    if decision.get('pilot_scores') != scores or decision.get('pairing_verified') is not True:
        raise ValueError('Decision score/pairing evidence drift')
    return sha(path)


def validate_ownership(login, reservation, login_sha, expected, job, operational=None):
    if not re.fullmatch('[0-9]+', job):
        raise ValueError('Actual Slurm ID required')
    if any(any(r.get(k) != v for k, v in expected.items()) for r in (login, reservation)):
        raise ValueError('Execution binding mismatch')
    if (login.get('status') != 'PASS' or not login.get('tracked_clean') or not login.get('source_blobs_verified')
        or login.get('published_commit') != expected['execution_commit'] or reservation.get('login_sha256') != login_sha
        or reservation.get('max_scientific_fits') != len(c.tasks()) or reservation.get('tasks') != c.tasks()
        or reservation.get('requested_seconds') != c.allocation_seconds()
        or reservation.get('jobs_requested') != 1 or reservation.get('status') != 'RESERVED'
        or not re.fullmatch('[0-9a-f]{32}', reservation.get('token', ''))):
        raise ValueError('Invalid immutable reservation')
    if (not isinstance(login.get('coverage_sha256'), str)
        or not re.fullmatch('[0-9a-f]{64}', login['coverage_sha256'])
        or login['coverage_sha256'] != reservation.get('coverage_sha256')):
        raise ValueError('Coverage reservation mismatch')
    if operational is not None and (operational.get('token') != reservation['token']
        or operational.get('job_id') not in (None, job)
        or operational.get('job_id') is None and operational.get('status') != 'SUBMISSION_UNKNOWN_NO_RETRY'):
        raise ValueError('Wrong submitted job owner')


def identity():
    m = verify()
    login, reservation = read(c.LOGIN), read(c.RESERVATION)
    commit = os.environ.get('RUN_COMMIT', '')
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Exact execution commit required')
    if os.environ.get('EXPECTED_STUDY_HASH') != m['source_hash'] or os.environ.get('RESERVATION_TOKEN') != reservation['token']:
        raise ValueError('Environment owner mismatch')
    expected, job = bindings(commit, m), os.environ.get('SLURM_JOB_ID', '')
    validate_ownership(login, reservation, sha(c.LOGIN), expected, job, read(c.SUBMISSION) if c.SUBMISSION.exists() else None)
    if (c.LOGS/'pipeline.lock').exists():
        lock = read(c.LOGS/'pipeline.lock')
        if lock.get('job_id') != job or lock.get('reservation_token') != reservation['token']:
            raise ValueError('Pipeline already owned')
    coverage = coverage_verify(commit)
    decision = confirmation_authorization()
    if coverage != login['coverage_sha256'] or decision != reservation.get('confirmation_decision_sha256'):
        raise ValueError('Coverage or confirmation decision changed after reservation')
    return dict(**expected, job_id=job, reservation_token=reservation['token'], reservation_sha256=sha(c.RESERVATION),
                login_verification_sha256=sha(c.LOGIN), coverage_sha256=coverage,
                confirmation_decision_sha256=decision)


def login_verify():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=c.ROOT, text=True).strip()
    m, commit = verify(), git('rev-parse', 'HEAD')
    if git('branch', '--show-current') != c.BRANCH or git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Clean canonical study branch required')
    if git('rev-parse', 'origin/'+c.BRANCH) != commit:
        raise ValueError('Exact execution not published')
    subprocess.run(['git', 'merge-base', '--is-ancestor', c.PUBLICATION, commit], cwd=c.ROOT, check=True)
    for path, expected in m['files'].items():
        if hashlib.sha256(subprocess.check_output(['git', 'show', commit+':'+path], cwd=c.ROOT)).hexdigest() != expected:
            raise ValueError('Published blob ' + path)
    return dict(**bindings(commit, m), status='PASS', tracked_clean=True, published_commit=commit,
                source_blobs_verified=True, verified_at=now(), runtime=runtime(False),
                coverage_sha256=coverage_verify(commit), confirmation_decision_sha256=confirmation_authorization())


def validate_smoke(record):
    if (record.get('batch'), record.get('history_length'), record.get('kernel_length'), record.get('steps_per_mode')) != (2048, 50, 56, 3):
        raise ValueError('Smoke shapes/steps')
    if [r.get('phase_mode') for r in record.get('rows', [])] != list(c.MODES):
        raise ValueError('Smoke variants')
    for row in record['rows']:
        if row.get('status') != 'PASS' or len(row.get('steps', [])) != 3:
            raise ValueError('Smoke incomplete')
        names = set(c.parameter_keys(row['phase_mode']))
        for i, step in enumerate(row['steps']):
            norms = step.get('gradient_norms', {})
            if (step.get('step') != i or step.get('finite_loss') is not True or not math.isfinite(step['loss'])
                or set(norms) != names or any(not math.isfinite(x) or x < 0 for x in norms.values())):
                raise ValueError('Smoke gradient/loss evidence')
            layers = step.get('phase_layers', [])
            if (step.get('observer_removed') is not True or [x.get('layer') for x in layers] != [0, 1]
                or any(x.get('correction_shape') != [2048, 50, 32]
                       or x.get('finite_correction') is not True or x.get('first_padding_neutral') is not True
                       or x.get('native_angle_dtype') != 'torch.float32' for x in layers)):
                raise ValueError('Smoke observer/phase tensor evidence')
            if row['phase_mode'] != 'baseline_dual':
                if any(not isinstance(step.get(k), (int, float)) or not math.isfinite(step[k]) or step[k] < 0
                       for k in ('W_before_l2', 'W_after_l2', 'W_update_l2')):
                    raise ValueError('Smoke W update evidence')
        if row.get('roundtrip_passed') is not True or row.get('roundtrip') != 'weights_only=True':
            raise ValueError('Smoke checkpoint roundtrip')
        if any(type(row.get(k)) is not int or row[k] <= 0 for k in ('peak_allocated_bytes', 'peak_reserved_bytes')):
            raise ValueError('Smoke GPU memory evidence')


def require_stage(path, base):
    r = read(path)
    if r.get('status') != 'PASS' or any(r.get(k) != v for k, v in base.items()):
        raise ValueError('Missing/failed/foreign stage ' + str(path))
    if path == c.GATE and not accepted_cases(r.get('cases', []), c.plan()['required_cases']):
        raise ValueError('Targeted required leaves incomplete')
    if path == c.SMOKE:
        validate_smoke(r)
        if r['targeted_gate_sha256'] != sha(c.GATE):
            raise ValueError('Wrong smoke gate')
    return sha(path)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    result = freeze() if args.freeze else verify()
    print(json.dumps(dict(source_hash=result['source_hash'], files=len(result['files']))))
