"""Frozen dependencies and old evidence verification, without rerunning gates."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from .config import (HERE, ROOT, PILOT, PILOT_HASH, PILOT_COMMIT, B_COMMIT, POLICY_SHA, STUDY, BATCH, INIT,
                     SUBMISSION, LOGIN_VERIFICATION, FAILED, LOGS, PARENT_COMMIT, PARENT_HASH)
from experiments.mamba3_context_time.provenance import atomic_json, sha, now
from experiments.mamba3_three_time.evidence import create_record
from experiments.mamba3_three_time.provenance import CORE, PIN, upstream


def manifest():
    old = json.loads((PILOT / 'source_manifest.json').read_text())
    names = set(old['files'])
    names.add(str(PILOT.relative_to(ROOT) / 'source_manifest.json'))
    preserved = PILOT / 'evidence/pilot_001/preservation_manifest.json'
    names.add(str(preserved.relative_to(ROOT)))
    names.update(r['destination'] for r in json.loads(preserved.read_text())['files'])
    names.update(str(p.relative_to(ROOT)) for p in [*HERE.glob('*.py'), *(HERE / 'tests').glob('*.py')])
    names.update(str(HERE.relative_to(ROOT) / p) for p in ('study_plan.json', 'README.md', 'historical_sources.json'))
    names.update(str(p.relative_to(ROOT)) for p in (HERE / 'evidence/submission_001').iterdir() if p.is_file())
    names.add('slurm/mamba3_three_time_confirmation.sh')
    archive = HERE / 'evidence/job4355052'
    preserved_parent = archive / 'preservation_manifest.json'
    names.add(str(preserved_parent.relative_to(ROOT)))
    names.update(r['destination'] for r in json.loads(preserved_parent.read_text())['files'])
    names.update(str(HERE.relative_to(ROOT) / n) for n in ('resume_plan_003.json', 'resume_lineage_003.json'))
    names.add('slurm/mamba3_three_time_confirmation_resume.sh')
    files = {p: sha(ROOT / p) for p in sorted(names)}
    return dict(schema_version=1, files=files, source_hash=hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        pilot_source_hash=PILOT_HASH, core_hash=CORE)


def historical_from_git():
    """Login/freeze only: expected digests originate in historical Git objects."""
    groups = {
        B_COMMIT: ('mamba3_baseline', 'mamba3_timeaware', 'mamba3_time_mechanisms', 'mamba3_context_time'),
        PILOT_COMMIT: ('mamba3_three_time',),
    }
    frozen = json.loads((PILOT / 'source_manifest.json').read_text())['files']
    checked = {}
    for commit, prefixes in groups.items():
        for p in frozen:
            if not p.endswith('.py') or not any(p.startswith('experiments/' + prefix + '/') for prefix in prefixes):
                continue
            data = subprocess.check_output(['git', 'show', f'{commit}:{p}'], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != sha(ROOT / p):
                raise ValueError('Execution source differs: ' + p)
            checked[p] = dict(execution_commit=commit, sha256=hashlib.sha256(data).hexdigest())
    return dict(schema_version=1, origin='historical Git blobs; git show <execution_commit>:<path>', files=checked)


def execution_sources():
    """Runtime: compare actual bytes with the login-verified historical manifest."""
    saved = json.loads((HERE / 'historical_sources.json').read_text())
    frozen = json.loads((PILOT / 'source_manifest.json').read_text())['files']
    prefixes_b = ('mamba3_baseline', 'mamba3_timeaware', 'mamba3_time_mechanisms', 'mamba3_context_time')
    expected = {p: B_COMMIT if any(p.startswith('experiments/' + n + '/') for n in prefixes_b) else PILOT_COMMIT
                for p in frozen if p.endswith('.py') and any(p.startswith('experiments/' + n + '/')
                for n in (*prefixes_b, 'mamba3_three_time'))}
    if set(saved['files']) != set(expected):
        raise ValueError('Historical source coverage differs')
    for p, commit in expected.items():
        row = saved['files'][p]
        if row['execution_commit'] != commit or row['sha256'] != frozen[p] or sha(ROOT / p) != row['sha256']:
            raise ValueError('Historical execution source mismatch: ' + p)
    return expected


def imported_sources():
    frozen = json.loads((HERE / 'source_manifest.json').read_text())['files']
    checked = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith('experiments.') or not getattr(module, '__file__', None):
            continue
        path = __import__('pathlib').Path(module.__file__).resolve()
        relative = str(path.relative_to(ROOT))
        if relative.endswith('__init__.py') and path.stat().st_size == 0:
            continue
        if relative not in frozen or sha(path) != frozen[relative]:
            raise ValueError('Unmanifested/changed imported module: ' + relative)
        checked[relative] = frozen[relative]
    return checked


def old_admission():
    from experiments.mamba3_three_time.validation_pilot.gate import expected_ids
    from experiments.mamba3_three_time.records_003 import required_pass
    from experiments.mamba3_three_time.validation_pilot.config import ACCEPTED
    from experiments.mamba3_three_time.validation_pilot.historical import audit
    saved = json.loads((PILOT / 'evidence/pilot_001/preservation_manifest.json').read_text())
    for r in saved['files']:
        if sha(ROOT / r['destination']) != r['sha256']:
            raise ValueError('Pilot evidence altered: ' + r['destination'])
    path = PILOT / 'runs/admission_001.json'
    row = json.loads(path.read_text())
    if (row['status'], row['execution_commit'], row['source_hash'], row['policy_sha256'], row['backend']) != (
            ACCEPTED, PILOT_COMMIT, PILOT_HASH, POLICY_SHA, 'upstream'):
        raise ValueError('Archived admission identity mismatch')
    if row['required_registry'] != expected_ids() or [x['case_id'] for x in row['cases']] != expected_ids():
        raise ValueError('Archived admission coverage incomplete')
    if not all(required_pass(c) and c['passed'] for c in row['cases']):
        raise ValueError('Archived admission case failed')
    audit()
    return sha(path)


def verify():
    from experiments.mamba3_three_time.validation_pilot.provenance import verify as verify_pilot
    if verify_pilot()['source_hash'] != PILOT_HASH or sha(PILOT / 'numeric_acceptance_v1.json') != POLICY_SHA:
        raise ValueError('Frozen pilot/policy changed')
    actual = manifest()
    if actual != json.loads((HERE / 'source_manifest.json').read_text()):
        raise ValueError('Confirmation sources changed')
    parent = json.loads((FAILED / 'source_manifest.json').read_text())
    for name in ('one_batch.py', 'initialization.py', 'report.py', 'study_plan.json'):
        p = str(HERE.relative_to(ROOT) / name)
        if sha(ROOT / p) != parent['files'][p]:
            raise ValueError('Scientific confirmation procedure changed: ' + name)
    from .resume_provenance import verify_lineage
    verify_lineage()
    old_admission()
    execution_sources()
    return actual


def bindings(commit, manifest_value):
    return dict(execution_commit=commit, source_hash=manifest_value['source_hash'], core_hash=CORE,
                policy_sha256=POLICY_SHA, historical_manifest_sha256=sha(HERE / 'historical_sources.json'),
                plan_sha256=sha(HERE / 'study_plan.json'), source_manifest_sha256=sha(HERE / 'source_manifest.json'))


def login_verify(commit):
    """Only login/submit may claim to have checked HEAD or published Git refs."""
    def git(*args):
        return subprocess.check_output(['git', '--no-optional-locks', *args], cwd=ROOT, text=True).strip()
    if git('rev-parse', 'HEAD') != commit or git('branch', '--show-current') != 'exp/mamba3-three-time':
        raise ValueError('Exact branch/HEAD required on login')
    if git('diff', 'HEAD', '--'):
        raise ValueError('Tracked sources dirty on login')
    published = git('ls-remote', 'origin', 'refs/heads/exp/mamba3-three-time').split()
    if len(published) != 2 or published[0] != commit:
        raise ValueError('Commit not published on expected branch')
    if historical_from_git() != json.loads((HERE / 'historical_sources.json').read_text()):
        raise ValueError('Historical manifest differs from actual Git objects')
    m = verify()
    return dict(**bindings(commit, m), status='PASS', verified_at=now(), tracked_sources_clean=True,
                historical_git_objects_verified=True, published_branch_commit=commit,
                mechanism='login Git HEAD/branch/published objects + tracked content validation')


def verify_submission(commit, m, job_id, submission_path=None, login_path=None):
    submission_path = Path(submission_path) if submission_path is not None else SUBMISSION
    login_path = Path(login_path) if login_path is not None else LOGIN_VERIFICATION
    row = json.loads(submission_path.read_text())
    login = json.loads(login_path.read_text())
    expected = bindings(commit, m)
    if any(login.get(k) != v or row.get(k) != v for k, v in expected.items()):
        raise ValueError('Login/reservation source bindings differ')
    if (login.get('status') != 'PASS' or login.get('tracked_sources_clean') is not True or
            login.get('historical_git_objects_verified') is not True or login.get('published_branch_commit') != commit):
        raise ValueError('Missing exact login validation')
    if row.get('login_verification_sha256') != sha(login_path):
        raise ValueError('Immutable login evidence changed')
    if (row.get('status') not in ('RESERVED', 'SUBMITTED') or
            row.get('job_id') not in (None, job_id) or row.get('retry_of_job') != '4354908' or
            row.get('parent_execution_commit') != PARENT_COMMIT or
            row.get('old_source_hash') != PARENT_HASH or row.get('new_execution_commit') != commit or
            row.get('new_source_hash') != m['source_hash'] or row.get('scientific_runs_started_in_parent') != 0 or
            row.get('parent_failure_sha256') != sha(FAILED / 'failure.json') or
            row.get('outputs_and_locks_absent') is not True or
            row.get('retry_reason') != 'launcher_git_unavailable_before_python'):
        raise ValueError('Allocation does not own the explicit infrastructure retry')
    return dict(login_verification_sha256=sha(login_path),
                execution_commit_verification='login-verified identifier + runtime file SHA256; not runtime Git HEAD',
                submission_attempt=2)


def identity(*, submission_path=None, login_path=None):
    m = verify()
    commit, job_id = os.environ.get('RUN_COMMIT', ''), os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or not re.fullmatch(r'[0-9]+', job_id):
        raise ValueError('Login-verified commit and real allocation ID required')
    if os.environ.get('EXPECTED_STUDY_HASH') != m['source_hash'] or os.environ.get('EXPECTED_CORE_HASH') != CORE:
        raise ValueError('Submission/source mismatch')
    provenance = verify_submission(commit, m, job_id, submission_path, login_path)
    execution_sources()
    imported_sources()
    upstream()
    return dict(study_id=STUDY, execution_commit=commit, source_hash=m['source_hash'], core_hash=CORE,
                pinned_commit=PIN, backend='upstream', architecture='SISO', job_id=job_id,
                policy_version='siso_numeric_acceptance_v1', policy_sha256=POLICY_SHA,
                admission_sha256=old_admission(), TEST='NOT_RUN', test_evaluation_count=0, **provenance)


def retry_parent():
    """Only this preserved pre-Python failure can authorize a new reservation."""
    failure = json.loads((FAILED / 'failure.json').read_text())
    if (failure['job_id'], failure['execution_commit'], failure['source_hash'], failure['stage'],
            failure['state'], failure['exit_code'], failure['scientific_fits'], failure['diagnostic_batches']) != (
            '4354908', PARENT_COMMIT, PARENT_HASH, 'launcher_before_python', 'FAILED', '1:0', 0, 0):
        raise ValueError('Not the authorized pre-Python failure')
    for name, meta in failure['files'].items():
        if sha(FAILED / name) != meta['sha256']:
            raise ValueError('Preserved failure bytes changed')
        if name != 'source_manifest.json' and sha(LOGS / name) != meta['sha256']:
            raise ValueError('Original parent reservation/log differs from preserved evidence')
    others = [p for p in LOGS.rglob('submission*.json') if p != LOGS / 'submission_001.json']
    if others:
        raise FileExistsError('Another submission already exists: ' + repr(others))
    from .config import unused
    unused(include_submission=True)
    return dict(retry_of_job='4354908', retry_reason='launcher_git_unavailable_before_python',
                parent_execution_commit=PARENT_COMMIT, old_source_hash=PARENT_HASH,
                scientific_runs_started_in_parent=0, parent_failure_sha256=sha(FAILED / 'failure.json'),
                checked_absent=failure['checked_absent'], outputs_and_locks_absent=True)


def require_evidence(path, base):
    row = json.loads(path.read_text())
    if row.get('status') != 'PASS' or any(row.get(k) != v for k, v in base.items()):
        raise ValueError('Required same-allocation evidence: ' + str(path))
    return row


def gates(base, batch_path=None, init_path=None):
    from .config import MODES, SEEDS
    from .state import paired, ATOL, RTOL
    batch = require_evidence(BATCH if batch_path is None else batch_path, base)
    checks = batch.get('checks', {})
    if (batch.get('checks_passed') is not True or batch.get('optimizer_steps') != 2 or
            batch.get('diagnostic_batches') != 1 or not batch.get('batch_sha256') or
            batch.get('forward_backward_calls') != 2 or
            batch.get('structural_tolerance') != dict(atol=ATOL, rtol=RTOL) or
            set(checks) != {'outputs', 'gradients', 'updates', 'parameters', 'optimizer'} or
            not all(c.get('passed') is True and c.get('tensors') and
                    all(r.get('passed') is True for r in c['tensors'].values()) for c in checks.values())):
        raise ValueError('Incomplete controlled step')
    init = require_evidence(INIT if init_path is None else init_path, base)
    if [(r['seed'], r['mode']) for r in init['rows']] != [(s, m) for s in (2026, *SEEDS) for m in MODES]:
        raise ValueError('Incomplete paired initialization')
    if init.get('forward_calls') != 0 or init.get('scientific_fits') != 0:
        raise ValueError('Initialization must not perform forward/fit')
    for i in range(0, len(init['rows']), 2):
        paired(*init['rows'][i:i+2])
    return init
