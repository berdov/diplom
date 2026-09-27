"""Exact parent lineage plus per-attempt ownership; runtime never invokes Git."""
import ast
import json
import math
import os
import re
import subprocess
from pathlib import Path
from . import config as c, provenance as p, resume_config as r


def read(path):
    return json.loads(path.read_text())


def verify_lineage():
    old = read(r.ARCHIVE/'source_manifest.json')
    lineage = read(r.LINEAGE)
    if old['source_hash'] != r.PARENT_HASH or lineage['parent_commit'] != r.PARENT_COMMIT:
        raise ValueError('Wrong frozen parent source lineage')
    actual = {n: p.sha(c.ROOT/n) for n in old['files']}
    changed = {n for n, h in actual.items() if h != old['files'][n]}
    if changed != set(lineage['changes']):
        raise ValueError('Unallowlisted source changes: ' + str(changed ^ set(lineage['changes'])))
    for n in changed:
        v = lineage['changes'][n]
        if v['before_sha256'] != old['files'][n] or v['after_sha256'] != actual[n] or not v['reason']:
            raise ValueError('Infrastructure allowlist digest mismatch: ' + n)
    # The only prepare() change is a pure metadata copy after optimizer creation.
    def prepare(path):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    before, after = prepare(r.ARCHIVE/'runner.py'), prepare(c.HERE/'runner.py')
    assignments = [n for n in ast.walk(after) if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and
                       t.slice.value == 'optimizer_settings' for t in n.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Call):
        raise ValueError('Missing canonical metadata-only prepare change')
    call = assignments[0].value
    if not isinstance(call.func, ast.Name) or call.func.id != 'canonical_optimizer_settings' or len(call.args) != 1:
        raise ValueError('Unexpected optimizer mutation')
    assignments[0].value = call.args[0]
    if ast.dump(before) != ast.dump(after):
        raise ValueError('Scientific setup/initialization procedure changed')
    return lineage


def preserved(originals=False):
    manifest = read(r.ARCHIVE/'preservation_manifest.json')
    for row in manifest['files']:
        if p.sha(c.ROOT/row['destination']) != row['sha256']:
            raise ValueError('Preserved evidence changed: ' + row['destination'])
        if originals and ('/runs/' in row['source'] or '/slurm_logs/' in row['source']):
            if p.sha(c.ROOT/row['source']) != row['sha256']:
                raise ValueError('Original parent evidence changed: ' + row['source'])
    return manifest


def parent_base():
    return dict(study_id=c.STUDY, execution_commit=r.PARENT_COMMIT, source_hash=r.PARENT_HASH,
                job_id=r.PARENT_JOB, core_hash=p.CORE, pinned_commit=p.PIN, backend='upstream', architecture='SISO',
                policy_version='siso_numeric_acceptance_v1', policy_sha256=c.POLICY_SHA,
                TEST='NOT_RUN', test_evaluation_count=0, submission_attempt=2)


def parent_record(name):
    row = read(r.ARCHIVE/'runs'/name)
    if any(row.get(k) != v for k, v in parent_base().items()):
        raise ValueError('Unauthorized parent identity: ' + name)
    return row


def inherited_gates(base=None):
    preserved()
    plan = r.plan()
    for entry in plan['inherited_gates']:
        for name in ('path', 'archive_path'):
            if p.sha(c.ROOT/entry[name]) != entry['sha256']:
                raise ValueError('Inherited gate SHA mismatch: ' + name)
    batch = parent_record('one_batch_001.json')
    init = p.gates(parent_base(), r.ARCHIVE/'runs/one_batch_001.json', r.ARCHIVE/'runs/initialization_001.json')
    if batch.get('failed_stages') != [] or not batch['state_mapping']['all_keys_shapes_dtypes_values_equal']:
        raise ValueError('Parent batch mapping/checks incomplete')
    expected = dict(outputs=3, gradients=45, updates=47, parameters=47, optimizer=135)
    for group, count in expected.items():
        checks = batch['checks'][group]['tensors']
        if len(checks) != count:
            raise ValueError('Incomplete inherited numerical checks: ' + group)
        for v in checks.values():
            if v.get('atol') != 1e-6 or v.get('rtol') != 1e-5 or not all(
                    type(v.get(k)) in (float, int) and math.isfinite(v[k]) for k in ('max_abs', 'mean_abs', 'l2_error')):
                raise ValueError('Invalid inherited numerical evidence')
    return init


def check_checkpoint(row, entry):
    from .report import validate
    validate(row, entry['mode'], entry['seed'])
    if row['status'] != 'PASS':
        raise ValueError('Completed checkpoint required')
    if row['checkpoint_path'] != str(c.ROOT/entry['checkpoint_path']) or row['checkpoint_metadata_path'] != str(c.ROOT/entry['metadata_path']):
        raise ValueError('Unexpected checkpoint path')
    meta_path = c.ROOT/entry['metadata_path']
    if entry.get('metadata_sha256') and p.sha(meta_path) != entry['metadata_sha256']:
        raise ValueError('Parent checkpoint metadata SHA mismatch')
    meta = read(meta_path)
    pairs = {'epoch': 'best_epoch', 'metrics': 'best_valid_metrics', 'checkpoint_sha256': 'checkpoint_sha256',
             'mode': 'mode', 'seed': 'seed', 'run_id': 'run_id', 'execution_commit': 'execution_commit',
             'source_hash': 'source_hash', 'core_hash': 'core_hash', 'config_sha256': 'config_sha256'}
    if any(meta[k] != row[v] for k, v in pairs.items()):
        raise ValueError('Checkpoint metadata/result mismatch')
    if entry.get('checkpoint_sha256', row['checkpoint_sha256']) != row['checkpoint_sha256'] or p.sha(c.ROOT/entry['checkpoint_path']) != row['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA mismatch')


def resolve(mode, seed, base, check_bytes=True):
    entry = next(v for v in r.plan()['source_index'] if (v['mode'], v['seed']) == (mode, seed))
    path = c.ROOT/entry['path']
    row = read(path)
    if (row.get('mode'), row.get('seed')) != (mode, seed) or row.get('TEST') != 'NOT_RUN' or row.get('test_evaluation_count') != 0:
        raise ValueError('Run identity/TEST mismatch')
    if entry['kind'] != 'NEW':
        if p.sha(path) != entry['sha256'] or any(row.get(k) != v for k, v in entry['identity'].items()):
            raise ValueError('Unauthorized inherited result')
        if row['status'] != 'PASS':
            raise ValueError('Only completed inherited result allowed')
    else:
        if any(row.get(k) != v for k, v in base.items()) or row['run_id'] != c.task(mode, seed)['run_id']:
            raise ValueError('Current reservation/run ownership mismatch')
    from .report import validate
    validate(row, mode, seed)
    if check_bytes and seed != 2026 and row['status'] == 'PASS':
        check_checkpoint(row, entry)
    return row


def parent_ready():
    preserved(originals=True)
    inherited_gates()
    dual = resolve('dual', 2027, {})
    if dual.get('scientific_fit_started') is not True or dual['stage'] != 'COMPLETED':
        raise ValueError('Parent dual is not completed')
    for mode, seed in r.ORDER:
        row = parent_record(c.task(mode, seed)['run_id']+'.json')
        if row.get('scientific_fit_started') is not False or row.get('actual_epochs', 0) != 0 or row.get('history', []) != []:
            raise ValueError('Parent fit already started; cannot repeat')
        if (mode, seed) == ('triple', 2027):
            if row['status'] != 'FAIL' or row['stage'] != 'SETUP' or row['error'] != "ValueError('Pair mismatch: optimizer_settings')" or 'first_train_batch_sha256' in row:
                raise ValueError('Not the authorized pre-fit metadata failure')
        elif row['status'] != 'NOT_RUN':
            raise ValueError('Unexpected parent run state')
    pipeline = read(r.ARCHIVE/'slurm_logs/pipeline_status.json')
    if pipeline['status'] != 'FAIL' or pipeline['scientific_fits'] != 1:
        raise ValueError('Unexpected parent pipeline')
    return dict(parent_job=r.PARENT_JOB, previous_completed_confirmation_fits=1,
                inherited_gates=[dict(path=v['path'], sha256=v['sha256']) for v in r.plan()['inherited_gates']],
                new_diagnostic_batches=0, new_diagnostic_optimizer_steps=0)


def bindings(commit, m):
    return dict(**p.bindings(commit, m), resume_plan_sha256=p.sha(r.PLAN), lineage_sha256=p.sha(r.LINEAGE),
                parent_job=r.PARENT_JOB, parent_execution_commit=r.PARENT_COMMIT, parent_source_hash=r.PARENT_HASH)


def login_verify(commit):
    value = p.login_verify(commit)
    old = read(r.ARCHIVE/'source_manifest.json')
    name = str(c.HERE.relative_to(c.ROOT)/'source_manifest.json')
    blob = subprocess.check_output(['git', 'show', f'{r.PARENT_COMMIT}:{name}'], cwd=c.ROOT)
    if blob != (r.ARCHIVE/'source_manifest.json').read_bytes():
        raise ValueError('Parent manifest is not the historical Git object')
    import hashlib
    for name, digest in old['files'].items():
        if hashlib.sha256(subprocess.check_output(['git', 'show', f'{r.PARENT_COMMIT}:{name}'], cwd=c.ROOT)).hexdigest() != digest:
            raise ValueError('Parent source manifest/Git mismatch: ' + name)
    parent_ready()
    return dict(value, **{k:v for k,v in bindings(commit, p.verify()).items() if k not in value},
                parent_git_objects_verified=True)


def identity(*, submission_path=None, login_path=None):
    m = p.verify()
    commit, job = os.environ.get('RUN_COMMIT', ''), os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch('[0-9a-f]{40}', commit) or not re.fullmatch('[0-9]+', job):
        raise ValueError('Login-verified commit and actual job required')
    expected = bindings(commit, m)
    if os.environ.get('EXPECTED_STUDY_HASH') != m['source_hash'] or os.environ.get('EXPECTED_CORE_HASH') != p.CORE:
        raise ValueError('Resume source environment mismatch')
    login_path = Path(login_path) if login_path else r.LOGIN
    login = read(login_path)
    submission = read(Path(submission_path) if submission_path else r.SUBMISSION)
    if any(login.get(k) != v or submission.get(k) != v for k, v in expected.items()):
        raise ValueError('Resume login/reservation bindings differ')
    if (login.get('status') != 'PASS' or login.get('parent_git_objects_verified') is not True or
            login.get('tracked_sources_clean') is not True or login.get('published_branch_commit') != commit or
            submission.get('login_verification_sha256') != p.sha(login_path) or
            submission.get('status') not in ('RESERVED','SUBMITTED') or submission.get('job_id') not in (None, job) or
            submission.get('max_scientific_fits') != 7 or submission.get('new_diagnostic_optimizer_steps') != 0):
        raise ValueError('Invalid resume ownership/limits')
    p.imported_sources()
    p.upstream()
    parent_ready()
    return dict(**expected, study_id=c.STUDY, job_id=job, pinned_commit=p.PIN, backend='upstream', architecture='SISO',
                policy_version='siso_numeric_acceptance_v1', admission_sha256=p.old_admission(),
                TEST='NOT_RUN', test_evaluation_count=0, execution_attempt=3, submission_attempt=3,
                login_verification_sha256=p.sha(login_path),
                execution_commit_verification='login-verified identifier + runtime file SHA256; not runtime Git HEAD')
