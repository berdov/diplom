"""Pilot content inheritance; login Git proof, compute content/ownership checks."""
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from . import config as c
from .records import read, sha, digest, now, create


def check_files(root, manifest):
    if digest(manifest['files']) != manifest['source_hash']:
        raise ValueError('Manifest digest mismatch')
    for name, expected in manifest['files'].items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or sha(p) != expected:
            raise ValueError('Source mismatch: ' + name)


def inherited():
    from experiments.mamba3_mimo_time.records import accepted_cases
    from experiments.mamba3_mimo_time import config as pilot
    m = read(c.PILOT_ROOT / 'source_manifest.json')
    if m['source_hash'] != c.PILOT_SOURCE:
        raise ValueError('Pilot manifest lineage mismatch')
    check_files(c.ROOT, m)
    if sha(c.POLICY) != c.POLICY_SHA or sha(c.ADMISSION) != c.ADMISSION_SHA or sha(c.SMOKE) != c.SMOKE_SHA:
        raise ValueError('Inherited policy/admission/smoke changed')
    a, s = read(c.ADMISSION), read(c.SMOKE)
    specs = pilot.plan()['required_cases']
    for row in (a, s):
        expected = dict(execution_commit=c.PILOT_COMMIT, source_hash=c.PILOT_SOURCE, job_id='4358147',
                        core_hash=c.CORE, pinned_commit=c.PIN, policy_sha256=c.POLICY_SHA,
                        architecture='MIMO', backend='upstream', rank=4, chunk=8, TEST='NOT_RUN', test_evaluation_count=0)
        if any(row.get(k) != v for k,v in expected.items()):
            raise ValueError('Inherited identity mismatch')
    if (a['status'] != pilot.ACCEPTED or len(specs) != 45 or sum(len(r['required_keys']) for r in specs) != 2342
            or a['required_cases'] != specs or not accepted_cases(a['cases'], specs)):
        raise ValueError('Incomplete inherited admission')
    if (s['status'] != 'PASS' or s['admission_sha256'] != c.ADMISSION_SHA
            or (s['batch'],s['history_length'],s['kernel_length'],s['steps_per_mode']) != (2048,50,56,2)
            or [r['mode'] for r in s['rows']] != list(c.MODES)
            or any(r['status'] != 'PASS' or len(r['steps']) != 2 or not all(t['finite_loss'] for t in r['steps']) for r in s['rows'])):
        raise ValueError('Incomplete inherited smoke')
    for mode in c.MODES:
        r = read(c.pilot_path(mode))
        if (r['status'] != 'PASS' or r['execution_commit'] != c.PILOT_COMMIT or r['source_hash'] != c.PILOT_SOURCE
                or r['job_id'] != '4358147' or r['mode'] != mode or r['seed'] != 2026
                or r['admission_sha256'] != c.ADMISSION_SHA or r['smoke_sha256'] != c.SMOKE_SHA):
            raise ValueError('Pilot scientific lineage mismatch')
    return dict(status='INHERITED_PASS', original_job_id='4358147', original_execution_commit=c.PILOT_COMMIT,
                original_source_hash=c.PILOT_SOURCE, admission_sha256=c.ADMISSION_SHA, smoke_sha256=c.SMOKE_SHA,
                cases=45, required_checks=2342, policy_sha256=c.POLICY_SHA,
                mathematical_dependencies_unchanged=True, repeated_gpu_checks=0,
                limitations='mimo_numeric_acceptance_v1; historical exact-zero failures remain historical failures')


def freeze():
    c.unused()
    inherited()
    old = read(c.PILOT_ROOT / 'source_manifest.json')
    files = dict(old['files'])
    sources = list(c.HERE.glob('*.py')) + list((c.HERE/'tests').glob('*.py'))
    sources += [c.HERE/'README.md', c.HERE/'study_plan.json', c.LAUNCHER,
                c.PILOT_ROOT/'source_manifest.json', c.ADMISSION, c.SMOKE,
                c.PILOT_ROOT/'slurm_logs/attempt_003/preflight/evidence.json']
    sources += [c.pilot_path(m) for m in c.MODES]
    for p in sources:
        files[str(p.relative_to(c.ROOT))] = sha(p)
    value = dict(source_hash=digest(files), files=files, pilot_execution=c.PILOT_COMMIT, pilot_source_hash=c.PILOT_SOURCE,
                 publication_commit=c.PUBLICATION, core_hash=c.CORE)
    create(c.HERE/'source_manifest.json', value)
    return value


def verify():
    m = read(c.HERE/'source_manifest.json')
    check_files(c.ROOT, m)
    c.plan()
    from experiments.mamba3_time_mechanisms.provenance import fingerprint
    if fingerprint() != c.CORE:
        raise ValueError('Core drift')
    inherited()
    return m


def imported_sources():
    m = verify()
    rows = {}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if not name.startswith('experiments.') or not file:
            continue
        p = Path(file).resolve()
        if not p.is_relative_to(c.ROOT):
            raise ValueError('Imported another checkout: ' + str(p))
        rel = str(p.relative_to(c.ROOT))
        if rel not in m['files'] or sha(p) != m['files'][rel]:
            raise ValueError('Unmanifested import: ' + rel)
        rows[name] = rel
    return rows


def runtime(cuda=False):
    import torch
    from experiments.mamba3_three_time.provenance import upstream
    expected = read(c.pilot_path('base'))['runtime']
    names = dict(torch='torch', recbole='recbole', mamba_ssm='mamba-ssm', triton='triton',
                 numpy='numpy', tilelang='tilelang', apache_tvm_ffi='apache-tvm-ffi')
    actual = {k:importlib.metadata.version(v) for k,v in names.items()}
    pin = upstream()
    if any(actual[k] != expected[k] for k in names) or torch.version.cuda != expected['cuda']:
        raise ValueError('Frozen environment drift')
    if pin['pinned_commit'] != c.PIN or pin['manifest_sha256'] != expected['upstream_manifest_sha256']:
        raise ValueError('Pinned upstream drift')
    if cuda and (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name()):
        raise ValueError('Exactly one visible A100 required')
    return dict(**actual, pinned_commit=c.PIN, upstream_manifest_sha256=pin['manifest_sha256'],
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if cuda else None, imported_sources=imported_sources())


def bindings(commit, manifest):
    return dict(study_id=c.STUDY, execution_commit=commit, source_hash=manifest['source_hash'],
                source_manifest_sha256=sha(c.HERE/'source_manifest.json'), plan_sha256=sha(c.HERE/'study_plan.json'),
                core_hash=c.CORE, pinned_commit=c.PIN, policy_version='mimo_numeric_acceptance_v1', policy_sha256=c.POLICY_SHA,
                inherited_admission_sha256=c.ADMISSION_SHA, inherited_smoke_sha256=c.SMOKE_SHA,
                pilot_execution_commit=c.PILOT_COMMIT, pilot_job_id='4358147',
                architecture='MIMO', backend='upstream', rank=4, chunk=8, TEST='NOT_RUN', test_evaluation_count=0)


def login_verify():
    import subprocess
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=c.ROOT,text=True).strip()
    m = verify()
    commit = git('rev-parse','HEAD')
    if git('branch','--show-current') != c.BRANCH or git('status','--porcelain','--untracked-files=no'):
        raise ValueError('Expected clean tracked confirmation branch')
    if git('rev-parse','origin/'+c.BRANCH) != commit:
        raise ValueError('Exact execution commit not published')
    subprocess.run(['git','merge-base','--is-ancestor',c.PUBLICATION,commit],cwd=c.ROOT,check=True)
    for name,expected in m['files'].items():
        blob = subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=c.ROOT)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('Published source blob mismatch: ' + name)
    return dict(**bindings(commit,m),status='PASS',tracked_clean=True,published_commit=commit,
                source_blobs_verified=True,verified_at=now(),runtime=runtime(False))


def validate_ownership(login, reservation, login_sha, expected, job, operational=None):
    if not re.fullmatch('[0-9]+',job):
        raise ValueError('Actual Slurm ID required')
    for record in (login,reservation):
        if any(record.get(k) != v for k,v in expected.items()):
            raise ValueError('Serialized execution binding mismatch')
    if (login.get('status') != 'PASS' or not login.get('tracked_clean') or not login.get('source_blobs_verified')
            or login.get('published_commit') != expected['execution_commit']
            or reservation.get('login_sha256') != login_sha or reservation.get('max_scientific_fits') != 12
            or reservation.get('tasks') != c.tasks() or reservation.get('jobs_requested') != 1
            or reservation.get('status') != 'RESERVED' or not re.fullmatch('[0-9a-f]{32}',reservation.get('token',''))):
        raise ValueError('Invalid reservation/login limits')
    if operational is not None and (operational.get('job_id') != job or operational.get('token') != reservation['token']):
        raise ValueError('Another submitted job owner')


def identity():
    m = verify()
    login, reservation = read(c.LOGIN), read(c.RESERVATION)
    commit = os.environ.get('RUN_COMMIT','')
    if not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Exact execution commit required')
    if os.environ.get('EXPECTED_STUDY_HASH') != m['source_hash'] or os.environ.get('RESERVATION_TOKEN') != reservation['token']:
        raise ValueError('Environment ownership mismatch')
    expected = bindings(commit,m)
    job = os.environ.get('SLURM_JOB_ID','')
    validate_ownership(login,reservation,sha(c.LOGIN),expected,job,read(c.SUBMISSION) if c.SUBMISSION.exists() else None)
    lock = c.LOGS/'pipeline.lock'
    if lock.exists():
        owner = read(lock)
        if owner.get('job_id') != job or owner.get('reservation_token') != reservation['token']:
            raise ValueError('Another allocation owns pipeline')
    return dict(**expected,job_id=job,reservation_token=reservation['token'],reservation_sha256=sha(c.RESERVATION),login_verification_sha256=sha(c.LOGIN))


def require_inherited(base):
    row = read(c.INHERITED)
    if any(row.get(k) != v for k,v in base.items()) or row.get('status') != 'PASS' or row.get('inherited') != inherited():
        raise ValueError('Incomplete inherited-admission record for this allocation')
    return sha(c.INHERITED)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    args = parser.parse_args()
    value = freeze() if args.freeze else verify()
    print(json.dumps(dict(source_hash=value['source_hash'],files=len(value['files']))))
