"""Login Git verification, compute file hashes, immutable reservation identity."""
import argparse
import ast
import importlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from . import config as c
from .records import read, sha, digest, create, now


def dependency_paths():
    # Conservative static closure includes function-local imports; never whole old manifests.
    found = set()
    pending = list(c.HERE.glob('*.py')) + list((c.HERE / 'tests').glob('*.py'))
    for root in ('mamba3_baseline', 'mamba3_timeaware', 'mamba3_time_mechanisms'):
        pending += list((c.ROOT / 'experiments' / root).glob('*.py'))
    def add_module(name):
        p = c.ROOT.joinpath(*name.split('.'))
        candidate = p.with_suffix('.py')
        if candidate.is_file():
            pending.append(candidate)
        elif (p / '__init__.py').is_file():
            pending.append(p / '__init__.py')
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for parent in path.relative_to(c.ROOT).parents:
            init = c.ROOT / parent / '__init__.py'
            if init.is_file() and init not in found:
                pending.append(init)
        package = list(path.relative_to(c.ROOT).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('experiments.'):
                        add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ''
                if node.level:
                    name = '.'.join(package[:len(package)-node.level+1] + ([name] if name else []))
                if name.startswith('experiments.'):
                    add_module(name)
                    for alias in node.names:
                        add_module(name + '.' + alias.name)
    for root in ('mamba3_baseline', 'mamba3_timeaware', 'mamba3_time_mechanisms'):
        found.update((c.ROOT / 'experiments' / root).glob('*.yaml'))
    found.update([c.PILOT, c.POLICY, c.HERE / 'study_plan.json', c.HERE / 'README.md',
                  c.ROOT / 'slurm/mamba3_mimo_time.sh',
                  c.ROOT / 'experiments/mamba3_three_time/upstream_manifest.json',
                  c.ROOT / 'experiments/mamba3_three_time/LICENSE.upstream',
                  c.ROOT / 'experiments/mamba3_timeaware/runs/train_time_stats_001.json',
                  c.ROOT / 'outputs/data/protocol_b_manifest.json'])
    preserved = read(c.PARENT_EVIDENCE/'preservation_manifest.json')
    found.update(c.ROOT/row['destination'] for row in preserved['files'])
    found.update([c.PARENT_EVIDENCE/'preservation_manifest.json', c.PARENT_EVIDENCE/'README.md',
                  c.ROOT/preserved['generated_settings']['path']])
    return sorted(found)


def freeze():
    import subprocess
    c.unused()
    files = {str(p.relative_to(c.ROOT)): sha(p) for p in dependency_paths()}
    historical = {}
    for path, value in files.items():
        if path.startswith('experiments/mamba3_mimo_time/') or path == 'slurm/mamba3_mimo_time.sh':
            continue
        blob = subprocess.check_output(['git', 'show', f'{c.HISTORICAL}:{path}'], cwd=c.ROOT)
        import hashlib
        if hashlib.sha256(blob).hexdigest() != value:
            raise ValueError('Historical dependency drift: ' + path)
        historical[path] = value
    value = dict(files=files, source_hash=digest(files), historical_execution=c.HISTORICAL,
                 historical_dependencies=historical, publication_commit=c.PUBLICATION, core_hash=c.CORE)
    (c.HERE / 'source_manifest.json').write_text(json.dumps(value, indent=2) + '\n')
    return value


def verify():
    m = read(c.HERE / 'source_manifest.json')
    if digest(m['files']) != m['source_hash']:
        raise ValueError('Manifest digest mismatch')
    bad = [p for p, expected in m['files'].items() if sha(c.ROOT / p) != expected]
    if bad:
        raise ValueError('Source mismatch: ' + repr(bad))
    from experiments.mamba3_time_mechanisms.provenance import fingerprint
    if fingerprint() != c.CORE:
        raise ValueError('Frozen temporal core drift')
    c.plan()
    from .retry import verify_parent
    verify_parent()
    return m


def imported_sources():
    m = verify()
    result = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if not name.startswith('experiments.') or not path:
            continue
        p = Path(path).resolve()
        if not p.is_relative_to(c.ROOT):
            raise ValueError('Imported another checkout: ' + str(p))
        rel = str(p.relative_to(c.ROOT))
        if rel not in m['files'] or sha(p) != m['files'][rel]:
            raise ValueError('Imported unmanifested source: ' + rel)
        result[name] = rel
    return result


def runtime(cuda=False):
    import torch
    from experiments.mamba3_three_time.provenance import upstream
    expected = read(c.PILOT)['runtime']
    names = dict(torch='torch', recbole='recbole', mamba_ssm='mamba-ssm', triton='triton',
                 numpy='numpy', tilelang='tilelang', apache_tvm_ffi='apache-tvm-ffi')
    actual = {k: importlib.metadata.version(n) for k, n in names.items()}
    if any(actual[k] != expected[k] for k in names):
        raise ValueError('Frozen environment version mismatch')
    pin = upstream()
    if pin['pinned_commit'] != c.PIN:
        raise ValueError('Mamba pin mismatch')
    if cuda and (not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name()):
        raise ValueError('One A100 required')
    return dict(**actual, pinned_commit=c.PIN, upstream_manifest_sha256=pin['manifest_sha256'],
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if cuda else None,
                imported_sources=imported_sources())


def bindings(commit, m):
    from .retry import parent_binding
    return dict(**parent_binding(), study_id=c.STUDY, execution_commit=commit, source_hash=m['source_hash'], core_hash=c.CORE,
                pinned_commit=c.PIN, policy_version=read(c.POLICY)['version'], policy_sha256=sha(c.POLICY),
                plan_sha256=sha(c.HERE / 'study_plan.json'), source_manifest_sha256=sha(c.HERE / 'source_manifest.json'),
                architecture='MIMO', backend='upstream', rank=4, chunk=8, TEST='NOT_RUN', test_evaluation_count=0)


def login_verify():
    import hashlib
    import subprocess
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=c.ROOT, text=True).strip()
    m = verify()
    commit = git('rev-parse', 'HEAD')
    if git('branch', '--show-current') != c.BRANCH or git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Expected clean tracked experiment branch')
    if git('rev-parse', 'origin/' + c.BRANCH) != commit:
        raise ValueError('Branch not published at exact commit')
    for path, expected in m['historical_dependencies'].items():
        blob = subprocess.check_output(['git', 'show', f'{c.HISTORICAL}:{path}'], cwd=c.ROOT)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('Historical source blob mismatch: ' + path)
    return dict(**bindings(commit, m), status='PASS', tracked_clean=True, historical_sources_verified=True,
                published_commit=commit, verified_at=now(), runtime=runtime(False))


def validate_ownership(login, reservation, login_sha, expected, job, operational=None):
    if not re.fullmatch(r'[0-9]+', job):
        raise ValueError('Actual Slurm job ID required')
    for record in (login, reservation):
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError('Serialized execution binding mismatch')
    if (login.get('status') != 'PASS' or not login.get('tracked_clean') or not login.get('historical_sources_verified')
            or login.get('published_commit') != expected['execution_commit']
            or reservation.get('login_sha256') != login_sha or reservation.get('max_scientific_fits') != 3
            or reservation.get('status') != 'RESERVED' or not re.fullmatch('[0-9a-f]{32}', reservation.get('token', ''))):
        raise ValueError('Invalid reservation/login limits')
    if operational is not None and (operational.get('job_id') != job or operational.get('token') != reservation['token']):
        raise ValueError('Submitted job owner mismatch')


def identity():
    m = verify()
    login, reservation = read(c.LOGIN), read(c.RESERVATION)
    expected = bindings(os.environ.get('RUN_COMMIT', ''), m)
    if not re.fullmatch('[0-9a-f]{40}', expected['execution_commit']):
        raise ValueError('Exact execution commit required')
    if os.environ.get('RESERVATION_TOKEN') != reservation['token'] or os.environ.get('EXPECTED_STUDY_HASH') != m['source_hash']:
        raise ValueError('Environment reservation binding mismatch')
    job = os.environ.get('SLURM_JOB_ID', '')
    validate_ownership(login, reservation, sha(c.LOGIN), expected, job,
                       read(c.SUBMISSION) if c.SUBMISSION.exists() else None)
    lock = c.LOGS / 'pipeline.lock'
    if lock.exists():
        owner = read(lock)
        if owner.get('job_id') != job or owner.get('reservation_token') != reservation['token']:
            raise ValueError('Another allocation owns the pipeline')
    return dict(**expected, job_id=job, reservation_token=reservation['token'], reservation_sha256=sha(c.RESERVATION),
                login_verification_sha256=sha(c.LOGIN))


def require_gate(base):
    from .records import accepted_cases
    gate = read(c.GATE)
    if (any(gate.get(k) != v for k, v in base.items()) or gate.get('status') != c.ACCEPTED
            or not accepted_cases(gate.get('cases', []), c.plan()['required_cases'])):
        raise ValueError('Incomplete/failed same-allocation admission')
    return sha(c.GATE)


def assert_upstream(model=None):
    from experiments.mamba3_three_time.backends import current
    if current() != 'upstream':
        raise ValueError('Diagnostic arithmetic forbidden')
    if model is not None and (not model.is_mimo or model.chunk_size != 8 or model.mimo_rank != 4):
        raise ValueError('Only MIMO rank4/chunk8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    print(json.dumps(freeze() if args.freeze else verify(), indent=2))
