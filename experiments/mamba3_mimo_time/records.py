"""Durable one-shot records and fail-closed required leaves; standard library only."""
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False, default=str)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        # Publish a complete record atomically, without replacing another owner.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def update(path, value):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Owner must create first: ' + str(path))
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    create(temp, value)
    os.replace(temp, path)


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return True


def leaf_pass(value):
    if not isinstance(value, dict) or value.get('passed') is not True or not finite_tree(value):
        return False
    def descendants(v):
        if isinstance(v, dict):
            if 'passed' in v and v['passed'] is not True:
                return False
            if v.get('status') in ('FAIL', 'INCONCLUSIVE', 'SKIP', 'NOT_RUN', 'ZERO_SIGNAL'):
                return False
            return all(descendants(x) for x in v.values())
        if isinstance(v, list):
            return all(descendants(x) for x in v)
        return True
    return descendants(value)


def accepted_cases(cases, specs):
    expected = [r['id'] for r in specs]
    actual = [r.get('case_id') for r in cases]
    if len(actual) != len(set(actual)) or actual != expected:
        return False
    return all(r.get('status') == 'PASS' and r.get('required_keys') == s['required_keys'] == sorted(r.get('checks', {}))
               and bool(r.get('checks')) and all(leaf_pass(v) for v in r['checks'].values()) for r, s in zip(cases, specs))


def case(checks, **evidence):
    if not checks:
        raise ValueError('Empty required checks')
    return dict(checks=checks, required_keys=sorted(checks), **evidence)


class Registry:
    def __init__(self, path, result, specs):
        self.path, self.result, self.specs = path, result, specs
        ids = [s['id'] for s in specs]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate registry ID')
        result.update(required_cases=specs, cases=[])

    def run(self, spec, function):
        import traceback
        i = len(self.result['cases'])
        if i >= len(self.specs) or spec != self.specs[i]:
            raise ValueError('Unexpected/duplicate case')
        row = dict(case_id=spec['id'], status='RUNNING', checks={}, required_keys=[])
        self.result['cases'].append(row)
        update(self.path, self.result)
        def save(value):
            row.update(value)
            update(self.path, self.result)
        try:
            save(function(save))
            good = bool(row['checks']) and sorted(row['checks']) == row['required_keys'] == spec['required_keys'] and all(leaf_pass(v) for v in row['checks'].values())
            row['missing_keys'] = sorted(set(spec['required_keys']) - set(row['checks']))
            row['unexpected_keys'] = sorted(set(row['checks']) - set(spec['required_keys']))
            row.update(status='PASS' if good else 'FAIL', failed_keys=[k for k, v in row['checks'].items() if not leaf_pass(v)])
        except BaseException:
            row.update(status='FAIL', traceback=traceback.format_exc())
            row['missing_keys'] = sorted(set(spec['required_keys']) - set(row['checks']))
            row['unexpected_keys'] = sorted(set(row['checks']) - set(spec['required_keys']))
        update(self.path, self.result)
        if row['status'] != 'PASS':
            raise RuntimeError('Required admission case failed: ' + row['case_id'])
