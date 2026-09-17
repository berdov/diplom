"""Standard-library preflight; no git dependency on compute nodes."""
import hashlib
import json
import os
from pathlib import Path

from experiments.mamba3_time_mechanisms.provenance import fingerprint, require_equivalence
from .config import (HERE, ROOT, CORE, EVIDENCE, EVIDENCE_SHA, STATS, STATS_SHA,
                     MANIFEST, MANIFEST_SHA, plan, scientific_settings, settings)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_manifest():
    sources = sorted([*HERE.glob('*.py'), *(HERE / 'tests').glob('*.py'), HERE / 'study_plan.json',
                      ROOT / 'slurm/mamba3_time_confirmation.sh'])
    files = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(schema_version=1, files=files, source_hash=digest)


def verify():
    if fingerprint() != CORE:
        raise ValueError('Frozen core changed')
    require_equivalence(EVIDENCE)
    for path, expected in ((EVIDENCE,EVIDENCE_SHA),(STATS,STATS_SHA),(MANIFEST,MANIFEST_SHA)):
        if sha(path) != expected:
            raise ValueError(f'Immutable artifact changed: {path}')
    expected = json.loads((HERE / 'source_manifest.json').read_text())
    actual = source_manifest()
    if actual != expected:
        raise ValueError('Study source manifest mismatch')
    for reuse in plan()['reuse']:
        path = ROOT / reuse['source_json']
        data = json.loads(path.read_text())
        if sha(path) != reuse['sha256'] or data['git_commit'] != reuse['execution_commit']:
            raise ValueError('Historical result changed')
        if (data['status'],data['test_evaluation_count'],data['config']['seed'],data['evaluation_mode']) != ('PASS',0,2026,'full-ranking'):
            raise ValueError('Invalid reuse scope')
        if scientific_settings(data['config']) != scientific_settings(settings(plan()['tasks'][0])):
            raise ValueError('Historical training configuration drift')
        stats = json.loads(STATS.read_text())
        if data['protocol']['recbole_inter_sha256'] != stats['protocol_b_inter_sha256'] or data['train_time_stats_sha256'] != STATS_SHA:
            raise ValueError('Historical dataset drift')
    return actual


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=str)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def acquire(task_paths, identity):
    if task_paths['result'].exists():
        raise FileExistsError('Result exists; no overwrite or automatic retry')
    task_paths['runtime'].mkdir(parents=True, exist_ok=True)
    with task_paths['lock'].open('x') as handle:
        json.dump(identity, handle)
        handle.flush()
        os.fsync(handle.fileno())
    if task_paths['result'].exists():
        raise FileExistsError('Result appeared while acquiring lock')
