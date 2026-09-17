"""Fingerprint implementation, without tying evidence to a documentation commit."""

import hashlib
import json
from pathlib import Path

PIN = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'
BASE = 'de0a137a6f46b5b214ccdc03a5ff1ede93454704'
HERE = Path(__file__).resolve().parent


def fingerprint():
    roots = (HERE, HERE.parent / 'mamba3_timeaware', HERE.parent / 'mamba3_baseline')
    paths = sorted(p for root in roots for p in root.iterdir() if p.suffix in ('.py', '.yaml'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(HERE.parent)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_equivalence(path):
    evidence = json.loads(Path(path).read_text())
    if (evidence.get('status'), evidence.get('pinned_commit'), evidence.get('source_fingerprint')) != ('PASS', PIN, fingerprint()):
        raise ValueError('Current-source GPU equivalence PASS required')
    for suite in ('A', 'B'):
        rows = [r for r in evidence['cases'] if r['suite'] == suite]
        if len(rows) != 4 or {(r['training'],r['length']) for r in rows} != {(t,l) for t in (False,True) for l in (50,64)}:
            raise ValueError('Incomplete GPU equivalence evidence')
        if any(not r['passed'] for r in rows):
            raise ValueError('Failed equivalence case')
    return evidence
