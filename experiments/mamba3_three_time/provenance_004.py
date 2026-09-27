"""New manifest without rewriting any historical snapshot/profile."""

import hashlib
import json
import os
from . import attempt_004 as attempt
from .provenance import CORE, PIN, sha, upstream
from experiments.mamba3_time_mechanisms.provenance import fingerprint


def manifest():
    # Preserve the exact previous dependency set and add only attempt004 sources.
    old=json.loads((attempt.HERE/'source_manifest_003.json').read_text())
    names=set(old['files'])
    names.update(str(p.relative_to(attempt.ROOT)) for p in attempt.HERE.glob('*004.py'))
    names.update(['experiments/mamba3_three_time/test_plan_004.json',
                  'experiments/mamba3_three_time/tests/test_attempt004_cpu.py',
                  'experiments/mamba3_three_time/evidence/attempt004_contract.md',
                  'slurm/mamba3_three_time_diagnostics_004.sh'])
    hashes={p:sha(attempt.ROOT/p) for p in sorted(names)}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(schema_version=1,attempt_id='004',source_hash=digest,core_hash=CORE,files=hashes)


def verify():
    if fingerprint()!=CORE:raise ValueError('Frozen core changed')
    frozen=json.loads((attempt.HERE/'frozen_snapshot.json').read_text())
    bad=[p for p,h in frozen['files'].items() if sha(attempt.ROOT/p)!=h]
    if bad:raise ValueError('Frozen sources/results changed: '+repr(bad))
    for previous in ('001','002','003'):
        m=json.loads((attempt.HERE/f'evidence/attempt_{previous}/preservation_manifest.json').read_text())
        for r in m['files']:
            p=attempt.ROOT/r.get('destination',r.get('destination_path'))
            if sha(p)!=r['sha256']:raise ValueError('Archived evidence changed: '+str(p))
    result=manifest()
    if result!=json.loads(attempt.SOURCE_MANIFEST.read_text()):raise ValueError('Attempt004 source mismatch')
    return result


def require_submission():
    m=verify()
    commit=os.environ.get('RUN_COMMIT','')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Exact commit required')
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Allocation required')
    if os.environ.get('EXPECTED_STUDY_HASH')!=m['source_hash'] or os.environ.get('EXPECTED_CORE_HASH')!=CORE:
        raise ValueError('Submission hash mismatch')
    return m
