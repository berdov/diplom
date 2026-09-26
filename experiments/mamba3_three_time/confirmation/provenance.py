"""Frozen dependencies and old evidence verification, without rerunning gates."""
import hashlib
import json
import os
import subprocess
import sys
from .config import HERE, ROOT, PILOT, PILOT_HASH, PILOT_COMMIT, B_COMMIT, POLICY_SHA, STUDY, BATCH, INIT
from experiments.mamba3_context_time.provenance import atomic_json, sha, now
from experiments.mamba3_three_time.evidence import create_record
from experiments.mamba3_three_time.provenance import CORE, PIN


def manifest():
    old = json.loads((PILOT / 'source_manifest.json').read_text())
    names = set(old['files'])
    names.add(str(PILOT.relative_to(ROOT) / 'source_manifest.json'))
    preserved = PILOT / 'evidence/pilot_001/preservation_manifest.json'
    names.add(str(preserved.relative_to(ROOT)))
    names.update(r['destination'] for r in json.loads(preserved.read_text())['files'])
    names.update(str(p.relative_to(ROOT)) for p in HERE.rglob('*.py') if 'slurm_logs' not in p.parts)
    names.update(str(HERE.relative_to(ROOT) / p) for p in ('study_plan.json', 'README.md'))
    names.add('slurm/mamba3_three_time_confirmation.sh')
    files = {p: sha(ROOT / p) for p in sorted(names)}
    return dict(schema_version=1, files=files, source_hash=hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        pilot_source_hash=PILOT_HASH, core_hash=CORE)


def execution_sources():
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
            checked[p] = commit
    return checked


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
    old_admission()
    execution_sources()
    return actual


def identity():
    m = verify()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if os.environ.get('RUN_COMMIT') != commit or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Exact allocation commit required')
    if os.environ.get('EXPECTED_STUDY_HASH') != m['source_hash'] or os.environ.get('EXPECTED_CORE_HASH') != CORE:
        raise ValueError('Submission/source mismatch')
    return dict(study_id=STUDY, execution_commit=commit, source_hash=m['source_hash'], core_hash=CORE,
                pinned_commit=PIN, backend='upstream', architecture='SISO', job_id=os.environ['SLURM_JOB_ID'],
                policy_version='siso_numeric_acceptance_v1', policy_sha256=POLICY_SHA,
                admission_sha256=old_admission(), TEST='NOT_RUN', test_evaluation_count=0)


def require_evidence(path, base):
    row = json.loads(path.read_text())
    if row.get('status') != 'PASS' or any(row.get(k) != v for k, v in base.items()):
        raise ValueError('Required same-allocation evidence: ' + str(path))
    return row


def gates(base):
    from .config import MODES, SEEDS
    from .state import paired, ATOL, RTOL
    batch = require_evidence(BATCH, base)
    checks = batch.get('checks', {})
    if (batch.get('checks_passed') is not True or batch.get('optimizer_steps') != 2 or
            batch.get('diagnostic_batches') != 1 or not batch.get('batch_sha256') or
            batch.get('forward_backward_calls') != 2 or
            batch.get('structural_tolerance') != dict(atol=ATOL, rtol=RTOL) or
            set(checks) != {'outputs', 'gradients', 'updates', 'parameters', 'optimizer'} or
            not all(c.get('passed') is True and c.get('tensors') and
                    all(r.get('passed') is True for r in c['tensors'].values()) for c in checks.values())):
        raise ValueError('Incomplete controlled step')
    init = require_evidence(INIT, base)
    if [(r['seed'], r['mode']) for r in init['rows']] != [(s, m) for s in (2026, *SEEDS) for m in MODES]:
        raise ValueError('Incomplete paired initialization')
    if init.get('forward_calls') != 0 or init.get('scientific_fits') != 0:
        raise ValueError('Initialization must not perform forward/fit')
    for i in range(0, len(init['rows']), 2):
        paired(*init['rows'][i:i+2])
    return init
