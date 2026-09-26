"""Content-addressed dependencies and one-shot pilot ownership."""
import hashlib
import json
import os
from .config import HERE, ROOT, PARENT, GATE, ACCEPTED, EXECUTION004, SOURCE004, settings, plan
from .policy import policy
from experiments.mamba3_three_time.provenance_004 import verify as verify004
from experiments.mamba3_three_time.provenance import upstream, CORE, PIN, sha
from experiments.mamba3_three_time.evidence import create_record
from experiments.mamba3_three_time.records_003 import required_pass
from experiments.mamba3_context_time.provenance import atomic_json, rng_hash, tensor_hash, save_state_dict, now


def manifest():
    names = set(json.loads((PARENT / 'source_manifest_004.json').read_text())['files'])
    names.update(json.loads((PARENT / 'frozen_snapshot.json').read_text())['files'])
    names.update(str(p.relative_to(ROOT)) for p in HERE.rglob('*.py') if 'slurm_logs' not in p.parts)
    names.update(str(HERE.relative_to(ROOT) / p) for p in ('study_plan.json', 'numeric_acceptance_v1.json', 'README.md'))
    names.add('slurm/mamba3_three_time_validation_pilot.sh')
    names.update('experiments/mamba3_three_time/' + p for p in (
        'source_manifest_003.json', 'source_manifest_004.json', 'frozen_snapshot.json'))
    for previous in ('001', '002', '003'):
        preserved = PARENT / f'evidence/attempt_{previous}/preservation_manifest.json'
        names.add(str(preserved.relative_to(ROOT)))
        names.update(r.get('destination', r.get('destination_path')) for r in json.loads(preserved.read_text())['files'])
    saved = json.loads((PARENT / 'evidence/attempt_004/preservation_manifest.json').read_text())
    names.add('experiments/mamba3_three_time/evidence/attempt_004/preservation_manifest.json')
    names.update(r['destination'] for r in saved['files'])
    files = {p: sha(ROOT / p) for p in sorted(names)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(schema_version=1, files=files, source_hash=digest, core_hash=CORE,
                model_execution_commit=EXECUTION004, model_source_hash=SOURCE004)


def verify():
    if verify004()['source_hash'] != SOURCE004:
        raise ValueError('Mathematical implementation004 changed')
    saved = json.loads((PARENT / 'evidence/attempt_004/preservation_manifest.json').read_text())
    for r in saved['files']:
        if sha(ROOT / r['destination']) != r['sha256']:
            raise ValueError('Historical evidence changed')
    actual = manifest()
    if actual != json.loads((HERE / 'source_manifest.json').read_text()):
        raise ValueError('Pilot source mismatch')
    policy()
    return actual


def identity():
    m = verify()
    if os.environ.get('EXPECTED_STUDY_HASH') != m['source_hash'] or os.environ.get('EXPECTED_CORE_HASH') != CORE:
        raise ValueError('Submission hash mismatch')
    commit = os.environ.get('RUN_COMMIT', '')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit) or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Exact published commit and Slurm allocation required')
    return dict(execution_commit=commit, source_hash=m['source_hash'], core_hash=CORE,
                model_source_hash=SOURCE004, pinned_commit=PIN, job_id=os.environ['SLURM_JOB_ID'],
                backend='upstream', architecture='SISO', policy_version=policy()['version'],
                policy_sha256=sha(HERE / 'numeric_acceptance_v1.json'), TEST='NOT_RUN', test_evaluation_count=0)


def require_gate():
    from .gate import expected_ids
    base = identity()
    record = json.loads(GATE.read_text())
    if any(record.get(k) != v for k, v in base.items()) or record['status'] != ACCEPTED:
        raise ValueError('Current allocation admission required')
    if [r['case_id'] for r in record['cases']] != expected_ids() or not all(required_pass(r) for r in record['cases']):
        raise ValueError('Incomplete/failed admission evidence')
    return sha(GATE)


def assert_upstream(model=None):
    from experiments.mamba3_three_time.backends import current
    if current() != 'upstream':
        raise ValueError('Only original upstream arithmetic authorized')
    if model is not None and (model.is_mimo or model.chunk_size != 64):
        raise ValueError('Only frozen SISO chunk64')


def backend_guard(model):
    assert_upstream(model)
    return model.times.register_forward_pre_hook(lambda _m, _a: assert_upstream(model))


def backbone_hash(model):
    return tensor_hash({k: v for k, v in model.state_dict().items() if not k.startswith('times.')})


def effective_check(config, mode):
    from recbole.config import Config
    from experiments.mamba3_time_mechanisms.model import MechanismMamba3Rec
    reference = json.loads((ROOT / plan()['historical_reference']['source_json']).read_text())['config']
    reference.update(checkpoint_dir=config['checkpoint_dir'], use_gpu=config['use_gpu'], device=config['device'])
    ref = Config(model=MechanismMamba3Rec, config_dict=reference)
    ignore = {'model', 'three_time_mode', 'time_mechanism_mode'}
    a = {k: v for k, v in config.final_config_dict.items() if k not in ignore}
    b = {k: v for k, v in ref.final_config_dict.items() if k not in ignore}
    if a != b or config['three_time_mode'] != mode:
        raise ValueError('Effective frozen config mismatch: ' + repr([k for k in set(a) | set(b) if a.get(k) != b.get(k)]))
    return dict(status='PASS', weight_decay=config['weight_decay'], clip_grad_norm=config['clip_grad_norm'],
                enable_amp=config['enable_amp'], enable_scaler=config['enable_scaler'])


def runtime(require_cuda=False):
    import importlib.metadata
    import torch
    ref = json.loads((ROOT / plan()['historical_reference']['source_json']).read_text())['runtime']
    names = dict(torch='torch', recbole='recbole', mamba_ssm='mamba-ssm', triton='triton',
                 numpy='numpy', tilelang='tilelang', apache_tvm_ffi='apache-tvm-ffi')
    values = {k: importlib.metadata.version(v) for k, v in names.items()}
    if any(values[k] != ref[k] for k in names):
        raise ValueError('Frozen environment mismatch')
    pinned = upstream()
    if require_cuda and (not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name()):
        raise ValueError('A100 allocation required')
    return dict(**values, pinned_commit=pinned['pinned_commit'], upstream_manifest_sha256=pinned['manifest_sha256'],
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if require_cuda else None,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
