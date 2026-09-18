"""Content-addressed sources and exclusive, durable one-shot guards."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from experiments.mamba3_time_confirmation.provenance import atomic_json, sha, verify as verify_confirmation
from experiments.mamba3_time_mechanisms.provenance import fingerprint, PIN
from .config import HERE, ROOT, CORE, CONFIRMATION, plan, settings, scientific_settings


def now():
    return datetime.now(timezone.utc).isoformat()


def source_manifest():
    sources = sorted([*HERE.glob('*.py'), *(HERE / 'tests').glob('*.py'), HERE / 'study_plan.json',
                      ROOT / 'slurm/mamba3_input_time.sh'])
    files = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(schema_version=1, files=files, source_hash=digest)


def verify():
    from experiments.mamba3_context_time.provenance import verify as verify_context
    if fingerprint() != CORE or verify_confirmation()['source_hash'] != CONFIRMATION:
        raise ValueError('Frozen core/confirmation changed')
    if verify_context()['source_hash'] != plan()['frozen_context_source_hash']:
        raise ValueError('Frozen context study changed')
    actual = source_manifest()
    if actual != json.loads((HERE / 'source_manifest.json').read_text()):
        raise ValueError('New study source manifest mismatch')
    reference = plan()['historical_reference']
    path = ROOT / reference['source_json']
    if sha(path) != reference['sha256']:
        raise ValueError('Historical separate changed')
    old = json.loads(path.read_text())
    for task in plan()['tasks']:
        if scientific_settings(settings(task)) != scientific_settings(old['config']):
            raise ValueError('Scientific configuration drift')
    return actual


def require_submission():
    manifest = verify()
    if os.environ.get('EXPECTED_CORE_HASH') != CORE or os.environ.get('EXPECTED_STUDY_HASH') != manifest['source_hash']:
        raise ValueError('Submission/source hashes differ')
    commit = os.environ.get('RUN_COMMIT', '')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Login-verified execution commit required')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm allocation required')
    return manifest


def exclusive(path, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(identity, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def acquire(paths, identity):
    if paths['result'].exists() or paths['checkpoint'].exists() or paths['metadata'].exists():
        raise FileExistsError('Existing scientific artifacts; no overwrite/retry')
    exclusive(paths['lock'], identity)
    if paths['result'].exists():
        raise FileExistsError('Result appeared during lock acquisition')


def tensor_hash(values):
    import torch
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def backbone_hash(model):
    return tensor_hash({k: v for k, v in model.state_dict().items() if not k.startswith('input_adapter.')})


def historical_backbone_hash(model):
    return tensor_hash({k: v for k, v in model.state_dict().items()
                        if not k.startswith(('input_adapter.', 'mechanisms.'))})


def rng_hash():
    import random
    import numpy as np
    import torch
    state = np.random.get_state()
    text = json.dumps([random.getstate(), [state[0], state[1].tolist(), *state[2:]]])
    values = {'cpu': torch.random.get_rng_state()}
    if torch.cuda.is_initialized():
        values.update({f'cuda:{i}': x for i, x in enumerate(torch.cuda.get_rng_state_all())})
    return hashlib.sha256((text + tensor_hash(values)).encode()).hexdigest()


def save_state_dict(model, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        torch.save(model.state_dict(), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return sha(path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        atomic_json(HERE / 'source_manifest.json', source_manifest())
    else:
        print(json.dumps(verify(), indent=2))
