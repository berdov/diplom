"""Imports/config/counts on login; full data digest only in the allocation."""
import json
import torch
from recbole.config import Config
from experiments.mamba3_three_time.config import SyntheticCatalog
from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
from experiments.mamba3_three_time.initialization import seed_all
from experiments.mamba3_time_confirmation.config import STATS, STATS_SHA, MANIFEST, MANIFEST_SHA
from .config import MODES, COUNTS, settings, ROOT, LOGS
from .provenance import verify, runtime, effective_check, backbone_hash, rng_hash, tensor_hash, sha, create_record
from .historical import audit


def initialization(device='cpu'):
    rows = []
    for mode in MODES:
        values = settings(mode)
        values.update(use_gpu=str(device).startswith('cuda'), device=device)
        config = Config(model=ThreeTimeMamba3Rec, config_dict=values)
        effective_check(config, mode)
        seed_all(2026)
        model = ThreeTimeMamba3Rec(config, SyntheticCatalog()).to(device)
        count = sum(p.numel() for p in model.parameters())
        independent = mode == 'dual' or not ({id(p) for p in model.times.calibrators['write'].parameters()} &
                                            {id(p) for p in model.times.calibrators['phase'].parameters()})
        rows.append(dict(mode=mode, parameters=count, backbone=backbone_hash(model), rng=rng_hash(),
                         calibrators={k: tensor_hash(v.state_dict()) for k, v in model.times.calibrators.items()},
                         independent=independent, zero_last=all(bool((v.last.weight == 0).all() and (v.last.bias == 0).all())
                                                               for v in model.times.calibrators.values())))
        del model
    passed = (all(r['parameters'] == COUNTS[r['mode']] and r['independent'] and r['zero_last'] for r in rows)
              and rows[0]['backbone'] == rows[1]['backbone'] and rows[0]['rng'] == rows[1]['rng']
              and len(set(v for r in rows for v in r['calibrators'].values())) == 1)
    return dict(passed=passed, rows=rows, device=device)


def run(full_data=False):
    m = verify()
    value = dict(status='RUNNING', source_hash=m['source_hash'], runtime=runtime(full_data))
    value['historical'] = audit()
    value['initialization'] = initialization()
    if not value['initialization']['passed']:
        raise ValueError('CPU initialization parity failed: ' + repr(value['initialization']))
    if sha(STATS) != STATS_SHA or sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError('Frozen data metadata changed')
    path = ROOT / 'data/processed/protocol_b/recbole/kuairand/kuairand.inter'
    if not path.is_file():
        raise FileNotFoundError(path)
    value['data'] = dict(path=str(path), size=path.stat().st_size, stats_sha256=STATS_SHA, manifest_sha256=MANIFEST_SHA)
    if full_data:
        digest = sha(path)
        if digest != json.loads(STATS.read_text())['protocol_b_inter_sha256']:
            raise ValueError('Protocol B data mismatch')
        value['data']['sha256'] = digest
    elif torch.cuda.is_initialized():
        raise ValueError('Login preflight initialized CUDA')
    value['status'] = 'PASS'
    return value


if __name__ == '__main__':
    value = run()
    create_record(LOGS / 'login_preflight.json', value)
    print(json.dumps(dict(status=value['status'], source_hash=value['source_hash'], initialization=value['initialization'])))
