"""CPU configuration/runtime checks, no CUDA allocation, data loader or fit."""
import json
from . import config as c
from .records import read, sha
from .provenance import verify, runtime, imported_sources


def run():
    import importlib
    import torch
    m = verify()
    for name in ('admission','smoke','runner','trainer','report','pipeline'):
        importlib.import_module('experiments.mamba3_mimo_time.' + name)
    from .state import initialization
    result = initialization('cpu')
    if torch.cuda.is_initialized():
        raise ValueError('CPU preflight initialized CUDA')
    old = read(c.PILOT)
    for path, expected in [(c.ROOT/'outputs/data/protocol_b_manifest.json',old['manifest_sha256']),
                            (c.ROOT/'experiments/mamba3_timeaware/runs/train_time_stats_001.json',old['train_time_stats_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Frozen data metadata changed')
    inter = c.ROOT / 'data/processed/protocol_b/recbole/kuairand/kuairand.inter'
    if not inter.is_file():
        raise FileNotFoundError(inter)
    return dict(status='PASS',source_hash=m['source_hash'],runtime=runtime(False),initialization=result,
                imported_sources=imported_sources(),cuda_forward=0,scientific_fits=0,TEST='NOT_RUN',test_evaluation_count=0)


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
