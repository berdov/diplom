"""Fresh CPU-only construction/config replay, without forward or data evaluation."""
import argparse
import importlib
import json
import os
import tempfile
import traceback
from pathlib import Path
from . import config as c
from .records import create, update, read, sha
from .provenance import verify, runtime, imported_sources


def run(evidence=None):
    result = dict(status='RUNNING',stages=[],model_forward_calls=0,scientific_fits=0,TEST='NOT_RUN',test_evaluation_count=0)
    if evidence:
        create(evidence,result)
    def persist():
        if evidence:
            update(evidence,result)
    def observe(stage,mode=None,config=None):
        import sys
        torch = sys.modules.get('torch')
        row = dict(stage=stage,mode=mode,cuda_mask=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   cuda_initialized=torch.cuda.is_initialized() if torch else False)
        if config is not None:
            row.update(device=str(config['device']),use_gpu=config['use_gpu'],gpu_id=config['gpu_id'])
        result['stages'].append(row)
        persist()
        if row['cuda_mask'] != '' or row['cuda_initialized']:
            raise ValueError('CPU preflight must not initialize/reopen CUDA')
        if config is not None and (row['device'],row['use_gpu'],row['gpu_id']) != ('cpu',False,''):
            raise ValueError('CPU Config reopened GPU')
    try:
        observe('process_entry')
        m = verify()
        result['source_hash'] = m['source_hash']
        for name in ('runner','pipeline','report','state'):
            importlib.import_module('experiments.mamba3_mimo_time.confirmation.'+name)
        observe('runtime_imports')
        from .state import initialization
        with tempfile.TemporaryDirectory() as directory:
            result['initialization'] = initialization(directory,observe)
        old = read(c.pilot_path('base'))
        for path, expected in [('outputs/data/protocol_b_manifest.json',old['manifest_sha256']),
                               ('experiments/mamba3_timeaware/runs/train_time_stats_001.json',old['train_time_stats_sha256'])]:
            if sha(c.ROOT/path) != expected:
                raise ValueError('Frozen data metadata changed')
        if not (c.ROOT/'data/processed/protocol_b/recbole/kuairand/kuairand.inter').is_file():
            raise FileNotFoundError('Frozen interaction file')
        result.update(runtime=runtime(False),imported_sources=imported_sources())
        observe('final_guard')
        result.update(status='PASS',final_cuda_initialized=False)
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
        persist()
        raise
    persist()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence',type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.evidence),indent=2))
