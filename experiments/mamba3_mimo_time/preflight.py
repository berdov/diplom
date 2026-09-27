"""Fresh masked CPU process; durable diagnostics before assertions, no forward."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from . import config as c
from .records import read, sha, create, update
from .process_env import visibility
from .provenance import verify, runtime, imported_sources


def run(evidence_path=None):
    path = Path(evidence_path) if evidence_path is not None else None
    result = dict(status='RUNNING', execution_attempt=c.EXECUTION_ATTEMPT, stages=[],
                  last_successful_stage=None, cuda_forward=0, optimizer_steps=0,
                  scientific_fits=0, TEST='NOT_RUN', test_evaluation_count=0)
    if path is not None:
        create(path, result)

    def persist():
        if path is not None:
            update(path, result)

    def observe(stage, mode=None, config=None, model=None):
        torch = sys.modules.get('torch')
        row = dict(stage=stage, mode=mode, cpu_mask=visibility(os.environ),
                   torch_imported=torch is not None,
                   cuda_initialized=torch.cuda.is_initialized() if torch is not None else None)
        if config is not None:
            row.update(effective_device=str(config['device']), use_gpu=config['use_gpu'], gpu_id=config['gpu_id'])
        if model is not None:
            row.update(parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                       buffer_devices=sorted({str(p.device) for p in model.buffers()}),
                       parameters=sum(p.numel() for p in model.parameters()))
        result['stages'].append(row)
        persist()
        # Preserve the offending stage before enforcing the CPU boundary.
        if row['cpu_mask'] != dict(present=True, value=''):
            raise ValueError('CPU mask absent/changed; isolate before starting Python')
        if row['cuda_initialized'] is True:
            raise ValueError('CPU preflight initialized CUDA at ' + stage)
        if config is not None and (row['effective_device'] != 'cpu' or row['use_gpu'] is not False or row['gpu_id'] != ''):
            raise ValueError('CPU Config must retain empty gpu_id and CPU device')
        if model is not None and any(v != 'cpu' for v in row['parameter_devices'] + row['buffer_devices']):
            raise ValueError('Non-CPU model tensor in preflight')
        result['last_successful_stage'] = stage
        persist()

    try:
        observe('process_entry')
        import importlib
        import torch
        observe('after_import_torch')
        m = verify()
        result['source_hash'] = m['source_hash']
        for name in ('admission','smoke','runner','trainer','report','pipeline'):
            importlib.import_module('experiments.mamba3_mimo_time.' + name)
        observe('after_runtime_imports')
        from .state import initialization
        result['initialization'] = initialization('cpu', observe=observe)
        observe('after_initialization')
        old = read(c.PILOT)
        for file, expected in [(c.ROOT/'outputs/data/protocol_b_manifest.json', old['manifest_sha256']),
                               (c.ROOT/'experiments/mamba3_timeaware/runs/train_time_stats_001.json', old['train_time_stats_sha256'])]:
            if sha(file) != expected:
                raise ValueError('Frozen data metadata changed')
        inter = c.ROOT/'data/processed/protocol_b/recbole/kuairand/kuairand.inter'
        if not inter.is_file():
            raise FileNotFoundError(inter)
        result.update(runtime=runtime(False), imported_sources=imported_sources())
        observe('after_runtime_and_imported_source_verification')
        if torch.cuda.is_initialized():
            raise ValueError('CPU preflight initialized CUDA at final guard')
        result.update(status='PASS', final_cuda_initialized=False)
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        persist()
        raise
    persist()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.evidence), indent=2))
