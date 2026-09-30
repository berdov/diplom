"""CPU integration and construction only; no MIMO forward, fitting or evaluation."""
import argparse
import importlib
import io
import json
import os
import re
import tempfile
import traceback
import unittest
from pathlib import Path
from . import config as c
from .provenance import verify,runtime,imported_sources
from experiments.mamba3_mimo_time.records import create,update,read,sha,now


def run(commit,attempt='001',evidence=None):
    result=dict(status='RUNNING',execution_commit=commit,execution_attempt=attempt,scientific_fits=0,
                TEST='NOT_RUN',test_evaluation_count=0,model_forward_calls=0,started_at=now())
    if evidence:create(evidence,result)
    try:
        if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact commit for CPU evidence')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA before Python/PyTorch import')
        m=verify(attempt);result['source_hash']=m['source_hash']
        import torch
        if torch.cuda.is_initialized():raise ValueError('CUDA initialized before CPU checks')
        for module in ('runner','pipeline','report','state','submit','continuation'):
            importlib.import_module('experiments.mamba3_head_timescales.confirmation.'+module)
        stream=io.StringIO()
        suite=unittest.defaultTestLoader.discover(str(c.HERE/'tests'))
        outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        result['cpu_tests']=dict(run=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),skipped=len(outcome.skipped),log=stream.getvalue())
        if evidence:update(evidence,result)
        if not outcome.wasSuccessful() or outcome.skipped or outcome.testsRun<20:raise ValueError('Required integration tests failed/skipped/missing')
        from .state import initialization
        with tempfile.TemporaryDirectory() as directory:result['initialization']=initialization(directory)
        old=read(c.pilot_path('fixed'))
        for path,expected in [('outputs/data/protocol_b_manifest.json',old['manifest_sha256']),
                              ('experiments/mamba3_timeaware/runs/train_time_stats_001.json',old['train_time_stats_sha256']),
                              ('data/processed/protocol_b/recbole/kuairand/kuairand.inter',old['protocol']['recbole_inter_sha256'])]:
            if sha(c.ROOT/path)!=expected:raise ValueError('Frozen data/reference changed')
        result.update(runtime=runtime(False,attempt),imported_sources=imported_sources(attempt),cuda_initialized=torch.cuda.is_initialized(),
                      cuda_mask=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if result['cuda_initialized']:raise ValueError('CPU preflight initialized CUDA')
        result['status']='PASS'
    except BaseException as exc:result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    result['finished_at']=now()
    if evidence:update(evidence,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution-commit',required=True);p.add_argument('--attempt',choices=['001','002'],default='001');p.add_argument('--evidence',type=Path)
    args=p.parse_args();r=run(args.execution_commit,args.attempt,args.evidence)
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='PASS' else 1)
