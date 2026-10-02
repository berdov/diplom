"""Fresh CPU-only tests/construction and source checks; no model forward on GPU."""
import argparse
import importlib
import io
import json
import os
import re
import traceback
import unittest
from pathlib import Path
from . import config as c
from .provenance import verify,runtime,imported_sources
from experiments.mamba3_mimo_time.records import create,update,read,sha,now


def run(commit,evidence=None):
    result=dict(execution_attempt='001',status='RUNNING',execution_commit=commit,scientific_fits=0,TEST='NOT_RUN',test_evaluation_count=0,
                mimo_model_forward_calls=0,cpu_fixture_forward_backward=True,calibrator_cpu_checks=True,started_at=now())
    if evidence:create(evidence,result)
    def persist():
        if evidence:update(evidence,result)
    try:
        if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact commit for CPU evidence')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide GPU before process/PyTorch import')
        m=verify();result['source_hash']=m['source_hash']
        import torch
        if torch.cuda.is_initialized():raise ValueError('CUDA initialized before CPU checks')
        for name in ('runner','pipeline','report','state','gate','smoke','submit'):
            importlib.import_module('experiments.mamba3_gap_trap.'+name)
        stream=io.StringIO()
        suite=unittest.defaultTestLoader.discover(str(c.HERE/'tests'))
        outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        result['cpu_tests']=dict(run=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),skipped=len(outcome.skipped),log=stream.getvalue())
        persist()
        if not outcome.wasSuccessful() or outcome.skipped or outcome.testsRun<12:raise ValueError('Required CPU tests failed/skipped/missing')
        old=read(c.PILOT)
        for path,expected in [('outputs/data/protocol_b_manifest.json',old['manifest_sha256']),
                              ('experiments/mamba3_timeaware/runs/train_time_stats_001.json',old['train_time_stats_sha256']),
                              ('data/processed/protocol_b/recbole/kuairand/kuairand.inter',old['protocol']['recbole_inter_sha256'])]:
            if sha(c.ROOT/path)!=expected:raise ValueError('Data/reference bytes changed')
        if torch.cuda.is_initialized():raise ValueError('CPU checks initialized CUDA')
        result.update(status='PASS',runtime=runtime(False),imported_sources=imported_sources(),cuda_initialized=False,
                      cuda_mask=os.environ.get('CUDA_VISIBLE_DEVICES'),parameter_counts=c.COUNTS,gradcheck='PASS')
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    result['finished_at']=now();persist()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execution-commit',required=True);parser.add_argument('--evidence',type=Path)
    args=parser.parse_args();r=run(args.execution_commit,args.evidence)
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='PASS' else 1)
