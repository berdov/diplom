"""CPU-only regressions plus all-seed construction; Git unnecessary on compute."""
import argparse
import io
import json
import os
import re
import traceback
import unittest
from pathlib import Path
from . import config as c
from .provenance import verify, runtime, imported_sources
from experiments.mamba3_mimo_time.records import create, update, read, sha, now


def run(commit,evidence=None):
    result=dict(status='RUNNING',execution_commit=commit,study_id=c.STUDY,scientific_fits=0,
                TEST='NOT_RUN',test_evaluation_count=0,mimo_model_forward_calls=0,started_at=now())
    if evidence:create(evidence,result)
    try:
        if not re.fullmatch('[0-9a-f]{40}',commit) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Exact commit and hidden GPU required')
        m=verify();c.index();result['source_hash']=m['source_hash']
        import torch
        if torch.cuda.is_initialized():raise ValueError('CUDA already initialized')
        from ..tests.test_centered import CenteredTests
        from .tests.test_confirmation import ConfirmationTests
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(CenteredTests),unittest.defaultTestLoader.loadTestsFromTestCase(ConfirmationTests)])
        stream=io.StringIO();outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        result['cpu_tests']=dict(run=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),skipped=len(outcome.skipped),log=stream.getvalue())
        if evidence:update(evidence,result)
        if not outcome.wasSuccessful() or outcome.skipped or outcome.testsRun<33:raise ValueError('CPU tests failed/skipped/missing')
        old=read(c.PILOT)
        for path,expected in [('outputs/data/protocol_b_manifest.json',old['manifest_sha256']),('experiments/mamba3_timeaware/runs/train_time_stats_001.json',old['train_time_stats_sha256']),('data/processed/protocol_b/recbole/kuairand/kuairand.inter',old['protocol']['recbole_inter_sha256'])]:
            if sha(c.ROOT/path)!=expected:raise ValueError('Data bytes changed')
        if torch.cuda.is_initialized():raise ValueError('CPU preflight initialized CUDA')
        result.update(status='PASS',runtime=runtime(False),imported_sources=imported_sources(),cuda_initialized=False,parameter_counts=c.COUNTS)
    except BaseException as exc:result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    result['finished_at']=now()
    if evidence:update(evidence,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution-commit',required=True);p.add_argument('--evidence',type=Path);p.add_argument('--deny-git',action='store_true');args=p.parse_args()
    if args.deny_git:
        import subprocess
        original=subprocess.Popen
        class NoGitPopen(original):
            def __init__(self,command,*a,**kw):
                if isinstance(command,(list,tuple)) and Path(str(command[0])).name=='git':raise RuntimeError('Git forbidden during compute verification')
                super().__init__(command,*a,**kw)
        subprocess.Popen=NoGitPopen
    result=run(args.execution_commit,args.evidence);print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PASS' else 1)
