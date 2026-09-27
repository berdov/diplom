"""Source/data checks only; login does not run model/GPU work."""
import argparse
import json
from pathlib import Path
from .config import HERE, ATTEMPT, ROOT, PILOT
from .provenance import verify, identity, login_verify, create_record, sha, imported_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allocation',action='store_true')
    parser.add_argument('--runtime-preflight-only',type=Path,metavar='FIXTURE_DIRECTORY')
    parser.add_argument('--commit')
    args=parser.parse_args()
    if args.runtime_preflight_only and (args.allocation or args.commit):
        raise ValueError('Runtime-only fixture check cannot run an allocation or login validation')
    m=verify()
    if args.runtime_preflight_only:
        directory=args.runtime_preflight_only.resolve()
        if directory==ATTEMPT.resolve() or not (directory/'test_fixture.json').is_file():
            raise ValueError('Explicit temporary test fixture required')
        base=identity(submission_path=directory/'submission.json',login_path=directory/'login.json')
    elif args.allocation:
        base=identity()
    else:
        if not args.commit:
            raise ValueError('Exact published commit required for login preflight')
        base=login_verify(args.commit)
    # Import both implementations and the reused trainer before checking actual paths.
    from experiments.mamba3_context_time.model import ContextMamba3Rec
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.validation_pilot.trainer import trainer_class
    from experiments.mamba3_three_time.validation_pilot.provenance import runtime
    from experiments.mamba3_time_confirmation.config import STATS,STATS_SHA,MANIFEST,MANIFEST_SHA
    import torch
    row=dict(status='RUNNING',source_hash=m['source_hash'],runtime=runtime(args.allocation),
             imported_sources=imported_sources(), provenance=base)
    if sha(STATS)!=STATS_SHA or sha(MANIFEST)!=MANIFEST_SHA:
        raise ValueError('Frozen data metadata changed')
    data=ROOT/'data/processed/protocol_b/recbole/kuairand/kuairand.inter'
    if not data.is_file():
        raise FileNotFoundError(data)
    row['data']=dict(path=str(data),size=data.stat().st_size,manifest_sha256=MANIFEST_SHA,stats_sha256=STATS_SHA)
    if args.allocation:
        row['data']['sha256']=sha(data)
        if row['data']['sha256']!=json.loads(STATS.read_text())['protocol_b_inter_sha256']:
            raise ValueError('Protocol B inter changed')
    elif torch.cuda.is_initialized():
        raise ValueError('Login preflight initialized CUDA')
    row['status']='PASS'
    if not args.runtime_preflight_only:
        create_record(ATTEMPT/('allocation_preflight.json' if args.allocation else 'login_preflight.json'),row)
    else:
        row.update(test_fixture=True,model_forward=0,optimizer_steps=0,scientific_fits=0,full_dataset_pass=0)
    print(json.dumps(dict(status='PASS',source_hash=m['source_hash'],imported_modules=len(row['imported_sources']),
                         runtime_preflight_only=bool(args.runtime_preflight_only),model_forward=0,optimizer_steps=0)))


if __name__=='__main__':
    main()
