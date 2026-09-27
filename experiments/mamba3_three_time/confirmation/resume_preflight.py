"""Read-only inherited evidence and runtime checks; no forward/backward/evaluation."""
import argparse
import json
from pathlib import Path
from . import config as c, provenance as p, resume_config as r, resume_provenance as q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allocation',action='store_true')
    parser.add_argument('--commit')
    parser.add_argument('--runtime-preflight-only',type=Path)
    args = parser.parse_args()
    if sum((args.allocation, bool(args.commit), bool(args.runtime_preflight_only))) != 1:
        raise ValueError('Exactly one preflight mode required')
    if args.runtime_preflight_only:
        directory = args.runtime_preflight_only.resolve()
        if directory == r.LOGS.resolve() or not (directory/'test_fixture.json').is_file():
            raise ValueError('Explicit temporary fixtures required')
        base = q.identity(submission_path=directory/'submission.json',login_path=directory/'login.json')
    elif args.allocation:
        base = q.identity()
    else:
        base = q.login_verify(args.commit)
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.validation_pilot.trainer import trainer_class
    from experiments.mamba3_three_time.validation_pilot.provenance import runtime
    from experiments.mamba3_time_confirmation.config import STATS, STATS_SHA, MANIFEST, MANIFEST_SHA
    import torch
    if p.sha(STATS) != STATS_SHA or p.sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError('Frozen data metadata changed')
    if not (c.ROOT/'data/processed/protocol_b/recbole/kuairand/kuairand.inter').is_file():
        raise FileNotFoundError('Protocol B data missing')
    row = dict(status='PASS',source_hash=p.verify()['source_hash'],provenance=base,
               inherited=q.parent_ready(),lineage=q.verify_lineage(),runtime=runtime(args.allocation),
               imported_sources=p.imported_sources(),model_forward=0,optimizer_steps=0,scientific_fits=0)
    if not args.allocation and torch.cuda.is_initialized():
        raise ValueError('CPU preflight initialized CUDA')
    if not args.runtime_preflight_only:
        p.create_record(r.LOGS/('allocation_preflight.json' if args.allocation else 'login_preflight.json'),row)
    print(json.dumps(dict(status='PASS',source_hash=row['source_hash'],model_forward=0,optimizer_steps=0,scientific_fits=0)))


if __name__ == '__main__':
    main()
