"""Login CPU construction and hashes only, never GPU forward or real data."""

import argparse
import importlib.metadata
import json
import torch
from .config import ARCHITECTURES
from .initialization import initialization_suite
from .provenance import verify, upstream
from .evidence import create_record, update_record


def inspect(save=lambda value:None):
    result = dict(status="RUNNING",source=verify(),upstream=upstream(),
        versions={name:importlib.metadata.version(name) for name in
                  ("torch","mamba-ssm","triton","tilelang","recbole","numpy")},
        initialization={}, scientific_fits=0,TRAIN=0,VALID=0,TEST=0)
    save(result)
    if torch.cuda.is_initialized():
        raise RuntimeError("Login preflight must not initialize CUDA")
    for arch in ARCHITECTURES:
        def persist(value):
            result["initialization"][arch] = value
            save(result)
        initialization_suite(arch,"cpu",persist)
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization in CPU preflight")
    result["status"] = "PASS"
    save(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    create_record(args.output,{"status":"STARTING"})
    try:
        result = inspect(lambda value:update_record(args.output,value))
    except Exception:
        import traceback
        # Keep incremental initialization rows; append error without replacing them.
        from pathlib import Path
        result = json.loads(Path(args.output).read_text())
        result.update(status="FAIL",traceback=traceback.format_exc())
        update_record(args.output,result)
        raise
    print(json.dumps(dict(status=result["status"],counts={a:{r["mode"]:r["actual_count"]
        for r in v["rows"]} for a,v in result["initialization"].items()},
        source_hash=result["source"]["source_hash"],versions=result["versions"],
        upstream_manifest_sha256=result["upstream"]["manifest_sha256"]),indent=2))
