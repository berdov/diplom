"""CPU-only import/hash/initialization checks in the existing environment."""

import importlib.metadata
import json
import traceback
import torch
from . import attempt_004 as attempt
from .provenance_004 import verify, upstream
from .evidence import create_record, update_record
from .initialization import initialization_suite
from .drift_capture import FIXED_LAUNCH


def main():
    attempt.require_unused()
    result=dict(attempt_id='004',status='RUNNING',initialization={},training_authorized=False,
                scientific_fits=0,TRAIN=0,VALID=0,TEST=0)
    create_record(attempt.PREFLIGHT,result)
    try:
        result.update(source=verify(),upstream=upstream(),versions={n:importlib.metadata.version(n)
            for n in ('torch','triton','tilelang','recbole','mamba-ssm')})
        from mamba_ssm.ops.triton.mamba3 import mamba3_siso_bwd
        from . import stable_adt
        result['matched_launch']=FIXED_LAUNCH
        for name,module in (('official',mamba3_siso_bwd),('candidate',stable_adt)):
            supported=any(all(getattr(c,k,None)==v for k,v in FIXED_LAUNCH.items()) for c in module.mamba3_siso_bwd_kernel_dqkv.configs)
            result[name+'_fixed_config_supported']=supported
            if not supported:raise RuntimeError('Frozen config not supported: '+name)
        for arch in ('SISO','MIMO'):
            def save(value):
                result['initialization'][arch]=value
                update_record(attempt.PREFLIGHT,result)
            initialization_suite(arch,'cpu',save)
        if torch.cuda.is_initialized():raise RuntimeError('Login preflight initialized CUDA')
        result['status']='PASS'
    except Exception:
        result.update(status='FAIL',traceback=traceback.format_exc())
        raise
    finally:update_record(attempt.PREFLIGHT,result)
    print(json.dumps(dict(status=result['status'],source_hash=result['source']['source_hash'],training_authorized=False)))


if __name__=='__main__':main()
