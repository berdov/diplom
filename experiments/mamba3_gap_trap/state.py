"""Exact common state and RNG mapping, one optional scalar outside the backbone."""
import torch
from . import config as c
from experiments.mamba3_context_time.provenance import tensor_hash
from experiments.mamba3_three_time.confirmation.state import rng_record
from experiments.mamba3_head_timescales.state import paired


def effective_check(config, variant):
    import json
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_mimo_time.records import read
    old=read(c.PILOT);values=dict(old['config']);cpu=config['use_gpu'] is False
    if cpu:values.update(device='cpu',use_gpu=False,gpu_id='')
    ref=Config(model=ThreeTimeMamba3Rec,config_dict=values)
    allowed={'checkpoint_dir','gap_trap_mode'} | ({'device','use_gpu','gpu_id'} if cpu else set())
    actual=json.loads(json.dumps(config.final_config_dict,default=str))
    for other in (json.loads(json.dumps(ref.final_config_dict,default=str)),old['effective_config']):
        bad=[k for k in set(actual)|set(other) if k not in allowed and actual.get(k)!=other.get(k)]
        if bad:raise ValueError('Effective config drift: '+repr(bad))
    if config['gap_trap_mode']!=variant or config['three_time_mode']!='dual' or config['seed']!=2026:
        raise ValueError('Variant/mode/seed drift')
    if cpu and any(str(x['device'])!='cpu' or x['use_gpu'] is not False or x['gpu_id']!='' for x in (config,ref)):
        raise ValueError('CPU Config reopened GPU')
    return dict(status='PASS',allowed_differences=sorted(allowed),cpu_reference_device=str(ref['device']))


def initial(model, loader=None):
    values=model.state_dict()
    extra={k:v for k,v in values.items() if k.startswith('gap_trap.')}
    expected={'gap_trap.alpha'} if model.gap_trap_mode=='gap_trap' else set()
    if set(extra)!=expected or any(v.shape!=torch.Size([]) or torch.count_nonzero(v) for v in extra.values()):
        raise ValueError('Expected exactly one deterministic zero scalar')
    return dict(initial_backbone_sha256=tensor_hash({k:v for k,v in values.items() if not k.startswith(('times.','gap_trap.'))}),
                initial_common_calibrator_hashes={n:tensor_hash(cal.state_dict()) for n,cal in model.times.calibrators.items()},
                initial_alpha={k:v.cpu().tolist() for k,v in extra.items()},rng_components=rng_record(loader))


def transfer_common(source,target):
    a,b=source.state_dict(),target.state_dict()
    allowed={'gap_trap.alpha'} if target.gap_trap_mode=='gap_trap' else set()
    if set(b)-set(a)!=allowed or set(a)-set(b):raise ValueError('Unexpected common state mapping')
    if any(v.shape!=b[k].shape or v.dtype!=b[k].dtype for k,v in a.items()):raise ValueError('Common shape/dtype drift')
    result=target.load_state_dict(a,strict=False)
    if set(result.missing_keys)!=allowed or result.unexpected_keys:raise ValueError('State transfer drift')
    if any(not torch.equal(v,target.state_dict()[k]) for k,v in a.items()):raise ValueError('Common state not exact')
    return sorted(a)
