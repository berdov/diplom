"""Explicit common-state mapping, immutable historical initialization and RNG."""
import json
import torch
from . import config as c
from experiments.mamba3_mimo_time.records import read
from experiments.mamba3_context_time.provenance import tensor_hash
from experiments.mamba3_three_time.confirmation.state import rng_record,compare_optimizer_settings


def effective_check(config,variant):
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    old=read(c.PILOT);values=dict(old['config']);cpu=config['use_gpu'] is False
    if cpu:values.update(device='cpu',use_gpu=False,gpu_id='')
    reference=Config(model=ThreeTimeMamba3Rec,config_dict=values)
    allowed={'checkpoint_dir','memory_mode'}|({'device','use_gpu','gpu_id'} if cpu else set())
    actual=json.loads(json.dumps(config.final_config_dict,default=str))
    for other in (json.loads(json.dumps(reference.final_config_dict,default=str)),old['effective_config']):
        bad=[k for k in set(actual)|set(other) if k not in allowed and actual.get(k)!=other.get(k)]
        if bad:raise ValueError('Effective config drift '+repr(bad))
    if config['memory_mode']!=variant or config['three_time_mode']!='dual' or config['seed']!=2026:raise ValueError('Variant/mode/seed drift')
    if cpu and any(str(x['device'])!='cpu' or x['use_gpu'] is not False or x['gpu_id']!='' for x in (config,reference)):raise ValueError('CPU config reopened GPU')
    return dict(status='PASS',allowed_differences=sorted(allowed),cpu_reference_device=str(reference['device']))


def initial(model,loader=None):
    hashes={k:tensor_hash(v.state_dict()) for k,v in model.times.calibrators.items()}
    return dict(initial_backbone_sha256=tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith('times.') and k!='beta'}),
                initial_common_calibrator_hashes=hashes,initial_beta=float(model.beta.detach()) if hasattr(model,'beta') else None,rng_components=rng_record(loader))


def require_initial(record):
    old=read(c.PILOT)
    for new,prior in [('initial_backbone_sha256','initial_backbone_sha256'),('initial_common_calibrator_hashes','initial_calibrator_hashes'),('rng_components','rng_components')]:
        if record[new]!=old[prior]:raise ValueError('Historical initialization/RNG mismatch: '+new)
    if record['initial_beta'] not in (None,0.):raise ValueError('Initial beta must be exactly zero')


def paired(a,b,first_batch=False):
    for key in ('initial_backbone_sha256','initial_common_calibrator_hashes','rng_components','protocol',
                'manifest_sha256','train_time_stats_sha256','verified_history_stats','precision'):
        if key in a or key in b:
            if a.get(key)!=b.get(key):raise ValueError('Pair mismatch '+key)
    if 'optimizer_settings' in a or 'optimizer_settings' in b:compare_optimizer_settings(a['optimizer_settings'],b['optimizer_settings'])
    if first_batch and a['first_train_batch_sha256']!=b['first_train_batch_sha256']:raise ValueError('First batch mismatch')


from .model import transfer_common
