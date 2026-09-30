"""Explicit common-state mapping; alpha is the only extra trainable state."""
import json
import torch
from . import config as c
from experiments.mamba3_mimo_time.records import read
from experiments.mamba3_context_time.provenance import tensor_hash
from experiments.mamba3_three_time.confirmation.state import rng_record, compare_optimizer_settings

COMMON = {'reference','first.weight','first.bias','last.weight','last.bias'}


def effective_check(config, variant):
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    old = read(c.PILOT)
    values = dict(old['config'])
    cpu = config['use_gpu'] is False
    if cpu:
        values.update(device='cpu',use_gpu=False,gpu_id='')
    ref = Config(model=ThreeTimeMamba3Rec,config_dict=values)
    allowed = {'checkpoint_dir','time_scale_mode'} | ({'device','use_gpu','gpu_id'} if cpu else set())
    actual = json.loads(json.dumps(config.final_config_dict,default=str))
    reference = json.loads(json.dumps(ref.final_config_dict,default=str))
    for other in (reference,old['effective_config']):
        bad=[k for k in set(actual)|set(other) if k not in allowed and actual.get(k)!=other.get(k)]
        if bad:
            raise ValueError('Effective config drift: '+repr(bad))
    if config['time_scale_mode'] != variant or config['three_time_mode'] != 'dual' or config['seed'] != 2026:
        raise ValueError('Variant/mode/seed drift')
    if cpu and any(str(cfg['device'])!='cpu' or cfg['use_gpu'] is not False or cfg['gpu_id']!='' for cfg in (config,ref)):
        raise ValueError('CPU Config reopened GPU')
    return dict(status='PASS',allowed_differences=sorted(allowed),cpu_reference_device=str(ref['device']))


def initial(model, loader=None):
    common, alpha = {}, {}
    for name, cal in model.times.calibrators.items():
        values=cal.state_dict()
        extra=set(values)-COMMON
        expected=set() if model.time_scale_mode=='fixed' else {'alpha'}
        if extra!=expected or not COMMON.issubset(values):
            raise ValueError('Unexpected calibrator state')
        common[name]=tensor_hash({k:values[k] for k in sorted(COMMON)})
        if extra:
            a=values['alpha']
            if tuple(a.shape)!=((1,) if model.time_scale_mode=='shared_tau' else (2,)) or torch.count_nonzero(a):
                raise ValueError('Alpha must start at exact deterministic zero')
            alpha[name]=dict(shape=list(a.shape),values=a.detach().cpu().tolist())
    return dict(initial_backbone_sha256=tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith('times.')}),
                initial_common_calibrator_hashes=common,initial_alpha=alpha,rng_components=rng_record(loader))


def paired(a,b,first_batch=False):
    for k in ('initial_backbone_sha256','initial_common_calibrator_hashes','rng_components','protocol','manifest_sha256',
              'train_time_stats_sha256','verified_history_stats','precision'):
        if k in a or k in b:
            if k not in a or k not in b or a[k]!=b[k]:
                raise ValueError('Pair mismatch: '+k)
    if 'optimizer_settings' in a or 'optimizer_settings' in b:
        compare_optimizer_settings(a['optimizer_settings'],b['optimizer_settings'])
    if first_batch and a['first_train_batch_sha256']!=b['first_train_batch_sha256']:
        raise ValueError('First consumed batch mismatch')


def transfer_common(source,target):
    a,b=source.state_dict(),target.state_dict()
    allowed=set(c.plan()['alpha_keys']) if target.time_scale_mode!='fixed' else set()
    if set(b)-set(a)!=allowed or set(a)-set(b):
        raise ValueError('Unexpected transfer keys')
    for k,v in a.items():
        if v.shape!=b[k].shape or v.dtype!=b[k].dtype:
            raise ValueError('Common state shape/dtype drift')
    result=target.load_state_dict(a,strict=False)
    if set(result.missing_keys)!=allowed or result.unexpected_keys:
        raise ValueError('Transfer missing/extra')
    if any(not torch.equal(v,target.state_dict()[k]) for k,v in a.items()):
        raise ValueError('Common state not exact')
    return sorted(a)
