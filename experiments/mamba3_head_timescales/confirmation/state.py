"""Exact effective config guard; unchanged common-state and optimizer mapping."""
import json
from . import config as c
from experiments.mamba3_mimo_time.records import read
from experiments.mamba3_head_timescales.state import initial, paired

REQUIRED = ('initial_backbone_sha256','initial_common_calibrator_hashes','initial_alpha','rng_components',
            'protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings')


def effective_check(config,variant,seed):
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    old=read(c.pilot_path(variant))
    values=dict(old['config'])
    cpu=config['use_gpu'] is False
    if cpu:values.update(device='cpu',use_gpu=False,gpu_id='')
    ref=Config(model=ThreeTimeMamba3Rec,config_dict=values)
    allowed={'seed','checkpoint_dir'} | ({'device','use_gpu','gpu_id'} if cpu else set())
    actual=json.loads(json.dumps(config.final_config_dict,default=str))
    reference=json.loads(json.dumps(ref.final_config_dict,default=str))
    for other in (reference,old['effective_config']):
        bad=[k for k in set(actual)|set(other) if k not in allowed and actual.get(k)!=other.get(k)]
        if bad:raise ValueError('Effective config drift: '+repr(bad))
    if config['time_scale_mode']!=variant or config['three_time_mode']!='dual' or config['seed']!=seed:
        raise ValueError('Variant/mode/seed drift')
    if cpu and any(str(x['device'])!='cpu' or x['use_gpu'] is not False or x['gpu_id']!='' for x in (config,ref)):
        raise ValueError('CPU Config reopened GPU')
    return dict(status='PASS',allowed_differences=sorted(allowed),cpu_reference_device=str(ref['device']),seed=seed)


def require_initial(record,first_batch=False):
    missing=[k for k in REQUIRED if k not in record]
    if missing:raise ValueError('Missing paired state: '+repr(missing))
    variant=record['time_scale_mode']
    expected={} if variant=='fixed' else {f'{m}.alpha':dict(shape=[1 if variant=='shared_tau' else 2],values=[0.0]*(1 if variant=='shared_tau' else 2)) for m in ('decay','scan')}
    # initial() uses calibrator names as keys, not full state_dict names.
    expected={k.removesuffix('.alpha'):v for k,v in expected.items()}
    if record['initial_alpha']!=expected:raise ValueError('Alpha initialization/shapes')
    if set(record['initial_common_calibrator_hashes'])!={'decay','scan'}:raise ValueError('Common substate mapping')
    if set(record['rng_components'])!={'python','numpy','cpu','cuda','aggregate','loader_generator'}:
        raise ValueError('Missing Python/NumPy/CPU/CUDA/loader RNG evidence')
    if first_batch and not record.get('first_train_batch_sha256'):raise ValueError('Missing consumed batch')


def paired_records(a,b,first_batch=False):
    require_initial(a,first_batch);require_initial(b,first_batch)
    if a['seed']!=b['seed'] or a['time_scale_mode']==b['time_scale_mode']:
        raise ValueError('Pair must contain distinct variants of same seed')
    paired(a,b,first_batch=first_batch)


def initialization(directory):
    import torch
    from recbole.config import Config
    from recbole.utils import init_seed
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.config import SyntheticCatalog
    from experiments.mamba3_head_timescales.model import HeadTimescaleMamba3Rec
    from experiments.mamba3_three_time.confirmation.state import canonical_optimizer_settings,compare_optimizer_settings
    result=[]
    for seed in (2026,*c.SEEDS):
        rows=[]
        for variant in c.MODES:
            cfg=Config(model=ThreeTimeMamba3Rec,config_dict=c.settings(variant,seed,device='cpu',checkpoint_dir=directory))
            check=effective_check(cfg,variant,seed)
            init_seed(cfg['seed']+cfg['local_rank'],cfg['reproducibility'])
            model=HeadTimescaleMamba3Rec(cfg,SyntheticCatalog()).cpu()
            if any(t.device.type!='cpu' for t in [*model.parameters(),*model.buffers()]):raise ValueError('Off-CPU construction')
            row=dict(seed=seed,variant=variant,parameter_count=sum(p.numel() for p in model.parameters()),
                     effective_config_check=check,**initial(model))
            if row['parameter_count']!=c.COUNTS[variant]:raise ValueError('CPU parameter count')
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=cfg['weight_decay'])
            metadata=canonical_optimizer_settings([{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups])
            compare_optimizer_settings(metadata,json.loads(json.dumps(metadata)))
            pilot=read(c.pilot_path(variant));compare_optimizer_settings(metadata,pilot['optimizer_settings'])
            row['optimizer_roundtrip']='PASS'
            if variant=='head_tau':
                for cal in model.times.calibrators.values():
                    if cal.alpha.shape!=(2,) or cal.alpha.stride()!=(1,) or cal.alpha.numel()!=2:
                        raise ValueError('Head alpha parameters are not independent contiguous scalars')
            if seed==2026:
                for key in ('initial_backbone_sha256','initial_common_calibrator_hashes','initial_alpha'):
                    if row[key]!=pilot[key]:raise ValueError('Seed2026 setup replay: '+key)
                for key in ('python','numpy','cpu'):
                    if row['rng_components'][key]!=pilot['rng_components'][key]:raise ValueError('Seed2026 RNG replay: '+key)
                row['pilot_replay']=dict(backbone='PASS',common_mlp_and_buffers='PASS',alpha='PASS',python_numpy_cpu='PASS',
                                         cuda='NOT_REPLAYED_CPU',loader_generator='NOT_REPLAYED_CPU',first_batch='NOT_REPLAYED_NO_FIT')
            rows.append(row)
            del optimizer,model
        for left,right in ((0,1),(0,2),(1,2)):paired(rows[left],rows[right])
        result.extend(rows)
    if torch.cuda.is_initialized():raise ValueError('CPU setup initialized CUDA')
    return dict(status='PASS',rows=result,model_forward_calls=0,scientific_fits=0)
