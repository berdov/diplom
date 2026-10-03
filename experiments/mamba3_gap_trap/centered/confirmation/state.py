"""Seed-aware guards; published centered initialization and pairing unchanged."""
import json
from . import config as c
from ..state import initial, paired
from experiments.mamba3_mimo_time.records import read


def effective_check(config,variant,seed):
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    old=read(c.pilot_path(variant));values=dict(old['config']);cpu=config['use_gpu'] is False
    if cpu:values.update(device='cpu',use_gpu=False,gpu_id='')
    ref=Config(model=ThreeTimeMamba3Rec,config_dict=values)
    allowed={'seed','checkpoint_dir'} | ({'device','use_gpu','gpu_id'} if cpu else set())
    actual=json.loads(json.dumps(config.final_config_dict,default=str))
    for other in (json.loads(json.dumps(ref.final_config_dict,default=str)),old['effective_config']):
        bad=[k for k in set(actual)|set(other) if k not in allowed and actual.get(k)!=other.get(k)]
        if bad:raise ValueError('Effective config drift: '+repr(bad))
    if config['gap_trap_mode']!=variant or config['three_time_mode']!='dual' or config['seed']!=seed or seed not in c.SEEDS:
        raise ValueError('Variant/mode/seed drift')
    if cpu and any(str(x['device'])!='cpu' or x['use_gpu'] is not False or x['gpu_id']!='' for x in (config,ref)):
        raise ValueError('CPU Config reopened GPU')
    return dict(status='PASS',allowed_differences=sorted(allowed),seed=seed)


def require_initial(record):
    required=('initial_backbone_sha256','initial_common_calibrator_hashes','initial_alpha','rng_components',
              'protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings')
    if any(k not in record for k in required):raise ValueError('Missing required paired evidence')
    expected={} if record['gap_trap_mode']=='fixed_replay' else {'gap_trap.alpha':0.}
    if record['initial_alpha']!=expected:raise ValueError('Alpha initialization')
    if set(record['initial_common_calibrator_hashes'])!={'decay','scan'}:raise ValueError('Calibrator/reference mapping')
    if set(record['rng_components'])!={'python','numpy','cpu','cuda','aggregate','loader_generator'}:raise ValueError('Missing RNG evidence')
    reference=read(c.historical(record['seed']))
    for key,old in [('initial_backbone_sha256','initial_backbone_sha256'),('initial_common_calibrator_hashes','initial_calibrator_hashes'),('rng_components','rng_components')]:
        if record[key]!=reference[old]:raise ValueError('Historical initialization/RNG drift: '+key)


def paired_records(a,b,first_batch=False):
    require_initial(a);require_initial(b)
    if a['seed']!=b['seed'] or a['gap_trap_mode']==b['gap_trap_mode']:raise ValueError('Wrong paired variants/seed')
    paired(a,b,first_batch=first_batch)
