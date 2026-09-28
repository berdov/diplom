"""Parameterized config checks and read-only initialization snapshots."""
import json
from . import config as c
from .records import read
from experiments.mamba3_mimo_time.state import initial, paired


def effective_check(config, mode, seed, observe=None):
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    old = read(c.pilot_path(mode))
    cpu = config['use_gpu'] is False
    values = dict(old['config'])
    if cpu:
        values.update(device='cpu',use_gpu=False,gpu_id='')
    ref = Config(model=ThreeTimeMamba3Rec,config_dict=values)
    if observe:
        observe('reference_config',mode,ref)
    actual = json.loads(json.dumps(config.final_config_dict,default=str))
    reference = json.loads(json.dumps(ref.final_config_dict,default=str))
    ignore = {'seed','checkpoint_dir'} | ({'device','use_gpu','gpu_id'} if cpu else set())
    for other in (reference,old['effective_config']):
        changed = [k for k in set(actual)|set(other) if k not in ignore and actual.get(k) != other.get(k)]
        if changed:
            raise ValueError('Frozen effective config drift: '+repr(changed))
    if config['seed'] != seed or config['three_time_mode'] != mode:
        raise ValueError('Seed/mode drift')
    if cpu:
        for cfg in (config,ref):
            if str(cfg['device']) != 'cpu' or cfg['use_gpu'] is not False or cfg['gpu_id'] != '':
                raise ValueError('CPU Config reopened GPU')
    return dict(status='PASS',allowed_differences=sorted(ignore),reference='successful MIMO pilot effective_config')


def initialization(checkpoint_dir, observe):
    import torch
    from recbole.config import Config
    from recbole.utils import init_seed
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.config import SyntheticCatalog
    result = []
    old_rows = read(c.PILOT_ROOT/'slurm_logs/attempt_003/preflight/evidence.json')['initialization']['rows']
    for seed in (2026,*c.SEEDS):
        rows = []
        for mode in c.MODES:
            cfg = Config(model=ThreeTimeMamba3Rec,config_dict=c.settings(mode,seed,'cpu',checkpoint_dir))
            observe('checked_config',mode,cfg)
            parity = effective_check(cfg,mode,seed,observe)
            init_seed(cfg['seed']+cfg['local_rank'],cfg['reproducibility'])
            net = ThreeTimeMamba3Rec(cfg,SyntheticCatalog()).to('cpu')
            observe('cpu_model_constructed',mode,cfg)
            if any(t.device.type != 'cpu' for t in [*net.parameters(),*net.buffers()]):
                raise ValueError('Off-CPU tensor')
            row = dict(mode=mode,seed=seed,parameter_count=sum(p.numel() for p in net.parameters()),
                       effective_config_check=parity,**initial(net))
            if row['parameter_count'] != c.COUNTS[mode]:
                raise ValueError('Parameter count drift')
            if seed == 2026:
                pilot = read(c.pilot_path(mode))
                cpu_old = next(r for r in old_rows if r['mode']==mode)
                for k in ('initial_backbone_sha256','initial_calibrator_hashes'):
                    if row[k] != pilot[k] or row[k] != cpu_old[k]:
                        raise ValueError('Seed2026 initialization replay mismatch: '+k)
                for k in ('python','numpy','cpu'):
                    if row['rng_components'][k] != pilot['rng_components'][k] or row['rng_components'][k] != cpu_old['rng_components'][k]:
                        raise ValueError('Seed2026 RNG replay mismatch: '+k)
                row['pilot_replay'] = dict(backbone='PASS',calibrators='PASS',python_numpy_cpu='PASS',
                                          cuda='NOT_REPLAYED_CPU',loader_generator='NOT_REPLAYED_CPU',first_batch='NOT_REPLAYED_NO_FIT')
            rows.append(row)
            del net
        for a,b in ((0,1),(0,2),(1,2)):
            paired(rows[a],rows[b])
        result.extend(rows)
    if torch.cuda.is_initialized():
        raise ValueError('CPU initialization initialized CUDA')
    return dict(status='PASS',rows=result,model_forward_calls=0,scientific_fits=0)
