"""Base-aware initialization and pairing; reuse canonical optimizer metadata."""
from . import config as c
from .records import read


def effective_check(config, mode):
    import json
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    old = read(c.PILOT)
    ref_values = dict(old['config'], use_gpu=config['use_gpu'], device=config['device'])
    ref = Config(model=ThreeTimeMamba3Rec, config_dict=ref_values)
    allowed = {'three_time_mode', 'mamba3_is_mimo', 'mamba3_chunk_size', 'mamba3_mimo_rank', 'checkpoint_dir'}
    clean = lambda d: {k: v for k, v in d.items() if k not in allowed}
    if clean(config.final_config_dict) != clean(ref.final_config_dict):
        raise ValueError('MIMO effective scientific drift')
    saved = dict(old['effective_config'])
    actual = json.loads(json.dumps(ref.final_config_dict, default=str))
    # Only technical CPU preflight may override these device values.
    ignore = allowed | ({'device', 'use_gpu'} if not config['use_gpu'] else set())
    differences = [k for k in set(saved) | set(actual) if k not in ignore and saved.get(k) != actual.get(k)]
    if differences:
        raise ValueError('Stored SISO effective flags drift: ' + repr(differences))
    if (config['three_time_mode'] != mode or not config['mamba3_is_mimo'] or config['mamba3_chunk_size'] != 8
            or config['mamba3_mimo_rank'] != 4):
        raise ValueError('Architecture/config scope')
    return dict(status='PASS', allowed_differences=sorted(allowed), stored_SISO_effective_verified=True)


def initial(model, loader=None):
    from experiments.mamba3_three_time.confirmation.state import rng_record
    from experiments.mamba3_context_time.provenance import tensor_hash
    calibrators = model.times.calibrators
    if model.times.mode == 'triple':
        pointers = [{p.untyped_storage().data_ptr() for p in calibrators[k].parameters()} for k in ('write', 'phase')]
        if pointers[0] & pointers[1]:
            raise ValueError('Aliased write/phase storage')
    return dict(initial_backbone_sha256=tensor_hash({k: v for k, v in model.state_dict().items() if not k.startswith('times.')}),
                initial_calibrator_hashes={k: tensor_hash(m.state_dict()) for k, m in calibrators.items()},
                rng_components=rng_record(loader), independent_write_phase_storage=True)


def paired(a, b, first_batch=False):
    from experiments.mamba3_three_time.confirmation.state import compare_optimizer_settings
    for key in ('initial_backbone_sha256', 'rng_components', 'protocol', 'manifest_sha256',
                'train_time_stats_sha256', 'verified_history_stats', 'precision'):
        if a.get(key) != b.get(key):
            raise ValueError('Pair mismatch: ' + key)
    if 'optimizer_settings' in a or 'optimizer_settings' in b:
        compare_optimizer_settings(a['optimizer_settings'], b['optimizer_settings'])
    ca, cb = a['initial_calibrator_hashes'], b['initial_calibrator_hashes']
    for row, cal in ((a, ca), (b, cb)):
        expected = {'base': set(), 'dual': {'decay', 'scan'}, 'triple': {'decay', 'write', 'phase'}}[row['mode']]
        if set(cal) != expected:
            raise ValueError('Calibrator keys mismatch for ' + row['mode'])
    if ca and cb:
        scan_a = ca.get('scan', ca.get('write'))
        scan_b = cb.get('scan', cb.get('write'))
        if ca['decay'] != cb['decay'] or scan_a != scan_b or cb.get('phase', scan_b) != scan_a:
            raise ValueError('Dual/triple initial calibrator mapping')
    if first_batch and a['first_train_batch_sha256'] != b['first_train_batch_sha256']:
        raise ValueError('Consumed first batch differs')


def initialization(device='cpu', save=None):
    from recbole.config import Config
    from recbole.utils import init_seed
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.config import SyntheticCatalog
    rows = []
    for mode in c.MODES:
        cfg = Config(model=ThreeTimeMamba3Rec, config_dict=c.settings(mode, device))
        parity = effective_check(cfg, mode)
        init_seed(2026, True)
        net = ThreeTimeMamba3Rec(cfg, SyntheticCatalog()).to(device)
        row = dict(mode=mode, parameter_count=sum(p.numel() for p in net.parameters()),
                   effective_config_check=parity, **initial(net))
        rows.append(row)
        if save is not None:
            save(dict(initialization_rows=rows))
        if row['parameter_count'] != c.COUNTS[mode]:
            raise ValueError('Actual parameter count mismatch')
        if sorted(n for n, _ in net.named_parameters()) != c.plan()['parameter_keys'][mode]:
            raise ValueError('Frozen parameter registry mismatch')
        if len(rows) > 1:
            paired(rows[0], row)
        del net
    paired(rows[1], rows[2])
    return dict(passed=True, rows=rows, forward_calls=0, scientific_fits=0)
