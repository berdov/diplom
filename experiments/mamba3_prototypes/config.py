from pathlib import Path
import yaml


def load_config(mode='kmeans'):
    here = Path(__file__).resolve().parent
    base = yaml.safe_load((here.parent / 'mamba3_baseline/config_kuairand.yaml').read_text())
    overlay = yaml.safe_load((here / 'config_kuairand.yaml').read_text())
    settings = {**base, **overlay, 'prototype_initialization': 'kmeans'}
    if mode == 'random':
        control = yaml.safe_load((here / 'config_random.yaml').read_text())
        settings.update(control)
        assert_control_parity(settings, {**base, **overlay, 'prototype_initialization': 'kmeans'})
    elif mode != 'kmeans':
        raise ValueError('Unknown prototype initialization')
    return settings


def assert_control_parity(control, reference):
    allowed = {'prototype_initialization', 'prototype_random_init_std', 'checkpoint_dir'}
    differences = {key for key in control.keys() | reference.keys()
                   if control.get(key) != reference.get(key)}
    if differences != allowed:
        raise ValueError(f'Unexpected control differences: {differences}')
    if control['prototype_initialization'] != 'random' or control['prototype_random_init_std'] != .02:
        raise ValueError('Random control fixes Normal(0, .02)')
