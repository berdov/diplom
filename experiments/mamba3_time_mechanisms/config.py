from pathlib import Path
import yaml
from experiments.mamba3_timeaware.config import load_config as frozen_config
from .time_mechanisms import MODES


def load_config(mode=None):
    settings = frozen_config()
    overlay = yaml.safe_load(Path(__file__).with_name('config_kuairand.yaml').read_text())
    settings.update(overlay)
    if mode is not None:
        settings['time_mechanism_mode'] = mode
    if settings['time_mechanism_mode'] not in MODES:
        raise ValueError('Unknown time mechanism mode')
    return settings
