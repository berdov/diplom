"""Load a small overlay without changing frozen Protocol B settings."""

from pathlib import Path
import yaml


def load_config():
    here = Path(__file__).resolve().parent
    base = yaml.safe_load((here.parent / "mamba3_baseline/config_kuairand.yaml").read_text())
    overlay = yaml.safe_load((here / "config_kuairand.yaml").read_text())
    return {**base, **overlay}
