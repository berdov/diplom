"""Synthetic construction only; does not create a dataset, trainer, or logger."""

from experiments.mamba3_timeaware.config import load_config

ARCHITECTURES = ("SISO", "MIMO")
SISO_COUNTS = {"base": 610440, "dual": 610572, "triple": 610638}


class SyntheticCatalog:
    def num(self, field):
        return 7112 if field == "item_id" else 23952


def settings(architecture, mode):
    if architecture not in ARCHITECTURES or mode not in SISO_COUNTS:
        raise ValueError("Unknown architecture/mode")
    result = load_config()
    result.update(three_time_mode=mode, mamba3_is_mimo=architecture == "MIMO",
                  mamba3_mimo_rank=4, mamba3_chunk_size=8 if architecture == "MIMO" else 64)
    return result


def construct_config(architecture, mode, device):
    from recbole.config import Config
    from .model import ThreeTimeMamba3Rec
    values = settings(architecture, mode)
    values.update(use_gpu=str(device).startswith("cuda"), device=device)
    return Config(model=ThreeTimeMamba3Rec, config_dict=values)
