"""Reuse the complete frozen model; only the constant control changes inputs."""
from experiments.mamba3_time_mechanisms.model import MechanismMamba3Rec
from experiments.mamba3_timeaware.model import TimeAwareMamba3Rec
from .inputs import constant_timestamps


class ConstantGapMamba3Rec(MechanismMamba3Rec):
    def forward(self, item_seq, item_seq_len, history_timestamps):
        # Real timestamps are deliberately unused, including real zero-gap events.
        times = constant_timestamps(item_seq, item_seq_len)
        return super().forward(item_seq, item_seq_len, times)


def model_class(mode):
    return dict(shared=TimeAwareMamba3Rec, separate=MechanismMamba3Rec,
                separate_constant_gap=ConstantGapMamba3Rec)[mode]
