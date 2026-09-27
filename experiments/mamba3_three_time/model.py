"""Frozen outer recommender, shared per-head temporal functions across layers."""

import torch
from experiments.mamba3_baseline.model import Mamba3Rec
from experiments.mamba3_timeaware.model import TimeAwareMamba3Rec
from experiments.mamba3_timeaware.time_inputs import history_gaps
from .calibrators import ThreeTimes
from .mixer import three_time_forward


class ThreeTimeMamba3Rec(TimeAwareMamba3Rec):
    def __init__(self, config, dataset):
        Mamba3Rec.__init__(self, config, dataset)
        self.time_sequence_field = config["TIME_FIELD"] + config["LIST_SUFFIX"]
        self.times = ThreeTimes(config["three_time_mode"], self.layers[0].mixer.nheads)
        if self.num_layers != 2 or self.is_outproj_norm:
            raise ValueError("Frozen two-layer backbone required")

    def encode_sequence(self, item_seq, item_seq_len, history_timestamps, *, oracle=None):
        valid = item_seq != 0
        expected = torch.arange(item_seq.shape[1], device=item_seq.device)[None, :] < item_seq_len[:, None]
        if (item_seq_len < 1).any() or not torch.equal(valid, expected):
            raise ValueError("Expected nonempty right-padded histories")
        gaps, active = history_gaps(history_timestamps, valid)
        decay, write, phase = self.times(gaps, active)
        hidden = self.input_norm(self.input_dropout(self.item_embedding(item_seq)))
        for layer in self.layers:
            u = layer.norm1(hidden).to(layer.mixer_dtype)
            if oracle == "official_base":
                if self.times.mode != "base":
                    raise ValueError("Base oracle only")
                if layer.mixer.is_mimo:
                    from .length_adapter import official_mixer
                    mixed = official_mixer(layer.mixer, u)
                else:
                    mixed = layer.mixer(u)
            else:
                mixed = three_time_forward(layer.mixer, u, decay, write, phase,
                                           official_tied=oracle == "official_dual")
            hidden = hidden + layer.dropout1(mixed.to(hidden.dtype))
            if layer.use_ffn:
                hidden = hidden + layer.dropout2(layer.ffn(layer.norm2(hidden)))
        return self.output_norm(hidden)

    def forward(self, item_seq, item_seq_len, history_timestamps):
        return self.gather_indexes(self.encode_sequence(item_seq, item_seq_len, history_timestamps), item_seq_len - 1)

    def step(self, *args, **kwargs):
        raise NotImplementedError("External step/cache is outside this study")
