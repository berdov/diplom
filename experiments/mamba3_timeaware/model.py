"""History-only DT conditioning; frozen recommender components are reused."""

import torch

from experiments.mamba3_baseline.model import Mamba3Rec
from .time_inputs import TimeCalibrator, history_gaps
from .time_mamba3 import time_mamba3_forward


class TimeAwareMamba3Rec(Mamba3Rec):
    def __init__(self, config, dataset):
        if config["time_scale_reference_source"] != "TRAIN":
            raise ValueError("Time reference must come from TRAIN only")
        if config["mamba3_is_mimo"] or config["num_layers"] != 2:
            raise ValueError("This experiment requires two SISO layers")
        super().__init__(config, dataset)
        self.time_sequence_field = config["TIME_FIELD"] + config["LIST_SUFFIX"]
        self.time_calibrator = TimeCalibrator(
            self.layers[0].mixer.nheads,
            config["time_scale_reference"], config["max_log_scale"],
        )
        if any(layer.mixer.nheads != self.time_calibrator.n_heads for layer in self.layers):
            raise ValueError("All layers must use the shared head count")

    def forward(self, item_seq, item_seq_len, history_timestamps):
        valid = item_seq != 0
        expected = torch.arange(item_seq.shape[1], device=item_seq.device)[None, :] < item_seq_len[:, None]
        if (item_seq_len < 1).any() or not torch.equal(valid, expected):
            raise ValueError("Expected nonempty, right-padded item histories")
        gaps, active = history_gaps(history_timestamps, valid)
        scale = self.time_calibrator(gaps, active)
        hidden = self.input_norm(self.input_dropout(self.item_embedding(item_seq)))
        for layer in self.layers:
            mixed = time_mamba3_forward(layer.mixer, layer.norm1(hidden).to(layer.mixer_dtype), scale)
            hidden = hidden + layer.dropout1(mixed.to(hidden.dtype))
            if layer.use_ffn:
                hidden = hidden + layer.dropout2(layer.ffn(layer.norm2(hidden)))
        return self.gather_indexes(self.output_norm(hidden), item_seq_len - 1)

    def _encode(self, interaction):
        return self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN],
                            interaction[self.time_sequence_field])

    def calculate_loss(self, interaction):
        logits = self._encode(interaction) @ self.item_embedding.weight.T
        return self.loss_fct(logits, interaction[self.POS_ITEM_ID])

    def predict(self, interaction):
        return (self._encode(interaction) * self.item_embedding(interaction[self.ITEM_ID])).sum(-1)

    def full_sort_predict(self, interaction):
        return self._encode(interaction) @ self.item_embedding.weight.T
