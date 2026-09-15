"""Frozen backbone and inherited RT scorer/CE; explicit temporal paths only."""

import torch
from experiments.mamba3_baseline.model import Mamba3Rec
from experiments.mamba3_timeaware.model import TimeAwareMamba3Rec
from experiments.mamba3_timeaware.time_inputs import history_gaps
from .time_mechanisms import TimeMechanisms, COUNTS
from .time_mamba3 import mechanism_mamba3_forward


class MechanismMamba3Rec(TimeAwareMamba3Rec):
    def __init__(self, config, dataset):
        if config['time_scale_reference_source'] != 'TRAIN' or config['mamba3_is_mimo'] or config['num_layers'] != 2:
            raise ValueError('Frozen TRAIN reference and two SISO layers required')
        # Bypass RT initialization to avoid registering an unused third calibrator.
        Mamba3Rec.__init__(self, config, dataset)
        self.time_sequence_field = config['TIME_FIELD'] + config['LIST_SUFFIX']
        self.mechanisms = TimeMechanisms(config['time_mechanism_mode'], self.layers[0].mixer.nheads,
                                         config['time_scale_reference'], config['max_log_scale'])
        self.diagnostic_collector = None
        if any(layer.mixer.nheads != self.mechanisms.n_heads for layer in self.layers):
            raise ValueError('Shared head count required')
        count = sum(p.numel() for p in self.parameters())
        if count != COUNTS[self.mechanisms.mode]:
            raise ValueError(f'Canonical parameter count mismatch: {count}')

    def forward(self, item_seq, item_seq_len, history_timestamps):
        valid = item_seq != 0
        expected = torch.arange(item_seq.shape[1], device=item_seq.device)[None, :] < item_seq_len[:, None]
        if (item_seq_len < 1).any() or not torch.equal(valid, expected):
            raise ValueError('Expected nonempty, right-padded item histories')
        gaps, active = history_gaps(history_timestamps, valid)
        decay, scan = self.mechanisms(gaps, active)
        if self.diagnostic_collector is not None:
            self.diagnostic_collector.update(decay, scan, active)
        hidden = self.input_norm(self.input_dropout(self.item_embedding(item_seq)))
        for layer in self.layers:
            mixed = mechanism_mamba3_forward(layer.mixer, layer.norm1(hidden).to(layer.mixer_dtype), decay, scan)
            hidden = hidden + layer.dropout1(mixed.to(hidden.dtype))
            if layer.use_ffn:
                hidden = hidden + layer.dropout2(layer.ffn(layer.norm2(hidden)))
        return self.gather_indexes(self.output_norm(hidden), item_seq_len - 1)
