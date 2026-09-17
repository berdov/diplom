"""Frozen separate construction order, then isolated temporal replacement."""
import torch

from experiments.mamba3_time_mechanisms.model import MechanismMamba3Rec
from experiments.mamba3_time_mechanisms.time_mamba3 import mechanism_mamba3_forward
from experiments.mamba3_timeaware.time_inputs import history_gaps
from .temporal import ContextTime, MODES, COUNTS


class ContextMamba3Rec(MechanismMamba3Rec):
    def __init__(self, config, dataset):
        mode = config['context_time_mode']
        if mode not in MODES or config['time_mechanism_mode'] != 'separate':
            raise ValueError('Frozen separate constructor and context mode required')
        super().__init__(config, dataset)
        self.context_mode = mode
        if mode != 'separate_replay':
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(config['seed'])
                # Assignment unregisters the old calibrators; no unused parameters remain.
                self.mechanisms = ContextTime(mode)
        if sum(p.numel() for p in self.parameters()) != COUNTS[mode]:
            raise ValueError('Full model parameter count mismatch')

    def forward(self, item_seq, item_seq_len, history_timestamps):
        valid = item_seq != 0
        expected = torch.arange(item_seq.shape[1], device=item_seq.device)[None, :] < item_seq_len[:, None]
        if (item_seq_len < 1).any() or not torch.equal(valid, expected):
            raise ValueError('Expected nonempty right-padded histories')
        gaps, active = history_gaps(history_timestamps, valid)
        u = self.item_embedding(item_seq)
        if self.context_mode == 'separate_replay':
            decay, scan = self.mechanisms(gaps, active)
            details = {}
        else:
            decay, scan, details = self.mechanisms(u, gaps, active)
        if self.diagnostic_collector is not None:
            self.diagnostic_collector.update(decay, scan, active, details, gaps)
        hidden = self.input_norm(self.input_dropout(u))
        for layer in self.layers:
            mixed = mechanism_mamba3_forward(layer.mixer, layer.norm1(hidden).to(layer.mixer_dtype), decay, scan)
            hidden = hidden + layer.dropout1(mixed.to(hidden.dtype))
            if layer.use_ffn:
                hidden = hidden + layer.dropout2(layer.ffn(layer.norm2(hidden)))
        return self.gather_indexes(self.output_norm(hidden), item_seq_len - 1)
