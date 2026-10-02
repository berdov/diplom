"""Fixed-reference dual model with an optional isolated Trap logit shift."""
import torch
from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
from experiments.mamba3_timeaware.time_inputs import history_gaps
from .modulation import GapTrap
from .mixer import gap_trap_forward


class GapTrapMamba3Rec(ThreeTimeMamba3Rec):
    def __init__(self, config, dataset):
        if config['three_time_mode'] != 'dual':
            raise ValueError('Fixed-reference dual only')
        super().__init__(config, dataset)
        self.gap_trap_mode = config['gap_trap_mode']
        if self.gap_trap_mode not in ('fixed_replay', 'gap_trap'):
            raise ValueError('Unknown gap_trap_mode')
        if self.gap_trap_mode == 'gap_trap':
            self.gap_trap = GapTrap()

    def encode_sequence(self, item_seq, item_seq_len, history_timestamps, *, oracle=None):
        if self.gap_trap_mode == 'fixed_replay':
            return super().encode_sequence(item_seq, item_seq_len, history_timestamps, oracle=oracle)
        if oracle is not None:
            raise ValueError('Use the separate unchanged model as oracle')
        valid = item_seq != 0
        expected = torch.arange(item_seq.shape[1], device=item_seq.device)[None, :] < item_seq_len[:, None]
        if (item_seq_len < 1).any() or not torch.equal(valid, expected):
            raise ValueError('Expected nonempty right-padded histories')
        gaps, active = history_gaps(history_timestamps, valid)
        decay, write, phase = self.times(gaps, active)
        shift = self.gap_trap(gaps, active)
        hidden = self.input_norm(self.input_dropout(self.item_embedding(item_seq)))
        for layer in self.layers:
            u = layer.norm1(hidden).to(layer.mixer_dtype)
            mixed = gap_trap_forward(layer.mixer, u, decay, write, phase, shift)
            hidden = hidden + layer.dropout1(mixed.to(hidden.dtype))
            if layer.use_ffn:
                hidden = hidden + layer.dropout2(layer.ffn(layer.norm2(hidden)))
        return self.output_norm(hidden)
