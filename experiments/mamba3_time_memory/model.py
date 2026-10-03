"""A within-window readout extension of the unchanged shared dual encoder."""
import torch
from torch import nn

from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
from .memory import MEMORY_MODES, read_memory, select


class TimeMemoryMamba3Rec(ThreeTimeMamba3Rec):
    def __init__(self, config, dataset):
        if config['three_time_mode'] != 'dual':
            raise ValueError('Only the frozen MIMO dual backbone is supported')
        mode = config['memory_mode']
        if mode not in ('no_memory', *MEMORY_MODES):
            raise ValueError('Unknown memory mode')
        super().__init__(config, dataset)
        self.memory_mode = mode
        if mode != 'no_memory':
            # zeros, not a random initializer: preserve all construction RNG.
            self.beta = nn.Parameter(torch.zeros((), dtype=torch.float32))
        # Optional ephemeral VALID callback, never a state tensor or cache.
        self.memory_observer = None

    def encode_sequence(self, item_seq, item_seq_len, history_timestamps, *, oracle=None):
        if self.memory_mode == 'no_memory':
            return super().encode_sequence(item_seq, item_seq_len, history_timestamps, oracle=oracle)
        # Exactly one existing encoder pass, including its original output_norm.
        hidden = super().encode_sequence(item_seq, item_seq_len, history_timestamps, oracle=oracle)
        valid = item_seq != 0
        selection = select(history_timestamps, valid, self.memory_mode)
        output, details = read_memory(hidden, selection, self.beta)
        if self.memory_observer is not None:
            with torch.no_grad():
                self.memory_observer(dict(
                    H=hidden.detach(), output=output.detach(),
                    selection={k: v.detach() if torch.is_tensor(v) else v for k, v in selection.items()},
                    reader={k: v.detach() if torch.is_tensor(v) else v for k, v in details.items()},
                    beta=self.beta.detach(), **{'lambda': self.beta.detach().tanh()},
                    valid=valid.detach(), timestamps=history_timestamps.detach(), mode=self.memory_mode))
        return output


def transfer_common(source, target):
    """Copy exactly shared keys, also accepting historical models without a mode.

The only optional state difference is this experiment's scalar beta. All
unrelated missing, extra, shape or dtype mismatches are rejected explicitly.
"""
    source_state, target_state = source.state_dict(), target.state_dict()
    source_keys = set(source_state) - {'beta'}
    target_keys = set(target_state) - {'beta'}
    if source_keys != target_keys:
        raise ValueError(f'Unexpected common state keys: {sorted(source_keys ^ target_keys)}')
    for state in (source_state, target_state):
        if 'beta' in state and (state['beta'].shape != torch.Size([]) or state['beta'].dtype != torch.float32):
            raise ValueError('Unexpected beta state')
    for key in source_keys:
        if source_state[key].shape != target_state[key].shape or source_state[key].dtype != target_state[key].dtype:
            raise ValueError(f'Common state mismatch: {key}')
        target_state[key] = source_state[key].detach().clone()
    target.load_state_dict(target_state, strict=True)
    if any(not torch.equal(value, target.state_dict()[key]) for key, value in source_state.items() if key != 'beta'):
        raise ValueError('Common state values did not transfer exactly')
    return dict(mapping='Identical common keys; beta excluded and target initialization retained',
                keys=sorted(source_keys), all_keys_shapes_dtypes_values_equal=True,
                source_has_beta='beta' in source_state, target_has_beta='beta' in target_state)
