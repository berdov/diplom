"""Shared periodic phase module around the unchanged MIMO-dual backbone."""

import torch

from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
from experiments.mamba3_timeaware.time_inputs import history_gaps
from .phase import MODES, PeriodicPhase
from .mixer import phase_forward


class AbsolutePhaseMamba3Rec(ThreeTimeMamba3Rec):
    def __init__(self, config, dataset):
        if config["three_time_mode"] != "dual":
            raise ValueError("Absolute phase experiment requires unchanged dual calibrators")
        super().__init__(config, dataset)
        self.phase_mode = config["phase_mode"]
        if self.phase_mode not in MODES:
            raise ValueError("Unknown phase mode")
        angles = {layer.mixer.num_rope_angles for layer in self.layers}
        if len(angles) != 1 or any(layer.mixer.nheads != 2 for layer in self.layers):
            raise ValueError("One shared phase matrix requires equal angle counts and two heads")
        self.phase_adapter = (None if self.phase_mode == "baseline_dual"
                              else PeriodicPhase(angles.pop(), self.phase_mode))
        self.phase_observer = None

    def encode_sequence(self, item_seq, item_seq_len, history_timestamps,
                        *, oracle=None, observer=None):
        # Exactly the previous implementation, with no new projection or state.
        if self.phase_mode == "baseline_dual" and oracle != "reference" and observer is None:
            return super().encode_sequence(item_seq, item_seq_len, history_timestamps, oracle=oracle)
        if oracle not in (None, "reference"):
            raise ValueError("Modified phase modes support only explicit recurrence oracle")
        valid = item_seq != 0
        expected = torch.arange(item_seq.shape[1], device=item_seq.device)[None, :] < item_seq_len[:, None]
        if (item_seq_len < 1).any() or not torch.equal(valid, expected):
            raise ValueError("Expected nonempty right-padded histories")
        gaps, active = history_gaps(history_timestamps, valid)
        decay, write, phase = self.times(gaps, active)
        correction, phi = None, None
        if self.phase_adapter is not None:
            correction, phi, active = self.phase_adapter.components(
                history_timestamps, valid, gaps=gaps, active=active)
        observe = observer if observer is not None else self.phase_observer
        metadata = dict(phase_features=phi, phase_active=active)
        hidden = self.input_norm(self.input_dropout(self.item_embedding(item_seq)))
        for index, layer in enumerate(self.layers):
            u = layer.norm1(hidden).to(layer.mixer_dtype)
            callback = None if observe is None else lambda values, i=index: observe(i, values)
            mixed = phase_forward(layer.mixer, u, decay, write, phase, correction,
                                  reference=oracle == "reference", observer=callback, metadata=metadata)
            hidden = hidden + layer.dropout1(mixed.to(hidden.dtype))
            if layer.use_ffn:
                hidden = hidden + layer.dropout2(layer.ffn(layer.norm2(hidden)))
        return self.output_norm(hidden)


def transfer_common(source, target):
    """Copy only exactly matching common state; reject hidden mappings/drift."""
    source_state, target_state = source.state_dict(), target.state_dict()
    exceptional = {"phase_adapter.W"}
    source_keys, target_keys = set(source_state) - exceptional, set(target_state) - exceptional
    if source_keys != target_keys:
        raise ValueError("Common state keys differ: " + repr(sorted(source_keys ^ target_keys)))
    for key in sorted(source_keys):
        old, new = source_state[key], target_state[key]
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError("Common state shape/dtype differs: " + key)
    target.load_state_dict({key: source_state[key] if key in source_keys else value
                            for key, value in target_state.items()}, strict=True)
    return sorted(source_keys)
