"""One input adapter; unchanged separate construction, mixer, loss and scorer."""
import torch

from experiments.mamba3_time_mechanisms.model import MechanismMamba3Rec
from experiments.mamba3_time_mechanisms.time_mamba3 import mechanism_mamba3_forward
from experiments.mamba3_timeaware.time_inputs import history_gaps
from .adapter import InputAdapter, COUNTS


class InputMamba3Rec(MechanismMamba3Rec):
    def __init__(self, config, dataset):
        if config['time_mechanism_mode'] != 'separate':
            raise ValueError('Frozen separate required')
        super().__init__(config, dataset)
        self.input_mode = config['input_time_mode']
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            # CPU module initialization only: do not alter deferred CUDA seed callbacks on login.
            torch.random.default_generator.manual_seed(config['seed'])
            self.input_adapter = InputAdapter(self.input_mode)
        if sum(p.numel() for p in self.parameters()) != COUNTS[self.input_mode]:
            raise ValueError('Full model parameter count mismatch')

    def forward(self, item_seq, item_seq_len, history_timestamps):
        return self.encode_embeddings(self.item_embedding(item_seq), item_seq, item_seq_len, history_timestamps)

    def encode_embeddings(self, u, item_seq, item_seq_len, history_timestamps):
        """Explicit embedding entry point also permits synthetic input-gradient checks."""
        valid = item_seq != 0
        expected = torch.arange(item_seq.shape[1], device=item_seq.device)[None, :] < item_seq_len[:, None]
        if (item_seq.dtype != torch.int64 or u.dtype != torch.float32
                or (item_seq_len < 1).any() or not torch.equal(valid, expected)):
            raise ValueError('Expected int64 IDs, fp32 embeddings and nonempty right padding')
        gaps, active = history_gaps(history_timestamps, valid)
        decay, scan = self.mechanisms(gaps, active)
        if self.diagnostic_collector is None:
            adapted = self.input_adapter(u, history_timestamps, valid)
        else:
            adapted, details = self.input_adapter(u, history_timestamps, valid, details=True)
            self.diagnostic_collector.update(u, valid, details, decay, scan, active)
        hidden = self.input_norm(self.input_dropout(adapted))
        for layer in self.layers:
            mixed = mechanism_mamba3_forward(layer.mixer, layer.norm1(hidden).to(layer.mixer_dtype), decay, scan)
            hidden = hidden + layer.dropout1(mixed.to(hidden.dtype))
            if layer.use_ffn:
                hidden = hidden + layer.dropout2(layer.ffn(layer.norm2(hidden)))
        return self.gather_indexes(self.output_norm(hidden), item_seq_len - 1)
