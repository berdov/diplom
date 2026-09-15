"""Vanilla Mamba3 + prototype fusion; inherited tied scorer and CE unchanged."""

from experiments.mamba3_baseline.model import Mamba3Rec
from .prototypes import PrototypeFusion


class ProtoMamba3Rec(Mamba3Rec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.prototypes = PrototypeFusion(self.hidden_size, config['prototype_k'], config['prototype_temperature'])

    def forward(self, item_seq, item_seq_len):
        h = super().forward(item_seq, item_seq_len)
        return self.prototypes(h)
