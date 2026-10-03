"""Independent copies of the same dual functions; unchanged Mamba-3 kernels."""
import copy
import torch
from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
from experiments.mamba3_three_time.mixer import three_time_forward
from experiments.mamba3_timeaware.time_inputs import history_gaps


class LayerTemporalMamba3Rec(ThreeTimeMamba3Rec):
    def __init__(self,config,dataset):
        if config['three_time_mode']!='dual':raise ValueError('Only frozen dual')
        super().__init__(config,dataset)
        self.temporal_sharing=config['temporal_sharing']
        if self.temporal_sharing not in ('shared_layers','layer_specific'):raise ValueError('Unknown layer sharing')
        if self.temporal_sharing=='layer_specific':
            # No constructor/reset and no random draws. Keep the original times
            # registration for layer0; do not register a second alias to it.
            self.layer1_times=copy.deepcopy(self.times)

    def temporal_sets(self):
        return (self.times,self.layer1_times) if self.temporal_sharing=='layer_specific' else (self.times,self.times)

    def encode_sequence(self,item_seq,item_seq_len,history_timestamps,*,oracle=None):
        if self.temporal_sharing=='shared_layers':
            return super().encode_sequence(item_seq,item_seq_len,history_timestamps,oracle=oracle)
        if oracle is not None:raise ValueError('Layer-specific oracle is outside this study')
        valid=item_seq!=0
        expected=torch.arange(item_seq.shape[1],device=item_seq.device)[None,:]<item_seq_len[:,None]
        if (item_seq_len<1).any() or not torch.equal(valid,expected):raise ValueError('Expected nonempty right-padded histories')
        gaps,active=history_gaps(history_timestamps,valid)
        scales=[times(gaps,active) for times in self.temporal_sets()]
        hidden=self.input_norm(self.input_dropout(self.item_embedding(item_seq)))
        for layer,(decay,write,phase) in zip(self.layers,scales):
            u=layer.norm1(hidden).to(layer.mixer_dtype)
            mixed=three_time_forward(layer.mixer,u,decay,write,phase)
            hidden=hidden+layer.dropout1(mixed.to(hidden.dtype))
            if layer.use_ffn:hidden=hidden+layer.dropout2(layer.ffn(layer.norm2(hidden)))
        return self.output_norm(hidden)
