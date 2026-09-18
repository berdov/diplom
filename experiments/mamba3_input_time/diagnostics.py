"""VALID-only detached streaming moments and bounded independent reservoirs."""
import torch
from experiments.mamba3_time_mechanisms.diagnostics import TemporalDiagnostics


class Moments:
    def __init__(self, width, capacity=512):
        self.width, self.capacity, self.count = width, capacity, 0
        self.total = torch.zeros(width, dtype=torch.float64)
        self.squares = self.total.clone()
        self.low = torch.full_like(self.total, torch.inf)
        self.high = -self.low
        self.sample = torch.empty(0, width, dtype=torch.float64)
        self.keys = torch.empty(0)
        self.rng = torch.Generator().manual_seed(81723)

    def update(self, values):
        values = values.detach().reshape(-1, self.width)
        if not len(values):
            return
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite diagnostics')
        self.count += len(values)
        self.total += values.double().sum(0).cpu()
        self.squares += values.double().square().sum(0).cpu()
        self.low = torch.minimum(self.low, values.amin(0).double().cpu())
        self.high = torch.maximum(self.high, values.amax(0).double().cpu())
        # Keep only top random priorities on device before transferring sample values.
        keys = torch.rand(len(values), generator=self.rng)
        selected = keys.topk(min(self.capacity, len(values))).indices
        combined = torch.cat((self.sample, values[selected.to(values.device)].double().cpu()))
        keys = torch.cat((self.keys, keys[selected]))
        selected = keys.topk(min(self.capacity, len(keys))).indices
        self.keys, self.sample = keys[selected], combined[selected]

    def result(self):
        if not self.count:
            return dict(count=0)
        mean = self.total / self.count
        return dict(count=self.count, mean=mean.tolist(), std=(self.squares/self.count-mean.square()).clamp_min(0).sqrt().tolist(),
                    minimum=self.low.tolist(), maximum=self.high.tolist(), reservoir_size=len(self.sample),
                    reservoir_quantiles=torch.quantile(self.sample, torch.tensor([.05,.5,.95], dtype=torch.float64), dim=0).tolist())


class InputDiagnostics:
    def __init__(self, mode):
        self.mode = mode
        self.scales = TemporalDiagnostics('separate', 512)
        self.stats = {name: Moments(width) for name, width in (
            ('delta_norm_ratio',1), ('entropy_heads',4), ('entropy_at_least_two_keys_heads',4),
            ('self_mass_heads',4), ('past_mass_heads',4), ('position_distance_heads',4), ('time_bias_heads',4))}

    @torch.no_grad()
    def update(self, u, valid, details, decay, scan, active):
        self.scales.update(decay, scan, active)
        self.stats['delta_norm_ratio'].update((details['delta'][valid].norm(dim=-1) / u.detach()[valid].norm(dim=-1).clamp_min(1e-12))[:, None])
        if 'weights' not in details:
            return
        weights = details['weights']
        entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1).transpose(1, 2)
        self.stats['entropy_heads'].update(entropy[valid])
        self.stats['entropy_at_least_two_keys_heads'].update(entropy[valid & (details['allowed'].sum(-1) >= 2)])
        diagonal = weights.diagonal(dim1=-2, dim2=-1).transpose(1, 2)
        self.stats['self_mass_heads'].update(diagonal[valid])
        self.stats['past_mass_heads'].update((1-diagonal)[valid])
        length = valid.shape[1]
        positions = torch.arange(length, device=valid.device)
        distance = (positions[:, None]-positions[None, :]).clamp_min(0)
        self.stats['position_distance_heads'].update((weights*distance).sum(-1).transpose(1,2)[valid])
        if details['bias'] is not None:
            self.stats['time_bias_heads'].update(details['bias'].permute(0,2,3,1)[details['allowed']])

    def result(self):
        return dict(mode=self.mode, attention_heads=4, mamba_temporal_heads=2, mamba_layers=2,
                    statistics={k:v.result() for k,v in self.stats.items()}, temporal=self.scales.result(),
                    sampling='VALID streaming moments; bounded independent priority reservoirs; no extra evaluation')
