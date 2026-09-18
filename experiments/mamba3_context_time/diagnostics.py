"""Detached streaming diagnostics; bounded samples and an independent RNG."""
import torch
from experiments.mamba3_time_mechanisms.diagnostics import TemporalDiagnostics


class ContextDiagnostics:
    def __init__(self, mode, capacity=2048):
        self.mode, self.capacity = mode, capacity
        self.scales = TemporalDiagnostics('separate', capacity)
        self.count = 0
        self.saturation = torch.zeros(2, 2, dtype=torch.float64)
        self.pi_sum = torch.zeros(4, dtype=torch.float64)
        self.argmax = torch.zeros(4, dtype=torch.float64)
        self.entropy = 0.0
        self.disagreement = torch.zeros(2, 2, dtype=torch.float64)
        self.rng = torch.Generator().manual_seed(81723)
        self.keys = torch.empty(0)
        self.sample = torch.empty(0, 5, dtype=torch.float64)

    @torch.no_grad()
    def update(self, decay, scan, active, details, gaps):
        self.scales.update(decay, scan, active)
        n = int(active.sum())
        if not n:
            return
        self.count += n
        values = torch.stack((decay[active], scan[active]), 1).detach()
        self.saturation += ((values < .51) | (values > 1.99)).double().sum(0).cpu()
        if 'log_experts' in details:
            q = details['log_experts'].detach()[active]
            self.disagreement += q.std(dim=1, unbiased=False).double().sum(0).cpu()
        if self.mode != 'routed':
            return
        pi = details['probabilities'].detach()[active].double().cpu()
        self.pi_sum += pi.sum(0)
        self.argmax += torch.bincount(pi.argmax(-1), minlength=4)
        self.entropy += float(-(pi * pi.clamp_min(1e-300).log()).sum())
        sample = torch.cat((gaps.detach()[active].double().cpu()[:, None], pi), -1)
        keys = torch.cat((self.keys, torch.rand(n, generator=self.rng)))
        sample = torch.cat((self.sample, sample))
        indices = keys.topk(min(self.capacity, len(keys))).indices
        self.keys, self.sample = keys[indices], sample[indices]

    def result(self):
        result = self.scales.result()
        if not self.count:
            return result
        result['near_bound_fractions_paths_heads'] = (self.saturation / self.count).tolist()
        if self.mode in ('routed', 'uniform'):
            result['mean_expert_log_scale_std_paths_heads'] = (self.disagreement / self.count).tolist()
        if self.mode == 'uniform':
            result['routing'] = {'kind': 'fixed_uniform', 'probabilities': [.25] * 4, 'learned': False}
        elif self.mode == 'routed':
            gaps, pi = self.sample[:, 0], self.sample[:, 1:]
            repeated = []
            for gap in gaps.unique():
                group = pi[gaps == gap]
                if len(group) > 1:
                    repeated.append(float(group.var(0, unbiased=False).mean()))
            result['routing'] = dict(kind='learned_dense', mean_probabilities=(self.pi_sum/self.count).tolist(),
                entropy=self.entropy/self.count, argmax_fractions=(self.argmax/self.count).tolist(),
                reservoir_size=len(pi), across_gaps_probability_variance=pi.var(0, unbiased=False).tolist(),
                repeated_gap_groups=len(repeated), same_gap_mean_probability_variance=(sum(repeated)/len(repeated) if repeated else None),
                sampling='bounded uniform reservoir, descriptive only; no additional evaluation')
        return result
