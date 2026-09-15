"""Detached diagnostics only; no routing decisions or regularization."""

import torch
from torch.nn import functional as F


class PrototypeDiagnostics:
    def __init__(self):
        self.count = 0
        self.probability = torch.zeros(8, dtype=torch.float64)
        self.usage = torch.zeros(8, dtype=torch.float64)
        self.entropy_sum = self.entropy_square = self.gate_sum = 0.0
        self.gate_count = 0

    @torch.no_grad()
    def update(self, alpha, gate):
        alpha = alpha.detach().double()
        entropy = -(alpha * alpha.clamp_min(1e-30).log()).sum(-1)
        self.count += len(alpha)
        self.probability += alpha.sum(0).cpu()
        self.usage += torch.bincount(alpha.argmax(-1), minlength=8).cpu()
        self.entropy_sum += entropy.sum().item()
        self.entropy_square += entropy.square().sum().item()
        self.gate_sum += gate.detach().double().sum().item()
        self.gate_count += gate.numel()

    @torch.no_grad()
    def result(self, module):
        if not self.count:
            raise ValueError('Empty diagnostic population')
        p = F.normalize(module.P.detach().double(), dim=-1)
        similarity = p @ p.T
        off_diagonal = similarity[~torch.eye(8, dtype=torch.bool, device=p.device)]
        mean = self.entropy_sum / self.count
        return dict(examples=self.count,
                    mean_assignment_probability=(self.probability / self.count).tolist(),
                    argmax_usage_share=(self.usage / self.count).tolist(),
                    assignment_entropy_mean=mean,
                    assignment_entropy_std=max(0., self.entropy_square / self.count - mean * mean) ** .5,
                    prototype_cosine_mean_off_diagonal=off_diagonal.mean().item(),
                    prototype_cosine_max_off_diagonal=off_diagonal.max().item(),
                    residual_strength=module.residual_strength.item(),
                    effective_residual_strength=module.residual_strength.tanh().item(),
                    mean_gate_activation=self.gate_sum / self.gate_count)
