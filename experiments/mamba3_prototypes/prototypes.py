"""Soft cosine prototypes with an exactly identity-initialized residual."""

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeFusion(nn.Module):
    def __init__(self, hidden_size=64, k=8, temperature=1.0):
        super().__init__()
        if (hidden_size, k, temperature) != (64, 8, 1.0):
            raise ValueError('v1 fixes hidden_size=64, K=8, temperature=1.0')
        self.P = nn.Parameter(torch.empty(k, hidden_size))
        nn.init.normal_(self.P, std=0.02)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.residual_strength = nn.Parameter(torch.zeros(()))
        self.temperature = temperature
        self.diagnostic_collector = None

    def components(self, h):
        logits = F.normalize(h, dim=-1) @ F.normalize(self.P, dim=-1).T / self.temperature
        alpha = torch.softmax(logits, dim=-1)
        z = alpha @ self.P
        gate = torch.sigmoid(self.gate(torch.cat((h, z), dim=-1)))
        return logits, alpha, z, gate

    def forward(self, h):
        _, alpha, z, gate = self.components(h)
        if self.diagnostic_collector is not None:
            self.diagnostic_collector.update(alpha, gate)
        return h + torch.tanh(self.residual_strength) * gate * z

    def set_centroids(self, centroids):
        values = torch.as_tensor(centroids, device=self.P.device, dtype=self.P.dtype)
        if values.shape != self.P.shape or not torch.isfinite(values).all():
            raise ValueError('Expected finite centroids [8,64]')
        with torch.no_grad():
            self.P.copy_(values)
