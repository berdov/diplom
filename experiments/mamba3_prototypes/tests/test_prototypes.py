import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.mamba3_prototypes.prototypes import PrototypeFusion
from experiments.mamba3_prototypes.diagnostics import PrototypeDiagnostics


def test_shapes_and_soft_assignment():
    module = PrototypeFusion()
    h = torch.randn(11, 64)
    logits, alpha, z, gate = module.components(h)
    assert module.P.shape == (8, 64)
    assert alpha.shape == (11, 8) and z.shape == gate.shape == (11, 64)
    assert module(h).shape == h.shape
    assert torch.isfinite(logits).all() and torch.isfinite(alpha).all()
    assert (alpha >= 0).all()
    torch.testing.assert_close(alpha.sum(-1), torch.ones(11))


@pytest.mark.parametrize('zero_input', [False, True])
def test_exact_identity(zero_input):
    module = PrototypeFusion()
    module.set_centroids(torch.randn(8, 64))
    h = torch.zeros(4, 64) if zero_input else torch.randn(4, 64)
    assert module.residual_strength.item() == 0
    assert torch.equal(module(h), h)


def test_nonzero_strength_gradients_and_learning():
    torch.manual_seed(2026)
    backbone = nn.Linear(16, 64)
    module = PrototypeFusion()
    with torch.no_grad():
        module.residual_strength.fill_(0.2)
    optimizer = torch.optim.Adam([*backbone.parameters(), *module.parameters()], lr=.001)
    before = module.P.detach().clone()
    loss = module(backbone(torch.randn(9, 16))).square().mean()
    loss.backward()
    for parameter in [*backbone.parameters(), *module.parameters()]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(module.P, before)


def test_identity_init_learning_delayed_one_step():
    torch.manual_seed(2026)
    module = PrototypeFusion()
    h = torch.randn(16, 64)
    optimizer = torch.optim.Adam(module.parameters(), lr=.001)
    before = module.P.detach().clone()
    for step in range(2):
        optimizer.zero_grad()
        module(h).square().mean().backward()
        if step == 0:
            assert module.P.grad.abs().sum() == 0
            assert module.gate.weight.grad.abs().sum() == 0
            assert module.residual_strength.grad.abs() > 0
        optimizer.step()
    assert not torch.equal(module.P, before)


def test_diagnostics_do_not_change_forward():
    module = PrototypeFusion()
    h = torch.randn(12, 64)
    expected = module(h)
    collector = PrototypeDiagnostics()
    module.diagnostic_collector = collector
    assert torch.equal(module(h), expected)
    result = collector.result(module)
    assert result['examples'] == 12
    assert sum(result['argmax_usage_share']) == pytest.approx(1)
    assert sum(result['mean_assignment_probability']) == pytest.approx(1)
    assert 0 <= result['mean_gate_activation'] <= 1
    assert result['assignment_entropy_std'] >= 0


def test_no_hard_assignment():
    path = Path(__file__).resolve().parents[1] / 'prototypes.py'
    names = [node.attr for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Attribute)]
    assert not {'argmax', 'one_hot'}.intersection(names)


@pytest.mark.parametrize('values', [torch.zeros(7, 64), torch.full((8, 64), float('nan'))])
def test_invalid_centroids_rejected(values):
    with pytest.raises(ValueError):
        PrototypeFusion().set_centroids(values)
