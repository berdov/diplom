import ast
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mamba3_prototypes.config import load_config, assert_control_parity
from experiments.mamba3_prototypes.prototypes import PrototypeFusion
from experiments.mamba3_prototypes.random_init import random_prototypes, metadata

HERE = Path(__file__).resolve().parents[1]


def make_fusion(seed):
    torch.manual_seed(seed)
    module = PrototypeFusion()
    module.set_centroids(random_prototypes(seed))
    return module


def test_random_seed_identity_and_rng_isolation():
    first, second = make_fusion(2026), make_fusion(2026)
    assert torch.equal(first.P, second.P)
    assert not torch.equal(first.P, make_fusion(2027).P)
    state = torch.random.get_rng_state()
    random_prototypes()
    assert torch.equal(state, torch.random.get_rng_state())
    h = torch.randn(16, 64)
    assert torch.equal(first(h), h)


def test_parameter_parity():
    random = make_fusion(2026)
    kmeans = PrototypeFusion()
    kmeans.set_centroids(torch.ones(8, 64))
    assert sum(p.numel() for p in random.parameters() if p.requires_grad) == 8769
    assert [(n, p.shape, p.requires_grad) for n, p in random.named_parameters()] == [
        (n, p.shape, p.requires_grad) for n, p in kmeans.named_parameters()]
    assert sum(p.numel() for p in random.parameters()) == sum(p.numel() for p in kmeans.parameters())


def test_config_parity_and_reject_drift():
    control, reference = load_config('random'), load_config()
    assert_control_parity(control, reference)
    control['learning_rate'] *= 2
    with pytest.raises(ValueError, match='differences'):
        assert_control_parity(control, reference)


def test_random_preflight_never_uses_checkpoint_or_kmeans(monkeypatch):
    # Execute the actual preflight body with lightweight dependencies, without GPU imports.
    tree = ast.parse((HERE / 'run.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
    real_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert 'sklearn' not in name and 'prototype_init' not in name
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    pin = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'
    sha = 'e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc'
    namespace = dict(load_config=load_config, Config=lambda **kw: kw['config_dict'],
                     ProtoMamba3Rec=object, PIN=pin, json=json,
                     verify_protocol=lambda *a, **kw: {'recbole_inter_sha256': sha},
                     importlib=SimpleNamespace(metadata=SimpleNamespace(distribution=lambda name:
                         SimpleNamespace(read_text=lambda path: json.dumps({'vcs_info': {'commit_id': pin}})))))
    # No torch, CHECKPOINT or file access dependency is supplied: any such use fails.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(HERE / 'run.py'), 'exec'), namespace)
    config, protocol, saved = namespace['preflight']('random')
    assert saved is None and config['prototype_initialization'] == 'random'
    assert random_prototypes().shape == (8, 64)


def test_random_initialization_branch_and_no_test():
    source = (HERE / 'run.py').read_text()
    tree = ast.parse(source)
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "mode == 'random'"
                  and any(isinstance(x, ast.ImportFrom) and x.module == 'random_init'
                          and any(a.name == 'random_prototypes' for a in x.names) for x in n.body))
    text = '\n'.join(ast.unparse(n) for n in branch.body)
    assert all(name not in text for name in ('prototype_init', 'CHECKPOINT', 'vanilla', 'train_dataset', 'sklearn'))
    assert metadata()['test_evaluation_count'] == 0
    assert "TEST='NOT_RUN'" in source
    evaluations = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr in ('evaluate', '_valid_epoch')]
    assert all(ast.unparse(n.args[0]) == 'valid_data' for n in evaluations)
    assert "run(mode='random')" in (HERE / 'run_random.py').read_text()
