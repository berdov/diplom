import ast
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.mamba3_prototypes.config import load_config
from experiments.mamba3_prototypes.prototype_init import fit_stream


HERE = Path(__file__).resolve().parents[1]


def test_config_preserves_vanilla():
    vanilla = yaml.safe_load((HERE.parent / 'mamba3_baseline/config_kuairand.yaml').read_text())
    config = load_config()
    for key in vanilla:
        if key != 'checkpoint_dir':
            assert config[key] == vanilla[key]
    assert config['prototype_k'] == 8 and config['prototype_temperature'] == 1.0


def test_model_api_and_scorer_unchanged():
    tree = ast.parse((HERE / 'model.py').read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert ast.unparse(model.bases[0]) == 'Mamba3Rec'
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    assert [a.arg for a in forward.args.args] == ['self', 'item_seq', 'item_seq_len']
    assert {node.name for node in model.body if isinstance(node, ast.FunctionDef)} == {'__init__', 'forward'}


def test_runner_no_test_path_and_smoke_before_fit():
    text = (HERE / 'run.py').read_text()
    tree = ast.parse(text)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Name) and node.func.id == 'data_preparation' for node in calls)
    evaluations = [node for node in calls if isinstance(node.func, ast.Attribute)
                   and node.func.attr in ('evaluate', '_valid_epoch')]
    assert all(ast.unparse(node.args[0]) == 'valid_data' for node in evaluations)
    assert text.index('smoke_result = smoke(') < text.index('score, metrics = trainer.fit(')
    assert "test_evaluation_count=0" in text and "TEST='NOT_RUN'" in text
    assert 'mamba3_timeaware' not in text


def test_streaming_kmeans_deterministic():
    rng = np.random.default_rng(2026)
    batches = [rng.normal(size=(32, 64)).astype(np.float32) for _ in range(3)]
    first, counts = fit_stream(iter(batches))
    second, _ = fit_stream(iter(batches))
    assert first.shape == (8, 64) and np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert counts['train_histories'] == 96 and counts['batches'] == 3


def test_empty_stream_fails():
    with pytest.raises(ValueError):
        fit_stream(iter(()))


def test_init_encoder_has_only_history_inputs():
    tree = ast.parse((HERE / 'prototype_init.py').read_text())
    encoder = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == 'vanilla')
    assert [ast.unparse(arg) for arg in encoder.args] == ['batch[vanilla.ITEM_SEQ]', 'batch[vanilla.ITEM_SEQ_LEN]']
