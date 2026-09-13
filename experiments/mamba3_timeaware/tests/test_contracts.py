import ast
from pathlib import Path
import subprocess
import sys
import importlib.util
import pytest

from experiments.mamba3_timeaware.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_config_inherits_protocol():
    config = load_config()
    assert config['eval_args']['mode'] == 'full'
    assert config['eval_args']['split'] == {'LS': 'valid_and_test'}
    assert config['time_scale_reference'] is None
    assert config['time_scale_reference_source'] == 'TRAIN'
    assert config['hidden_size'] * config['mamba3_expand'] // config['mamba3_headdim'] == 2
    assert config['num_layers'] == 2
    assert config['checkpoint_dir'] == 'experiments/mamba3_timeaware/checkpoints'


def test_cpu_import_without_mamba():
    subprocess.run([sys.executable, '-c',
                    'import sys; import experiments.mamba3_timeaware.time_mamba3; '
                    'assert "mamba_ssm" not in sys.modules'], check=True)


def test_full_model_import():
    if any(importlib.util.find_spec(name) is None for name in ('mamba_ssm', 'recbole')):
        pytest.skip('Official CUDA Mamba3/RecBole dependencies unavailable on Mac')
    from experiments.mamba3_timeaware.model import TimeAwareMamba3Rec
    assert TimeAwareMamba3Rec.__name__ == 'TimeAwareMamba3Rec'


def test_model_static_contract():
    tree = ast.parse((ROOT / 'model.py').read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    forward = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    assert [a.arg for a in forward.args.args] == ['self', 'item_seq', 'item_seq_len', 'history_timestamps']
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert sum(isinstance(n.func, ast.Name) and n.func.id == 'TimeCalibrator' for n in calls) == 1
    embedding = [n for n in ast.walk(forward) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'item_embedding']
    assert len(embedding) == 1 and ast.unparse(embedding[0].args[0]) == 'item_seq'
    assert not (ROOT / 'run.py').exists()


def test_no_evaluation_runner_calls():
    for path in ROOT.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                assert name not in {'evaluate', 'fit', 'data_preparation', 'create_dataset'}
