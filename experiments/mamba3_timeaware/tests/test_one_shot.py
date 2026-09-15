import ast
import json
from pathlib import Path

import pytest

from experiments.mamba3_timeaware.one_shot import OneShotResult


def test_existing_result_aborts(tmp_path):
    result = tmp_path / 'result.json'
    result.write_text('original')
    with pytest.raises(FileExistsError):
        OneShotResult(result, tmp_path / 'lock')
    assert result.read_text() == 'original'


def test_parallel_guard_and_failed_attempt_blocks_retry(tmp_path):
    result = tmp_path / 'result.json'
    lock = tmp_path / 'lock'
    OneShotResult(result, lock)
    with pytest.raises(FileExistsError):
        OneShotResult(result, lock)
    assert not result.exists()


def test_atomic_publish_no_overwrite(tmp_path):
    result = tmp_path / 'result.json'
    lock = tmp_path / 'lock'
    guard = OneShotResult(result, lock)
    guard.publish({'status': 'PASS', 'test_evaluation_count': 1})
    assert json.loads(result.read_text())['test_evaluation_count'] == 1
    assert not result.with_suffix('.json.tmp').exists()
    assert lock.exists()
    with pytest.raises(FileExistsError):
        guard.publish({'status': 'wrong'})


def test_bad_payload_does_not_publish(tmp_path):
    result = tmp_path / 'result.json'
    guard = OneShotResult(result, tmp_path / 'lock')
    with pytest.raises(ValueError):
        guard.publish({'metric': float('nan')})
    assert not result.exists()


def test_final_runner_static_contract():
    path = Path(__file__).resolve().parents[1] / 'mamba3_timeaware_final_test.py'
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '') for node in calls]
    assert 'fit' not in names and 'backward' not in names and 'TrainDataLoader' not in names
    evaluates = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'evaluate']
    assert len(evaluates) == 1
    assert ast.unparse(evaluates[0].args[0]) == 'test_data'
    evaluator = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    builder = next(node for node in evaluator.body if isinstance(node, ast.FunctionDef) and node.name == '_build_optimizer')
    assert ast.unparse(builder.body[0]) == 'return None'
