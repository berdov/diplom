import ast
import json
from pathlib import Path
import subprocess

import pytest
import torch
from experiments.mamba3_time_mechanisms.config import load_config
from experiments.mamba3_timeaware.config import load_config as frozen
from experiments.mamba3_time_mechanisms.provenance import BASE, require_equivalence
from experiments.mamba3_time_mechanisms.time_mechanisms import MODES

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


@pytest.mark.parametrize('mode',MODES)
def test_config_parity(mode):
    config = load_config(mode)
    assert {k:v for k,v in config.items() if k!='time_mechanism_mode'} == frozen()
    assert config['time_mechanism_mode']==mode


def test_frozen_files_byte_stable():
    paths = ['experiments/mamba3_baseline','experiments/mamba3_timeaware',
             'experiments/mamba3_prototypes','experiments/results.csv','reports/RESULTS.md']
    assert not subprocess.check_output(['git','diff',BASE,'--',*paths],cwd=ROOT)


def test_model_and_runner_contracts():
    model = ast.parse((HERE/'model.py').read_text())
    cls = next(n for n in model.body if isinstance(n,ast.ClassDef))
    assert ast.unparse(cls.bases[0]) == 'TimeAwareMamba3Rec'
    assert {n.name for n in cls.body if isinstance(n,ast.FunctionDef)} == {'__init__','forward'}
    runner = (HERE/'run.py').read_text()
    assert "TEST='NOT_RUN'" in runner and 'test_evaluation_count=0' in runner
    assert 'del reserved' in runner and 'require_equivalence(args.equivalence_json)' in runner
    tree = ast.parse(runner)
    evaluations = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr in ('evaluate','_valid_epoch')]
    assert all(ast.unparse(n.args[0])=='valid_data' for n in evaluations)
    from experiments.mamba3_time_mechanisms.run import RUN_IDS
    assert set(RUN_IDS)=={'decay_only','scan_only','separate'}
    for path in HERE.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert not any(x in (node.module or '') for x in ('prototypes','multitask','moe'))


def test_no_fake_equivalence(tmp_path):
    path = tmp_path/'evidence.json'
    path.write_text(json.dumps(dict(status='SKIP')))
    with pytest.raises(ValueError): require_equivalence(path)


def test_gpu_suites_prepared_without_execution():
    source = (HERE/'gpu_equivalence.py').read_text()
    assert 'suite_a' in source and 'suite_b' in source
    assert "for length in (50,64)" in source and 'for training in (False,True)' in source
    assert "status='SKIP'" in source
    assert 'atol=1e-6, rtol=1e-5' in source
    assert 'optimizer' not in source.lower().replace('optimizer,','')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable: GPU equivalence NOT RUN')
def test_gpu_equivalence_explicitly_deferred():
    pytest.skip('This stage prepares GPU suites only; execute under separate authorization')
