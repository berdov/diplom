import json
from pathlib import Path
import subprocess
import sys
from copy import deepcopy
import pytest
import yaml
from experiments.moo_representative_challengers.safety import DataAccessGuard, validate_config, TASK_ORDER
from experiments.moo_representative_challengers.common import ROOT, load_config


def test_task_order_and_no_test_configuration():
    cfg=load_config(); inputs=yaml.safe_load((ROOT/cfg['source']['optuna_config']).read_text())
    validate_config(cfg,inputs)
    assert TASK_ORDER==('rank','is_click','long_view','is_like','is_profile_enter')
    cfg['objectives']['task_order'][0],cfg['objectives']['task_order'][1]=cfg['objectives']['task_order'][1],cfg['objectives']['task_order'][0]
    with pytest.raises(RuntimeError): validate_config(cfg,inputs)
    cfg=load_config(); inputs['recbole_overrides']['benchmark_filename'].append('test')
    with pytest.raises(RuntimeError): validate_config(cfg,inputs)


def test_loader_guard_prevents_construction():
    observed=[]; guard=DataAccessGuard([])
    factory=guard.loader_factory(lambda c,p:observed.append(p))
    factory({},'train'); factory({},'valid')
    with pytest.raises(RuntimeError): factory({},'test')
    assert observed==['train','valid']


def test_runtime_audit_blocks_real_open_and_symlink_and_data_write(tmp_path):
    # Separate interpreter: audit hooks cannot be uninstalled.
    code='''
import os, sys
from pathlib import Path
from experiments.moo_representative_challengers.safety import DataAccessGuard
p=Path(sys.argv[1]); shared=p/'immutable'; shared.mkdir()
(shared/'train.inter').write_text('allowed')
(p/'test.inter').write_text('forbidden')
(p/'alias.inter').symlink_to(p/'test.inter')
g=DataAccessGuard([shared]); g.install()
assert (shared/'train.inter').read_text()=='allowed'
for target in [p/'test.inter', p/'alias.inter']:
    try: target.read_text()
    except RuntimeError: pass
    else: raise AssertionError('TEST opened')
try: (shared/'train.inter').write_text('changed')
except RuntimeError: pass
else: raise AssertionError('Shared dataset overwritten')
assert (shared/'train.inter').read_text()=='allowed'
assert g.blocked_test_accesses==2
'''
    # Parent's pytest-generated path contains a test_ component, so use a fresh neutral directory.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='moo_guard_') as directory:
        subprocess.run([sys.executable,'-c',code,directory],cwd=ROOT,check=True,capture_output=True,text=True)


def test_cli_rejects_test_and_tuning_before_importing_dataset():
    for stage in ['final_test','test','tuning']:
        result=subprocess.run([sys.executable,'-m','experiments.moo_representative_challengers.run',
            '--method','ferero','--stage',stage],cwd=ROOT,capture_output=True,text=True)
        assert result.returncode==2 and 'invalid choice' in result.stderr
