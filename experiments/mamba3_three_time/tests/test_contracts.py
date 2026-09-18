import ast
import importlib.util
import json
from pathlib import Path
import pytest
from experiments.mamba3_three_time.config import settings
from experiments.mamba3_three_time.provenance import HERE, ROOT, verify, sha


def test_frozen_sources_and_full_manifest():
    result = verify()
    assert result["core_hash"] == "460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f"
    for path in [*HERE.glob("*.py"),*(HERE/"tests").glob("*.py")]:
        assert str(path.relative_to(ROOT)) in result["files"]
    upstream = json.loads((HERE/"upstream_manifest.json").read_text())
    assert sha(HERE/"LICENSE.upstream") == upstream["files"]["LICENSE"]


def test_config_and_no_scientific_calls():
    for arch in ("SISO","MIMO"):
        for mode in ("base","dual","triple"):
            config = settings(arch,mode)
            assert config["time_scale_reference"] == 838393
            assert config["mamba3_chunk_size"] == (64 if arch == "SISO" else 8)
            assert config["mamba3_mimo_rank"] == 4
            assert config["train_batch_size"] == 2048 and config["eval_batch_size"] == 4096
    for path in HERE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Call):
                name = node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,"id","")
                assert name not in {"fit","evaluate","create_dataset","data_preparation","_valid_epoch"}


def test_real_model_construction_when_dependencies_available(tmp_path):
    if any(importlib.util.find_spec(name) is None for name in ("recbole","mamba_ssm")):
        pytest.skip("RecBole/pinned Mamba unavailable locally; actual counts checked by cluster CPU preflight")
    from experiments.mamba3_three_time.initialization import initialization_suite
    rows=[]
    for arch in ("SISO","MIMO"):
        result = initialization_suite(arch,"cpu",lambda row:rows.append(row))
        assert result["status"] == "PASS"


def test_launcher_is_technical_and_single_submit():
    text = (ROOT/"slurm/mamba3_three_time_correctness.sh").read_text()
    for token in ("--time=01:30:00","--no-requeue","--mem=0","--gres=gpu:a100:1",
                  "--partition=rocky","--account=proj_1833","--constraint=type_e","--cpus-per-task=4"):
        assert token in text
    assert "git " not in text and "sbatch" not in text
    submit = (HERE/"submit.py").read_text()
    assert submit.count('["sbatch",') == 1
    assert not any(name in submit for name in ("sacct", "squeue", "sleep", "watch"))
