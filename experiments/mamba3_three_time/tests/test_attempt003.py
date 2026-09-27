import copy
import json
import pytest
import torch
from experiments.mamba3_three_time import attempt_003 as attempt
from experiments.mamba3_three_time.provenance import HERE, ROOT
from experiments.mamba3_three_time.records_003 import Registry, required_pass, interrupt_file
from experiments.mamba3_three_time.evidence import create_record
from experiments.mamba3_three_time.length_adapter import prepare, AXES
from experiments.mamba3_three_time.fixtures import kernel_inputs, clone_inputs, scalar_loss
from experiments.mamba3_three_time.reference import recurrence
from experiments.mamba3_three_time.gpu_checks_003 import specs, suite_factories


def registry(expected=("one",)):
    result,saved = {},[]
    r = Registry(result,lambda:saved.append(copy.deepcopy(result)),list(expected))
    return r,result,saved


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError,match="Duplicate"):
        registry(("x","x"))
    r,_,_ = registry()
    r.start("one")
    with pytest.raises(ValueError,match="Duplicate"):
        r.start("one")


def test_nested_required_failure_and_advisory():
    r,result,_ = registry()
    r.start("one")
    row = r.finish("one",dict(passed=True,status="PASS",nested=dict(passed=False)))
    assert not row["passed"] and row["status"]=="FAIL" and not r.close()
    assert not required_pass(dict(passed=True,foo=[dict(passed=False)]))
    assert required_pass(dict(passed=True,foo=dict(required=False,passed=False)))


def test_missing_and_interrupted_cases():
    r,result,_ = registry(("one","two"))
    r.start("one")
    assert not r.close()
    assert result["progress"]["one"]["status"]=="INTERRUPTED"
    assert result["coverage"]["missing"]==["one","two"]


def test_gate_persists_before_assertion():
    r,result,saved = registry()
    with pytest.raises(RuntimeError,match="persisted"):
        r.run("one",lambda:dict(passed=False,checks={"x":dict(passed=False)}),gate=True)
    assert saved[-1]["progress"]["one"]["status"]=="FAIL"
    assert saved[-1]["cases"][0]["passed"] is False


def test_exception_and_parent_interruption(tmp_path):
    r,result,_ = registry()
    row = r.run("one",lambda:1/0)
    assert "ZeroDivisionError" in row["traceback"]
    assert result["progress"]["one"]["status"]=="FAIL"
    path=tmp_path/"evidence.json"
    create_record(path,dict(status="RUNNING",progress={"x":dict(status="RUNNING")}))
    interrupt_file(path)
    saved=json.loads(path.read_text())
    assert saved["status"]==saved["progress"]["x"]["status"]=="INTERRUPTED"


@pytest.mark.parametrize("length",[1,7,8,9,15,16,17,50,65])
def test_padding_axes_alias_neutrality_and_gradients(length):
    source=kernel_inputs("MIMO",length=length,device="cpu",dtype=torch.float64,tiny=True,tied=True)
    padded,meta=prepare(source)
    assert meta["kernel_length"]==(length+7)//8*8
    assert padded["dw"] is padded["dp"]
    for key,axis in AXES.items():
        assert torch.equal(padded[key].narrow(axis,0,length),source[key])
        assert not torch.count_nonzero(padded[key].narrow(axis,length,meta["kernel_length"]-length))
        if length%8==0:
            assert padded[key] is source[key]
    assert all(padded[k] is source[k] for k in source if k not in AXES)
    extra,extra_meta=prepare(source,extra_chunks=1)
    original=recurrence(**source)
    wrapped=recurrence(**extra)[:,:length]
    assert extra_meta["kernel_length"]==meta["kernel_length"]+8
    assert torch.allclose(original,wrapped,atol=1e-12,rtol=1e-12)
    leaves=list(dict.fromkeys(source.values()))
    a=torch.autograd.grad(scalar_loss(original),leaves,retain_graph=True)
    b=torch.autograd.grad(scalar_loss(wrapped),leaves)
    assert all(torch.allclose(x,y,atol=1e-12,rtol=1e-12) for x,y in zip(a,b))


def test_attempt003_disjoint_guards_and_old_immutable(monkeypatch,tmp_path):
    monkeypatch.setattr(attempt,"LOGS",tmp_path/"attempt_003")
    monkeypatch.setattr(attempt,"RUNS",tmp_path/"runs")
    for name in ("LOCK","PIPELINE","SUMMARY"):
        monkeypatch.setattr(attempt,name,tmp_path/(name+"003"))
    attempt.LOGS.mkdir()
    attempt.require_unused()
    attempt.LOCK.write_text("reserved")
    with pytest.raises(FileExistsError):
        attempt.require_unused()
    import hashlib
    for previous in ("001","002"):
        directory=HERE/"evidence"/("attempt_"+previous)
        manifest=json.loads((directory/"preservation_manifest.json").read_text())
        for row in manifest["files"]:
            path=ROOT/row.get("destination",row.get("destination_path"))
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"]


def test_plan_registry_resources_and_frozen_tolerances():
    plan=json.loads(attempt.PLAN.read_text())
    old=json.loads((HERE/"test_plan_002.json").read_text())
    assert plan["reference"]==old["reference"] and plan["structural"]==old["structural"]
    for arch in ("SISO","MIMO"):
        rows=specs(arch,plan)
        assert len(rows)==len(set(k for k,_ in rows))
        assert len(suite_factories(arch,plan))==(34 if arch=="SISO" else 43)
    code=(HERE/"submit_003.py").read_text()
    assert code.count('["sbatch",')==1
    assert not any(s in code for s in ("sacct","squeue","sleep","watch"))
    launcher=(ROOT/"slurm/mamba3_three_time_correctness_003.sh").read_text()
    for token in ("--partition=rocky","--account=proj_1833","--constraint=type_e","--gres=gpu:a100:1",
                  "--cpus-per-task=4","--mem=0","--time=01:30:00","--no-requeue"):
        assert token in launcher


def test_launch_recorder_does_not_mutate_original_wrapper_or_cache(monkeypatch):
    import sys
    from types import SimpleNamespace, FunctionType
    from experiments.mamba3_three_time.drift_capture import invoke, FIXED_LAUNCH
    monkeypatch.setitem(sys.modules,"triton",SimpleNamespace(runtime=SimpleNamespace(
        driver=SimpleNamespace(active=SimpleNamespace(get_current_target=lambda:"cuda:80")))))
    calls=[]
    config=SimpleNamespace(**FIXED_LAUNCH,kwargs={})
    class Jit:
        src="frozen kernel"
        cache_key="key"
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                calls.append((grid,args,kwargs))
                return SimpleNamespace(name="kernel",hash="hash",metadata=None)
            return call
    class Tuner:
        configs=[config]
        best_config=config
        fn=Jit()
        cache={"unchanged":"value"}
        def __getitem__(self,grid):
            return self.fn[grid]
    tuner=Tuner()
    def template(x):
        mamba3_siso_bwd_kernel_dqkv[(2,1)](x,CHUNK_SIZE=64)
        return (x,)
    fn=FunctionType(template.__code__,dict(mamba3_siso_bwd_kernel_dqkv=tuner),"wrapper")
    before=dict(tuner.cache)
    x=torch.tensor(2.)
    for matched in (False,True):
        values,records=invoke(fn,{"x":x},matched=matched)
        assert values[0] is x and records[0]["config"]["num_warps"]==4
        assert records[0]["compile_flags"]==dict(CHUNK_SIZE=64)
        assert fn.__globals__["mamba3_siso_bwd_kernel_dqkv"] is tuner
        assert tuner.cache==before
    assert all(calls[-1][2][k]==v for k,v in FIXED_LAUNCH.items())
