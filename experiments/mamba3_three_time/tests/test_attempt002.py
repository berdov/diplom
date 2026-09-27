import ast
import copy
import json
from pathlib import Path
import pytest
import torch
from experiments.mamba3_three_time import attempt
from experiments.mamba3_three_time.backends import selected, current
from experiments.mamba3_three_time.diagnostics import tail_stats
from experiments.mamba3_three_time.gpu_checks_002 import expected_suite_names, localization_verdict
from experiments.mamba3_three_time.provenance import HERE, ROOT


@pytest.mark.parametrize("length",[1,7,8,17,63,64,65,97])
def test_adt_identity_indices_partial_and_carry(length):
    torch.manual_seed(2026)
    r = torch.randint(-32,33,(length,),dtype=torch.int64)
    v = torch.randint(-32,33,(length,),dtype=torch.int64)
    for start in range(0,length,64):
        a,b = torch.zeros(64,dtype=torch.int64),torch.zeros(64,dtype=torch.int64)
        size = min(64,length-start)
        a[:size],b[:size] = r[start:start+size],v[start:start+size]
        for carry in (-73,0,59):
            original = a + a.sum()+carry+(b-a).cumsum(0)-b
            exclusive = torch.cat((torch.zeros(1,dtype=b.dtype),b.cumsum(0)[:-1]))
            stable = a.flip(0).cumsum(0).flip(0)+exclusive+carry
            direct = torch.tensor([a[i:].sum()+b[:i].sum()+carry for i in range(64)])
            assert torch.equal(original,stable) and torch.equal(stable,direct)
            assert stable[0] == a.sum()+carry
            assert stable[-1] == a[-1]+b[:-1].sum()+carry


def test_real_valued_adt_derivatives():
    torch.manual_seed(6)
    r,v,c = [torch.randn(n,dtype=torch.float64,requires_grad=True) for n in (17,17,1)]
    original = r+r.sum()+c+(v-r).cumsum(0)-v
    stable = r.flip(0).cumsum(0).flip(0)+torch.cat((v[:1]*0,v.cumsum(0)[:-1]))+c
    weights = torch.randn(17,dtype=torch.float64)
    assert torch.allclose(original,stable,atol=1e-12,rtol=1e-12)
    a = torch.autograd.grad((weights*original).sum(),(r,v,c))
    b = torch.autograd.grad((weights*stable).sum(),(r,v,c))
    assert all(torch.allclose(x,y,atol=1e-12,rtol=1e-12) for x,y in zip(a,b))


@pytest.mark.parametrize("length,prefix",[(17,7),(65,31),(97,65)])
def test_reverse_scan_and_cross_chunk_carry(length,prefix):
    torch.manual_seed(2026)
    x = torch.randn(length,2,dtype=torch.float64)
    x[prefix:] = 0
    carry = torch.zeros(2,dtype=x.dtype)
    out = torch.empty_like(x)
    for start in reversed(range(0,length,64)):
        block = x[start:start+64]
        out[start:start+64] = block.flip(0).cumsum(0).flip(0)+carry
        carry = carry+block.sum(0)
    assert torch.allclose(out,x.flip(0).cumsum(0).flip(0),atol=1e-12,rtol=1e-12)
    assert torch.count_nonzero(out[prefix:]) == 0


def test_diagnostic_tail_does_not_use_shared_embedding_table():
    gradient = torch.zeros(1,17,2,3)
    gradient[:, :7] = 1
    gradient[0,9,1,2] = 1e-11
    row = tail_stats(gradient,7)
    assert not row["exact_zero"] and row["nonzero_entries"] == 1
    assert row["first_nonzero_indices"] == [[0,9,1,2]]
    assert row["safe_denominator"] == row["prefix_l2"]
    assert row["future_l2"] > 0


def test_backend_selection_is_scoped_and_default_not_promoted():
    assert current() == "upstream"
    with selected("stable_scan"):
        assert current() == "stable_scan"
    assert current() == "upstream"
    with pytest.raises(ValueError):
        with selected("unknown"):
            pass


def test_attempt002_no_overwrite_and_separate_paths(tmp_path,monkeypatch):
    monkeypatch.setattr(attempt,"LOGS",tmp_path/"attempt_002")
    monkeypatch.setattr(attempt,"RUNS",tmp_path/"runs")
    for name in ("LOCK","PIPELINE","SUMMARY"):
        monkeypatch.setattr(attempt,name,tmp_path/(name+"_002"))
    attempt.LOGS.mkdir()
    (attempt.LOGS/"old_001.lock").write_text("old")
    attempt.require_unused()
    attempt.LOCK.write_text("owned")
    with pytest.raises(FileExistsError):
        attempt.require_unused()
    assert (attempt.LOGS/"old_001.lock").read_text() == "old"


def test_plans_tolerances_preserved_and_coverage():
    old = json.loads((HERE/"evidence/attempt_001/test_plan.json").read_text())
    new = json.loads(attempt.PLAN.read_text())
    assert new["reference"] == old["reference"] and new["structural"] == old["structural"]
    assert new["lengths"]["MIMO"] == [1,7,8,9,15,16,17,50,65]
    assert len(expected_suite_names("SISO",new)) == 34
    assert len(expected_suite_names("MIMO",new)) == 43
    assert localization_verdict([])["status"] == "INCONCLUSIVE"


def test_new_launcher_and_submit_are_single_attempt():
    source = (HERE/"submit_002.py").read_text()
    assert source.count('["sbatch",') == 1
    assert not any(s in source for s in ("squeue","sacct","sleep","watch"))
    launcher = (ROOT/"slurm/mamba3_three_time_correctness_002.sh").read_text()
    for s in ("--time=01:30:00","--mem=0","--no-requeue","--cpus-per-task=4","--gres=gpu:a100:1",
              "--partition=rocky","--account=proj_1833","--constraint=type_e","slurm_logs/attempt_002/"):
        assert s in launcher
    assert "sbatch" not in launcher and "git " not in launcher
    assert 'weights_only=True' in (HERE/"suites.py").read_text()


def test_candidates_do_not_receive_prefix_or_change_forward():
    for name in ("stable_angle.py","stable_adt.py"):
        tree = ast.parse((HERE/name).read_text())
        functions = [node for node in tree.body if isinstance(node,ast.FunctionDef)]
        assert len(functions) == 2
        assert not any("prefix" in a.arg for f in functions for a in f.args.args)
    code = (HERE/"stable_angle.py").read_text()
    assert "reverse=True" in code and "sech2_approx" in code
    code = (HERE/"stable_adt.py").read_text()
    assert "reverse=True" in code and "tl.gather(prefix_v" in code


def test_native_evidence_survives_backward_failure(monkeypatch):
    from experiments.mamba3_three_time import gpu_checks_002 as checks
    class FailingBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):
            return x.clone()
        @staticmethod
        def backward(ctx,dy):
            raise RuntimeError("synthetic native shared-memory failure")
    x = torch.ones(1,17,2,3,requires_grad=True)
    monkeypatch.setattr(checks,"kernel_inputs",lambda *a,**k:{"x":x})
    monkeypatch.setattr(checks,"kernel_output",lambda *a,**k:FailingBackward.apply(x))
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    result, snapshots = {}, []
    with pytest.raises(RuntimeError,match="shared-memory"):
        checks.native_gate("SISO",result,lambda:snapshots.append(copy.deepcopy(result)))
    assert snapshots[-2]["native_support"]["backward"] == "RUNNING"
    assert snapshots[-1]["native_support"]["forward"] == "PASS"
    assert snapshots[-1]["native_support"]["backward"] == "FAIL"
    assert snapshots[-1]["native_support"]["status"] == "HARDWARE_OR_KERNEL_BLOCKED"
    assert "synthetic native shared-memory failure" in snapshots[-1]["native_support"]["traceback"]


def test_copy_validation_rejects_unrelated_changes():
    from experiments.mamba3_three_time.scan_validation import validate_one, ANGLE_OLD, ANGLE_NEW
    def source(body):
        return "def kernel():\n"+"\n".join("    "+s for s in body.splitlines())+"\ndef wrapper():\n    return 1\n"
    assert validate_one(source(ANGLE_OLD),source(ANGLE_NEW),ANGLE_OLD,ANGLE_NEW,("kernel","wrapper"))
    with pytest.raises(ValueError,match="unrelated"):
        validate_one(source(ANGLE_OLD),source(ANGLE_NEW).replace("return 1","return 2"),ANGLE_OLD,ANGLE_NEW,("kernel","wrapper"))
