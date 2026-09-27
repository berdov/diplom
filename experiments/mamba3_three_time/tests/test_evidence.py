import json
import pytest
import torch
from experiments.mamba3_three_time.evidence import create_record, update_record, tensor_records, tensor_difference, compare
from experiments.mamba3_three_time.initialization import assert_saved


def test_failure_preserved_before_raise(tmp_path):
    path = tmp_path/"initialization.json"
    create_record(path,{"status":"RUNNING"})
    bad = {"backbone_match_within_architecture":{
        "passed":False,"modes":["base","triple"],"differences":tensor_difference(
            {"weight":torch.ones(2)}, {"weight":torch.zeros(2)})}}
    with pytest.raises(AssertionError,match="weight"):
        assert_saved(bad,{"rows":[{"mode":"base"},{"mode":"triple"}]},lambda e:update_record(path,e))
    saved = json.loads(path.read_text())
    assert saved["status"] == "FAIL" and len(saved["rows"]) == 2
    assert saved["checks"]["backbone_match_within_architecture"]["differences"][0]["max_abs_diff"] == 1
    with pytest.raises(FileExistsError):
        create_record(path,{})


def test_tensor_hash_not_pickle_and_nonfinite():
    x = torch.arange(12,dtype=torch.bfloat16).reshape(3,4)
    assert tensor_records({"x":x}) == tensor_records({"x":x.T.contiguous().T})
    assert tensor_records({"x":x}) != tensor_records({"x":x.float()})
    assert compare(None,x)["passed"] is False
    result = compare(torch.tensor([float("nan")]),torch.ones(1))
    assert not result["finite"] and not result["passed"]
    json.dumps(result,allow_nan=False)
