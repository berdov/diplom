"""Named initialization checks, with evidence persisted before assertions."""

import hashlib
import json
import random
import numpy as np
import torch
from .calibrators import MODES
from .config import SISO_COUNTS, SyntheticCatalog, construct_config
from .evidence import tensor_bytes, tensor_records, state_hash, tensor_difference


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def rng_records():
    digest = lambda data: hashlib.sha256(data).hexdigest()
    numpy_state = np.random.get_state()
    return dict(python=digest(repr(random.getstate()).encode()),
        numpy=digest(numpy_state[1].tobytes()+repr((numpy_state[0], *numpy_state[2:])).encode()),
        cpu=digest(tensor_bytes(torch.get_rng_state())),
        cuda=[digest(tensor_bytes(s)) for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def split_state(model):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return ({k: v for k, v in state.items() if not k.startswith("times.")},
            {k: v for k, v in state.items() if k.startswith("times.")})


def assert_saved(checks, evidence, save):
    evidence["checks"] = checks
    evidence["status"] = "PASS" if all(c["passed"] for c in checks.values()) else "FAIL"
    save(evidence)
    if evidence["status"] != "PASS":
        raise AssertionError(json.dumps({k: v for k, v in checks.items() if not v["passed"]}))


def initialization_suite(architecture, device, save):
    from .model import ThreeTimeMamba3Rec
    if str(device).startswith("cuda") and not torch.cuda.is_initialized():
        raise RuntimeError("Set device and warm CUDA before config/seed/model")
    evidence = dict(architecture=architecture, status="RUNNING", rows=[], checks={},
        order=["set_device_and_warm_CUDA (GPU only)", "Config", "seed_all", "CPU model with isolated calibrators", "model.to(device)", "snapshot"])
    states, rngs, counts = {}, {}, {}
    for mode in MODES:
        before = torch.cuda.is_initialized()
        config = construct_config(architecture, mode, device)
        seed_all()
        model = ThreeTimeMamba3Rec(config, SyntheticCatalog()).to(device)
        counts[mode] = sum(p.numel() for p in model.parameters())
        states[mode] = split_state(model)
        rngs[mode] = rng_records()
        backbone, calibrators = (tensor_records(s) for s in states[mode])
        evidence["rows"].append(dict(architecture=architecture, mode=mode, actual_count=counts[mode],
            backbone_tensors=backbone, backbone_hash=state_hash(backbone),
            calibrator_tensors=calibrators, calibrator_hash=state_hash(calibrators),
            rng=rngs[mode], cuda_initialized_before=before, cuda_initialized_after=torch.cuda.is_initialized(),
            order=evidence["order"]))
        save(evidence)
        del model
    expected = SISO_COUNTS if architecture == "SISO" else {
        "base": 714888, "dual": 715020, "triple": 715086}
    mismatches = {mode: tensor_difference(states["base"][0], states[mode][0]) for mode in ("dual", "triple")}
    rng_differences = {mode: [key for key in rngs["base"] if rngs["base"][key] != rngs[mode][key]] for mode in ("dual", "triple")}
    common = {}
    for destination, source in (("decay", "decay"), ("write", "scan"), ("phase", "scan")):
        extract = lambda mode, name: {k.split(f"times.calibrators.{name}.", 1)[1]: v
            for k, v in states[mode][1].items() if k.startswith(f"times.calibrators.{name}.")}
        common[destination] = tensor_difference(extract("dual", source), extract("triple", destination))
    checks = dict(parameter_count_match=dict(passed=counts == expected, actual=counts, expected=expected),
        backbone_match_within_architecture=dict(passed=not any(mismatches.values()), differences=mismatches),
        rng_match_within_architecture=dict(passed=not any(rng_differences.values()), differing_fields=rng_differences),
        dual_triple_common_initialization_match=dict(passed=not any(common.values()), differences=common))
    assert_saved(checks, evidence, save)
    return evidence
