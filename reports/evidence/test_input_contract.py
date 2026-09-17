"""CPU wiring tests, NOT a RecBole pipeline or Mamba GPU forward test."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.mamba3_timeaware.time_inputs import TimeCalibrator, history_gaps
from experiments.mamba3_time_mechanisms.time_mechanisms import TimeMechanisms


ROOT = Path(__file__).resolve().parents[2]


def methods(path, names):
    tree = ast.parse((ROOT / path).read_text())
    class_name = {"mamba3_baseline": "Mamba3Rec", "mamba3_timeaware": "TimeAwareMamba3Rec",
                  "mamba3_time_mechanisms": "MechanismMamba3Rec"}[Path(path).parts[1]]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = [node for node in cls.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    scope = {"torch": torch, "history_gaps": history_gaps}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), scope)
    return {name: scope[name] for name in names}


def cpu_receiver(kind):
    # Actual wrapper methods, but zero mixer layers: only input/scorer wiring.
    base = "experiments/mamba3_baseline/model.py"
    rt = "experiments/mamba3_timeaware/model.py"
    names = ["forward", "calculate_loss", "full_sort_predict"]
    funcs = methods(base if kind == "vanilla" else rt,
                    names if kind == "vanilla" else names + ["_encode"])
    if kind == "separate":
        funcs.update(methods("experiments/mamba3_time_mechanisms/model.py", ["forward"]))
    cls = type("CPUWiringDouble", (torch.nn.Module,), funcs)
    model = cls()
    model.ITEM_SEQ, model.ITEM_SEQ_LEN = "item_id_list", "item_length"
    model.POS_ITEM_ID, model.time_sequence_field = "item_id", "timestamp_list"
    with torch.random.fork_rng():
        torch.manual_seed(11)
        model.item_embedding = torch.nn.Embedding(7112, 64, padding_idx=0)
        model.time_calibrator = TimeCalibrator(2, 838393.)
        model.mechanisms = TimeMechanisms("separate")
    model.input_norm = model.input_dropout = model.output_norm = torch.nn.Identity()
    model.layers = []
    model.diagnostic_collector = None
    model.loss_fct = torch.nn.CrossEntropyLoss()
    model.gather_indexes = lambda values, index: values[torch.arange(len(index)), index]
    return model.eval()


def interaction():
    return {"item_id_list": torch.tensor([[1, 1, 0], [2, 3, 4]]),
            "item_length": torch.tensor([2, 3]),
            "timestamp_list": torch.tensor([[1649673604040, 1649673611472, 0],
                                             [1649673604040, 1649673604040, 1649673604050]],
                                            dtype=torch.float64),
            "item_id": torch.tensor([1, 4]), "timestamp": torch.tensor([1., 2.]),
            "user_id": torch.tensor([7, 8]), "t_score": torch.tensor([3., 4.])}


@pytest.mark.parametrize("kind", ["vanilla", "rt", "separate"])
def test_encoder_fields_shapes_and_target_independence(kind):
    model, data = cpu_receiver(kind), interaction()
    allowed = {key: data[key] for key in ["item_id_list", "item_length", "timestamp_list"]}
    scores = model.full_sort_predict(allowed)
    assert scores.shape == (2, 7112)
    assert model.item_embedding(data["item_id_list"]).shape == (2, 3, 64)
    if kind != "vanilla":
        assert model._encode(allowed).shape == (2, 64)
    changed = dict(data, item_id=torch.tensor([6, 7]), timestamp=torch.tensor([9., 10.]),
                   user_id=torch.tensor([90, 91]), t_score=torch.tensor([1e10, 2e10]))
    assert torch.equal(scores, model.full_sort_predict(changed))
    assert not torch.equal(model.calculate_loss(data), model.calculate_loss(changed))


@pytest.mark.parametrize("kind", ["rt", "separate"])
def test_reject_non_right_padded_history(kind):
    data = interaction()
    data["item_id_list"][0] = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match="right-padded"):
        cpu_receiver(kind).full_sort_predict(data)


def test_gaps_first_padding_and_real_zero_gap():
    data = interaction()
    gaps, active = history_gaps(data["timestamp_list"], data["item_id_list"] != 0)
    assert gaps.dtype == torch.float64
    assert gaps.tolist() == [[0, 7432, 0], [0, 0, 10]]
    assert active.tolist() == [[False, True, False], [False, True, True]]
    calibrator = TimeCalibrator(2, 838393.)
    with torch.no_grad():
        calibrator.last.bias.fill_(0.5)
    scales = calibrator(gaps, active)
    assert scales.shape == (2, 3, 2) and scales.dtype == torch.float32
    assert torch.equal(scales[~active], torch.ones_like(scales[~active]))
    assert (scales[active] > 1).all()


def test_item_time_alignment_and_float32_ties():
    # Small observed TRAIN sample plus a synthetic exact tie with row-id ordering.
    times = np.array([1649673604040, 1649673611472, 1649673704264], dtype=np.int64)
    items = np.array([1, 1, 2])
    assert np.spacing(np.float32(times[0])) == 131072
    assert times.astype(np.float32)[0] == times.astype(np.float32)[1]
    order = np.argsort(times.astype(np.float32), kind="stable")
    assert order.tolist() == [0, 1, 2]
    assert list(zip(items[order][:2], times[order][:2])) == [(1, times[0]), (1, times[1])]
    assert items[order][2] == 2  # separate TRAIN target, not a history position
    records = [(2, 1000, 9), (1, 1000, 8)]
    ordered = sorted(records, key=lambda r: (r[1], r[2]))
    assert [r[0] for r in ordered] == [1, 2]


def test_precise_adapter_replaces_history_not_target():
    # RecBole augmentation is an explicit double; test the real adapter only.
    class AugmentationDouble:
        def data_augmentation(self):
            self.inter_feat = {"item_id_list": torch.tensor([[1, 1, 0]]),
                               "timestamp_list": torch.zeros(1, 3),
                               "timestamp": torch.tensor([1649673704264.]),
                               "_rt_exact_timestamp": torch.tensor([1649673704264.], dtype=torch.float64),
                               "_rt_exact_timestamp_list": torch.tensor(
                                   [[1649673604040, 1649673611472, 0]], dtype=torch.float64)}

    path = ROOT / "experiments/mamba3_timeaware/dataset.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    scope = {"torch": torch, "SequentialDataset": AugmentationDouble}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    dataset = scope["PreciseHistoryDataset"]()
    dataset.time_field, dataset.config = "timestamp", {"LIST_SUFFIX": "_list"}
    dataset.field2type, dataset.field2source, dataset.field2seqlen = {}, {}, {}
    dataset.data_augmentation()
    assert dataset.inter_feat["timestamp_list"].dtype == torch.float64
    assert dataset.inter_feat["timestamp"].dtype == torch.float32
    assert dataset.inter_feat["item_id_list"].tolist() == [[1, 1, 0]]
    assert dataset.inter_feat["timestamp_list"].tolist() == [[1649673604040, 1649673611472, 0]]
    assert not any(key.startswith("_rt_exact_timestamp") for key in dataset.inter_feat)
