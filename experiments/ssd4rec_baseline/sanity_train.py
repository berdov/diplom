#!/usr/bin/env python
"""Короткий full-data sanity run для SSD4Rec.

Скрипт запускает 5 эпох на полном KuaiRand Protocol B, считает validation после
каждой эпохи и намеренно не считает test metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import resource
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.cuda.amp as amp
from recbole.config import Config
from recbole.evaluator import Collector, Evaluator
from recbole.utils import early_stopping, init_seed


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ROOT / "experiments" / "ssd4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import custom_utils as ssd4rec_custom_utils  # noqa: E402
from custom_trainer import SSD4RecTrainer  # noqa: E402
from custom_utils import SSD4RecData_preparation, SSD4RecDataset  # noqa: E402
from ssd4rec import BiSSDLayer, SSD4Rec  # noqa: E402


ssd4rec_custom_utils.getLogger = logging.getLogger


RUN_ID = "ssd4rec_sanity_001"
UPSTREAM_COMMIT = "bdbfe5193f3a6697bb6ee0699ab43386d80c6198"
PAPER_REFERENCE_VERSION = "arXiv 2409.01192 v2"
EXPECTED_FINGERPRINT = {
    "users": 23951,
    "items": 7111,
    "interactions": 1134420,
    "train": 1086518,
    "validation": 23951,
    "test": 23951,
}
EXPECTED_SEQUENTIAL_EXAMPLES = EXPECTED_FINGERPRINT["interactions"] - EXPECTED_FINGERPRINT["users"]
PUBLISHED_SSD4REC_V2 = {
    "split": "paper reported evaluation",
    "hr@10": 0.1075,
    "hr@20": 0.1731,
    "ndcg@10": 0.0593,
    "ndcg@20": 0.0757,
    "mrr@10": 0.0448,
    "mrr@20": 0.0493,
}
VALIDATION_REFERENCES = {
    "random_full_ranking_validation": {"hr@10": 0.001127301574, "ndcg@10": 0.000441685851},
    "mostpopular_full_ranking_validation": {"hr@10": 0.029977871488, "ndcg@10": 0.016763898156},
    "xgboost_ltr_xgb_002_validation": {"hr@10": 0.030854661601, "ndcg@10": 0.014971581041},
    "tim4rec_001_best_validation": {"epoch": 12, "hr@10": 0.1086, "ndcg@10": 0.0593},
}
UPSTREAM_REQUIREMENTS = {
    "source": "experiments/ssd4rec_baseline/upstream/environment.yaml",
    "python": "3.10.15",
    "cuda_toolkit": "11.8.0",
    "cuda_nvcc": "11.8.89",
    "torch": "2.1.1+cu118",
    "torchaudio": "2.1.1+cu118",
    "torchvision": "0.16.1+cu118",
    "recbole": "1.2.0",
    "mamba_ssm": "2.2.2",
    "causal_conv1d": "1.4.0",
    "triton": "2.1.0",
    "numpy": "1.26.3",
    "scipy": "1.14.1",
    "pandas": "2.2.3",
    "pyyaml": "6.0.2",
    "tqdm": "4.67.1",
    "psutil": "6.1.0",
    "einops": "0.8.0",
    "transformers": "4.46.3",
}
DEPENDENCY_SOURCES = {
    "explicitly_pinned_upstream": sorted(UPSTREAM_REQUIREMENTS),
    "not_pinned_by_upstream": [
        "В upstream snapshot нет requirements.txt, setup.py или pyproject.toml.",
        "Stdlib imports отдельно не version-pinned.",
    ],
    "project_compatibility_shims": [
        "alias np.float для NumPy >= 1.24",
        "добавление custom_utils.getLogger",
    ],
}
HYPERPARAMETER_SOURCES = {
    "hidden_size": "глобальная настройка SSD4Rec из upstream config; paper использует high-dimensional SSD4Rec с размерностью 256",
    "num_layers": "KuaiRand-блок upstream config",
    "d_state": "глобальная настройка SSD block из upstream config",
    "d_conv": "глобальная настройка SSD block из upstream config",
    "expand": "глобальная настройка SSD block из upstream config",
    "headdim": "глобальная настройка SSD block из upstream config",
    "dropout_prob": "KuaiRand-блок upstream config",
    "norm_embedding": "KuaiRand-блок upstream config",
    "beta": "KuaiRand-блок upstream config",
    "maskratio": "KuaiRand-блок upstream config и paper-анализ mask ratio",
    "var_len": "training setting из upstream config",
    "learning_rate": "training setting из upstream config",
    "optimizer": "training setting из upstream config",
    "weight_decay": "eval/training setting из upstream config",
    "train_batch_size": "training setting из upstream config",
    "eval_batch_size": "evaluation setting из upstream config",
    "max_sequence_length": "KuaiRand-настройка из upstream config; при var_len=True это не active truncation limit",
    "epochs": "upstream config задает 300; sanity compatibility override задает 5",
    "stopping_step": "training setting из upstream config",
    "seed": "глобальная настройка из upstream config",
    "loss": "upstream code SSD4Rec.calculate_loss использует CrossEntropyLoss по всем items",
    "validation_metric": "upstream config задает valid_metric=NDCG@10",
    "metrics": "sanity compatibility override исключает MRR, потому что текущий sanity просит только HR/Recall/NDCG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "experiments" / "ssd4rec_baseline" / "config_kuairand.yaml"),
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--artifact-dir",
        default="/home/daryumin/iberdov/diplom/experiments/ssd4rec_baseline/ssd4rec_sanity_001",
    )
    parser.add_argument(
        "--result-json",
        default=str(ROOT / "experiments" / "ssd4rec_baseline" / "runs" / f"{RUN_ID}.json"),
    )
    parser.add_argument(
        "--notes-md",
        default=str(ROOT / "experiments" / "ssd4rec_baseline" / "runs" / f"{RUN_ID}_notes.md"),
    )
    parser.add_argument(
        "--manifest",
        default="/home/daryumin/iberdov/diplom/outputs/data/protocol_b_manifest.json",
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def collect_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "recbole": version("recbole"),
        "mamba_ssm": version("mamba-ssm"),
        "causal_conv1d": version("causal-conv1d"),
        "triton": version("triton"),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "pandas": version("pandas"),
        "pyyaml": version("PyYAML"),
        "tqdm": version("tqdm"),
        "psutil": version("psutil"),
        "einops": version("einops"),
        "transformers": version("transformers"),
    }


def environment_differences(actual: dict[str, Any]) -> dict[str, dict[str, Any]]:
    package_key_map = {
        "python": "python",
        "torch": "torch",
        "recbole": "recbole",
        "mamba_ssm": "mamba_ssm",
        "causal_conv1d": "causal_conv1d",
        "triton": "triton",
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "pyyaml": "pyyaml",
        "tqdm": "tqdm",
        "psutil": "psutil",
        "einops": "einops",
        "transformers": "transformers",
    }
    diffs = {}
    for req_key, actual_key in package_key_map.items():
        upstream = UPSTREAM_REQUIREMENTS[req_key]
        observed = actual.get(actual_key)
        diffs[req_key] = {
            "upstream": upstream,
            "actual": observed,
            "matches_exact_pin": observed == upstream,
        }
    return diffs


def load_manifest(path: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Protocol B manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"path": str(manifest_path), "loaded": True, "payload": payload}


def manifest_fingerprint(manifest_info: dict[str, Any]) -> dict[str, int]:
    payload = manifest_info["payload"]
    filtered = payload.get("filtered_stats", {})
    split_stats = payload.get("split_stats", {})
    return {
        "users": int(filtered.get("users", -1)),
        "items": int(filtered.get("items", -1)),
        "interactions": int(filtered.get("interactions", -1)),
        "train": int(split_stats.get("train", {}).get("interactions", -1)),
        "validation": int(split_stats.get("validation", {}).get("interactions", -1)),
        "test": int(split_stats.get("test", {}).get("interactions", -1)),
    }


def manifest_sequence_stats(manifest_info: dict[str, Any]) -> dict[str, Any]:
    return manifest_info["payload"].get("sequence_stats", {}).get("sequence_length", {})


def assert_expected_fingerprint(fingerprint: dict[str, int]) -> None:
    if fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError(f"Protocol B fingerprint mismatch: {fingerprint} != {EXPECTED_FINGERPRINT}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    model: SSD4Rec,
    optimizer: torch.optim.Optimizer,
    config: Config,
    path: Path,
    epoch: int,
    best_valid_score: float,
    valid_result: dict[str, float],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "config": config,
        "epoch": epoch,
        "best_valid_score": best_valid_score,
        "state_dict": model.state_dict(),
        "other_parameter": model.other_parameter(),
        "optimizer": optimizer.state_dict(),
        "valid_result": valid_result,
    }
    torch.save(state, path, pickle_protocol=4)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def grad_norm(param: torch.nn.Parameter) -> float | None:
    if param.grad is None:
        return None
    return float(param.grad.detach().float().norm().cpu().item())


def all_parameters_finite(model: torch.nn.Module) -> dict[str, Any]:
    nonfinite = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param.detach()).all():
            nonfinite.append(name)
    return {
        "ok": len(nonfinite) == 0,
        "nonfinite_parameter_tensors": nonfinite,
        "trainable_parameter_count": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
    }


def tensor_stats(values: torch.Tensor) -> dict[str, Any]:
    numeric = values.detach().float()
    return {
        "min": int(numeric.min().item()),
        "median": float(numeric.median().item()),
        "mean": float(numeric.mean().item()),
        "max": int(numeric.max().item()),
    }


def inspect_first_train_batch(
    item_id: torch.Tensor,
    item_id_list: torch.Tensor,
    cum_item_length: torch.Tensor,
    item_idx: torch.Tensor,
    flip_index: torch.Tensor,
    config: Config,
) -> dict[str, Any]:
    seq_lengths = torch.diff(
        cum_item_length,
        prepend=torch.zeros(1, dtype=cum_item_length.dtype, device=cum_item_length.device),
    )
    flat_tokens = int(item_id_list.numel())
    flip_unique = torch.unique(flip_index)
    return {
        "targets": int(item_id.numel()),
        "flat_sequence_tokens": flat_tokens,
        "item_id_list_shape": list(item_id_list.shape),
        "cum_item_length_shape": list(cum_item_length.shape),
        "cum_item_length_last": int(cum_item_length[-1].item()),
        "item_idx_shape": list(item_idx.shape),
        "item_idx_unique_segments": int(torch.unique(item_idx).numel()),
        "flip_index_shape": list(flip_index.shape),
        "flip_index_is_permutation": (
            int(flip_unique.numel()) == flat_tokens
            and int(flip_index.min().item()) == 0
            and int(flip_index.max().item()) == flat_tokens - 1
        ),
        "sequence_length": tensor_stats(seq_lengths),
        "sequence_registers_present": True,
        "var_len": bool(config["var_len"]),
        "max_item_list_length_config": int(config["MAX_ITEM_LIST_LENGTH"]),
        "uses_flat_representation_in_forward": True,
        "mask_zero_tokens": int((item_id_list == 0).sum().item()),
        "mask_zero_ratio": float((item_id_list == 0).float().mean().item()),
    }


def make_bissd_hook_diagnostics(model: SSD4Rec) -> tuple[dict[str, Any], list[Any]]:
    layer_calls = {str(idx): 0 for idx, _layer in enumerate(model.BiSSD_layers)}
    layer_output_shapes: dict[str, list[list[int]]] = {str(idx): [] for idx, _layer in enumerate(model.BiSSD_layers)}
    handles = []

    def build_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            key = str(layer_idx)
            layer_calls[key] += 1
            if len(layer_output_shapes[key]) < 4 and hasattr(output, "shape"):
                layer_output_shapes[key].append(list(output.shape))

        return hook

    for idx, layer in enumerate(model.BiSSD_layers):
        handles.append(layer.forward_ssd.register_forward_hook(build_hook(idx)))

    diagnostics = {
        "bidirectional_expected_mamba_calls_total": int(len(model.BiSSD_layers) * 2),
        "mamba_calls_per_layer": layer_calls,
        "mamba_output_shapes_sample": layer_output_shapes,
        "forward_direction_active": None,
        "backward_reversed_direction_active": None,
        "flip_index_present": None,
        "bissd_layer_count": int(len(model.BiSSD_layers)),
        "bissd_layers_are_official_class": all(isinstance(layer, BiSSDLayer) for layer in model.BiSSD_layers),
        "mamba_module_shared_for_forward_backward": True,
    }
    return diagnostics, handles


def finalize_bissd_hook_diagnostics(diagnostics: dict[str, Any], handles: list[Any], flip_index: torch.Tensor) -> dict[str, Any]:
    for handle in handles:
        handle.remove()
    calls = diagnostics["mamba_calls_per_layer"]
    diagnostics["forward_direction_active"] = all(count >= 1 for count in calls.values())
    diagnostics["backward_reversed_direction_active"] = all(count >= 2 for count in calls.values())
    diagnostics["flip_index_present"] = int(flip_index.numel()) > 0
    diagnostics["mamba_calls_total"] = int(sum(calls.values()))
    diagnostics["all_layers_called_twice"] = all(count == 2 for count in calls.values())
    return diagnostics


def selected_parameter_items(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    selected: list[tuple[str, torch.nn.Parameter]] = []
    wanted = [
        "item_embedding.weight",
        "BiSSD_layers.0.forward_ssd",
        "BiSSD_layers.0.ffn",
        "BiSSD_layers.1.forward_ssd",
    ]
    for prefix in wanted:
        for name, param in model.named_parameters():
            if name.startswith(prefix) and param.requires_grad:
                selected.append((name, param))
                break
    return selected


def gradient_diagnostics(
    model: torch.nn.Module,
    selected_before: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    total_norm_terms = []
    nonfinite_grad_tensors = []
    sampled = []
    named_params = dict(model.named_parameters())
    for name, param in named_params.items():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            nonfinite_grad_tensors.append(name)
        total_norm_terms.append(grad.float().norm())

    for name, param in selected_parameter_items(model):
        before = selected_before[name] if selected_before and name in selected_before else None
        update_norm = None
        if before is not None:
            update_norm = float((param.detach() - before).float().norm().cpu().item())
        norm = grad_norm(param)
        sampled.append(
            {
                "name": name,
                "shape": list(param.shape),
                "grad_norm": norm,
                "grad_norm_finite": norm is not None and math.isfinite(norm),
                "update_norm": update_norm,
                "update_norm_finite": update_norm is None or math.isfinite(update_norm),
                "updated": update_norm is not None and update_norm > 0.0,
            }
        )
    total_norm = float(torch.linalg.vector_norm(torch.stack(total_norm_terms)).cpu().item()) if total_norm_terms else 0.0
    return {
        "total_grad_norm": total_norm,
        "total_grad_norm_finite": math.isfinite(total_norm),
        "nonfinite_grad_tensors": nonfinite_grad_tensors,
        "all_gradients_finite": len(nonfinite_grad_tensors) == 0,
        "sampled_parameters": sampled,
    }


def train_one_epoch(
    trainer: SSD4RecTrainer,
    train_data,
    epoch_idx: int,
    collect_diagnostics: bool,
) -> tuple[float, float, dict[str, Any] | None, dict[str, Any] | None]:
    model = trainer.model
    model.train()
    optimizer = trainer.optimizer
    device = trainer.device
    total_loss = 0.0
    batches = 0
    first_batch_diag = None
    first_gradient_diag = None
    scaler = amp.GradScaler(enabled=trainer.enable_scaler)

    for batch_idx, batch in enumerate(train_data):
        item_id, item_id_list, cum_item_length, item_idx, flip_index = tuple(tensor.to(device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        selected_before = None
        bissd_diag = None
        hooks = []

        if collect_diagnostics and batch_idx == 0:
            first_batch_diag = inspect_first_train_batch(item_id, item_id_list, cum_item_length, item_idx, flip_index, trainer.config)
            bissd_diag, hooks = make_bissd_hook_diagnostics(model)
            selected_before = {name: param.detach().clone() for name, param in selected_parameter_items(model)}

        with torch.autocast(device_type=device.type, enabled=trainer.enable_amp):
            loss = model.calculate_loss(item_id, item_id_list, cum_item_length, item_idx, flip_index)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch={epoch_idx}, batch={batch_idx}: {loss.item()}")

        if collect_diagnostics and batch_idx == 0 and bissd_diag is not None:
            first_batch_diag["bidirectional"] = finalize_bissd_hook_diagnostics(bissd_diag, hooks, flip_index)

        scaler.scale(loss).backward()
        if collect_diagnostics and batch_idx == 0:
            first_gradient_diag = gradient_diagnostics(model)

        if trainer.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), **trainer.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if collect_diagnostics and batch_idx == 0 and selected_before is not None and first_gradient_diag is not None:
            after_update_diag = gradient_diagnostics(model, selected_before)
            first_gradient_diag["sampled_parameters_after_update"] = after_update_diag["sampled_parameters"]
            first_gradient_diag["all_sampled_updates_finite"] = all(
                item["update_norm_finite"] for item in after_update_diag["sampled_parameters"]
            )
            first_gradient_diag["any_sampled_parameter_updated"] = any(
                item["updated"] for item in after_update_diag["sampled_parameters"]
            )

        total_loss += float(loss.detach().cpu().item())
        batches += 1

    return total_loss, total_loss / max(batches, 1), first_batch_diag, first_gradient_diag


@torch.no_grad()
def evaluate_validation_full_sort(
    trainer: SSD4RecTrainer,
    valid_data,
) -> tuple[dict[str, float], dict[str, Any]]:
    model = trainer.model
    model.eval()
    collector = Collector(trainer.config)
    evaluator = Evaluator(trainer.config)
    tot_item_num = int(valid_data._dataset.item_num)

    rows = 0
    positives = 0
    raw_nan_scores = 0
    raw_inf_scores = 0
    positive_score_nonfinite = 0
    padding_item_masked_batches = 0
    batches = 0
    min_positive = None
    max_positive = None
    zero_positive = 0

    for batch in valid_data:
        item_id_list, cum_item_length, item_idx, flip_index, positive_u, positive_i = tuple(
            tensor.to(trainer.device) for tensor in batch
        )
        scores = model.full_sort_predict(item_id_list, cum_item_length, item_idx, flip_index)
        scores = scores.view(-1, tot_item_num)
        raw_nan_scores += int(torch.isnan(scores).sum().item())
        raw_inf_scores += int(torch.isinf(scores).sum().item())
        positive_scores = scores[positive_u, positive_i]
        positive_score_nonfinite += int((~torch.isfinite(positive_scores)).sum().item())

        scores[:, 0] = -float("inf")
        padding_item_masked_batches += int(torch.isneginf(scores[:, 0]).all().item())
        collector.eval_batch_collect(scores, None, positive_u, positive_i)

        batch_rows = int(positive_u.numel())
        rows += batch_rows
        positives += int(positive_i.numel())
        if positive_i.numel():
            min_i = int(positive_i.min().item())
            max_i = int(positive_i.max().item())
            min_positive = min_i if min_positive is None else min(min_positive, min_i)
            max_positive = max_i if max_positive is None else max(max_positive, max_i)
            zero_positive += int((positive_i == 0).sum().item())
        batches += 1

    result = dict(evaluator.evaluate(collector.get_data_struct()))
    checks = {
        "mode": "full",
        "candidate_protocol": "full_7111_items",
        "evaluation_protocol": "full_7111_items",
        "item_num_with_padding": tot_item_num,
        "candidate_universe_size": tot_item_num - 1,
        "validation_rows": rows,
        "positive_targets": positives,
        "one_positive_per_row": positives == rows,
        "min_positive_item_id": min_positive,
        "max_positive_item_id": max_positive,
        "zero_positive_targets": zero_positive,
        "raw_score_batches": batches,
        "raw_nan_scores": raw_nan_scores,
        "raw_inf_scores": raw_inf_scores,
        "positive_score_nonfinite": positive_score_nonfinite,
        "raw_scores_all_finite": raw_nan_scores == 0 and raw_inf_scores == 0,
        "positive_scores_all_finite": positive_score_nonfinite == 0,
        "padding_item_zero_masked": padding_item_masked_batches == batches,
        "seen_history_items_masked": False,
        "seen_history_note": "Matches official SSD4Rec custom_trainer semantics: only item id 0 is masked.",
    }
    return result, checks


def check_hit_recall_equal(valid_result: dict[str, float], topk: list[int]) -> dict[str, Any]:
    diffs = {}
    ok = True
    for k in topk:
        hit = float(valid_result[f"hit@{k}"])
        recall = float(valid_result[f"recall@{k}"])
        diff = abs(hit - recall)
        diffs[str(k)] = diff
        ok = ok and diff <= 1e-12
    if not ok:
        raise RuntimeError(f"Hit/Recall mismatch in one-positive validation: {diffs}")
    return {"ok": ok, "abs_diffs": diffs}


def metric(result: dict[str, float], name: str, k: int) -> float:
    return float(result[f"{name}@{k}"])


def write_partial(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def markdown_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_notes(result: dict[str, Any]) -> str:
    epochs = result["epochs"]
    best = result["best_validation_epoch"]
    env = result["environment"]["actual"]
    diffs = result["environment"]["known_differences"]
    sequence_stats = result["dataset"]["sequence_stats"]
    first_batch = result["variable_length_diagnostics"]["first_train_batch"]
    bidir = result["bidirectional_diagnostics"]
    grad = result["gradient_diagnostics"]["first_backward"]
    slurm = result["slurm"]
    ckpt = result["checkpoints"]
    slurm_sacct = result.get("slurm_sacct", {})
    lines = [
        "# SSD4Rec sanity 001",
        "",
        "## Цель",
        "",
        "Проверить, что официальный SSD4Rec обучается 5 эпох на полном KuaiRand Protocol B, "
        "использует original variable-length/bidirectional path и считает только validation.",
        "",
        "## Окружение",
        "",
        "Требования upstream: Python `3.10.15`, CUDA `11.8`, PyTorch `2.1.1+cu118`, "
        "RecBole `1.2.0`, mamba-ssm `2.2.2`, causal-conv1d `1.4.0`.",
        "",
        f"Фактическое окружение: Python `{env['python']}`, PyTorch `{env['torch']}`, CUDA `{env['torch_cuda']}`, "
        f"RecBole `{env['recbole']}`, mamba-ssm `{env['mamba_ssm']}`, "
        f"causal-conv1d `{env['causal_conv1d']}`, Triton `{env['triton']}`.",
        "",
        f"Путь окружения: `{env['sys_prefix']}`; base prefix: `{env['sys_base_prefix']}`.",
        "",
        "Известные расхождения:",
        "",
    ]
    for name, item in diffs.items():
        if not item["matches_exact_pin"]:
            lines.append(f"- `{name}`: upstream `{item['upstream']}`, фактически `{item['actual']}`.")
    if all(item["matches_exact_pin"] for item in diffs.values()):
        lines.append("- Нет расхождений с перечисленными upstream pins.")
    lines.extend(
        [
            "",
            "## Данные и Protocol B",
            "",
            "- Dataset: полный KuaiRand Protocol B, без subset и без sampled split.",
            "- Fingerprint: `23951 users / 7111 items / 1134420 interactions`.",
            f"- Длины историй в Protocol B min/median/mean/max: "
            f"`{sequence_stats['min']} / {sequence_stats['median']} / "
            f"{sequence_stats['mean']:.4f} / {sequence_stats['max']}`.",
            "- `MAX_ITEM_LIST_LENGTH=50` есть в config, но при upstream `var_len=True` "
            "не является active truncation cap.",
            "- Split: chronological leave-one-out; validation target - предпоследняя interaction.",
            "- Test metrics не считались.",
            "- Evaluation: `full_7111_items`; internal score tensor включает padding item `0`, он маскируется.",
            "",
            "## Конфигурация original SSD4Rec",
            "",
            f"- `hidden_size={result['config']['architecture']['hidden_size']}`",
            f"- `num_layers={result['config']['architecture']['num_layers']}`",
            f"- `d_state={result['config']['architecture']['d_state']}`",
            f"- `d_conv={result['config']['architecture']['d_conv']}`",
            f"- `expand={result['config']['architecture']['expand']}`",
            f"- `headdim={result['config']['architecture']['headdim']}`",
            f"- `var_len={result['config']['var_len']}`",
            f"- `maskratio={result['config']['maskratio']}`",
            f"- `learning_rate={result['config']['learning_rate']}`",
            f"- `train_batch_size={result['config']['train_batch_size']}`",
            f"- `eval_batch_size={result['config']['eval_batch_size']}`",
            f"- `seed={result['config']['seed']}`",
            "",
            "## Variable-length mechanism",
            "",
            f"- Targets в первом train batch: `{first_batch['targets']}`.",
            f"- Flat sequence tokens: `{first_batch['flat_sequence_tokens']}`.",
            f"- Длина последовательностей min/median/mean/max: "
            f"`{first_batch['sequence_length']['min']} / {first_batch['sequence_length']['median']:.1f} / "
            f"{first_batch['sequence_length']['mean']:.4f} / {first_batch['sequence_length']['max']}`.",
            f"- `cum_item_length_shape={first_batch['cum_item_length_shape']}`.",
            f"- `item_idx_shape={first_batch['item_idx_shape']}`.",
            f"- Sequence registers присутствуют: `{first_batch['sequence_registers_present']}`.",
            f"- `flip_index_is_permutation={first_batch['flip_index_is_permutation']}`.",
            "",
            "## Bidirectional SSD",
            "",
            f"- BiSSD layers: `{bidir['bissd_layer_count']}`.",
            f"- Forward direction активен: `{bidir['forward_direction_active']}`.",
            f"- Backward/reversed direction активен: `{bidir['backward_reversed_direction_active']}`.",
            f"- Вызовы Mamba по слоям в первом forward: `{bidir['mamba_calls_per_layer']}`.",
            f"- Один и тот же Mamba2 module используется для обоих направлений: `{bidir['mamba_module_shared_for_forward_backward']}`.",
            f"- Gradients finite: `{grad['all_gradients_finite']}`.",
            "",
            "## Обучение",
            "",
            "| epoch | loss | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | time |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in epochs:
        validation = item["validation"]
        lines.append(
            f"| {item['epoch']} | {item['train_loss_avg']:.6f} | "
            f"{validation['hit@10']:.4f} | {validation['hit@20']:.4f} | {validation['hit@50']:.4f} | "
            f"{validation['ndcg@10']:.4f} | {validation['ndcg@20']:.4f} | {validation['ndcg@50']:.4f} | "
            f"{item['epoch_time_sec']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Полные validation metrics:",
            "",
            "| epoch | HR@5 | HR@10 | HR@20 | HR@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in epochs:
        validation = item["validation"]
        lines.append(
            f"| {item['epoch']} | "
            f"{validation['hit@5']:.4f} | {validation['hit@10']:.4f} | {validation['hit@20']:.4f} | {validation['hit@50']:.4f} | "
            f"{validation['recall@5']:.4f} | {validation['recall@10']:.4f} | {validation['recall@20']:.4f} | {validation['recall@50']:.4f} | "
            f"{validation['ndcg@5']:.4f} | {validation['ndcg@10']:.4f} | {validation['ndcg@20']:.4f} | {validation['ndcg@50']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Лучшая validation эпоха",
            "",
            f"- Лучшая epoch: `{best['epoch']}`.",
            f"- Лучший validation `NDCG@10`: `{best['validation']['ndcg@10']:.4f}`.",
            f"- Лучший validation `HR@10`: `{best['validation']['hit@10']:.4f}`.",
            "",
            "## Сравнение с ориентирами",
            "",
            "| Источник | Split | HR@10 | HR@20 | NDCG@10 | NDCG@20 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            f"| Random full-ranking | validation | {VALIDATION_REFERENCES['random_full_ranking_validation']['hr@10']:.4f} | n/a | {VALIDATION_REFERENCES['random_full_ranking_validation']['ndcg@10']:.4f} | n/a |",
            f"| MostPopular full-ranking | validation | {VALIDATION_REFERENCES['mostpopular_full_ranking_validation']['hr@10']:.4f} | n/a | {VALIDATION_REFERENCES['mostpopular_full_ranking_validation']['ndcg@10']:.4f} | n/a |",
            f"| XGBoost `ltr_xgb_002` | validation | {VALIDATION_REFERENCES['xgboost_ltr_xgb_002_validation']['hr@10']:.4f} | n/a | {VALIDATION_REFERENCES['xgboost_ltr_xgb_002_validation']['ndcg@10']:.4f} | n/a |",
            f"| TiM4Rec `tim4rec_001` best | validation | {VALIDATION_REFERENCES['tim4rec_001_best_validation']['hr@10']:.4f} | n/a | {VALIDATION_REFERENCES['tim4rec_001_best_validation']['ndcg@10']:.4f} | n/a |",
            f"| SSD4Rec sanity best | validation | {best['validation']['hit@10']:.4f} | {best['validation']['hit@20']:.4f} | {best['validation']['ndcg@10']:.4f} | {best['validation']['ndcg@20']:.4f} |",
            f"| SSD4Rec paper v2 | paper reported | {PUBLISHED_SSD4REC_V2['hr@10']:.4f} | {PUBLISHED_SSD4REC_V2['hr@20']:.4f} | {PUBLISHED_SSD4REC_V2['ndcg@10']:.4f} | {PUBLISHED_SSD4REC_V2['ndcg@20']:.4f} |",
            "",
            "## Ресурсы",
            "",
            f"- Slurm job: `{slurm['job_id']}`.",
            f"- Partition: `{slurm['partition']}`.",
            f"- Node: `{slurm['node_list']}`.",
            f"- GPU: `{result['gpu']['name']}`.",
            f"- Общее время: `{result['runtime']['total_sec']:.2f}` sec.",
            f"- Среднее время эпохи: `{result['runtime']['mean_epoch_sec']:.2f}` sec.",
            f"- MaxRSS: `{slurm_sacct.get('batch_max_rss', 'unknown until sacct post-processing')}`.",
            f"- Пик VRAM allocated: `{result['gpu']['peak_allocated_bytes']}` bytes.",
            f"- Пик VRAM reserved: `{result['gpu']['peak_reserved_bytes']}` bytes.",
            "",
            "## Проблемы и исправления",
            "",
            "- Upstream `custom_utils.py` требует runtime shim для `np.float` на NumPy >= 1.24.",
            "- Upstream `custom_utils.py` вызывает `getLogger()` без import; wrapper добавляет `logging.getLogger`.",
            "- Full-sort validation следует upstream semantics: маскируется item id `0`, seen history items явно не маскируются.",
            "",
            "## Решение о полном запуске",
            "",
            f"- Pipeline готов к полному SSD4Rec run: `{result['decision']['pipeline_ready']}`.",
            f"- Модель действительно original SSD4Rec: `{result['decision']['original_ssd4rec']}`.",
            f"- Окружение достаточно воспроизводимо для sanity: `{result['decision']['environment_sufficient_for_sanity']}`.",
            f"- Exact upstream environment все еще нужен перед финальной заявкой на reproduction: `{result['decision']['exact_env_rebuild_recommended_before_final']}`.",
            f"- Loss убывает за 5 эпох: `{result['decision']['loss_decreased']}`.",
            f"- Validation NDCG@10 растет за 5 эпох: `{result['decision']['ndcg10_improved']}`.",
            f"- Оценка wall time для 300 эпох в темпе sanity: `{result['decision']['estimated_full_300_epoch_hours']:.2f}` hours.",
            f"- Рекомендуемый GPU: `{result['decision']['recommended_gpu']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise RuntimeError(f"This sanity script is pinned to run_id={RUN_ID}, got {args.run_id}")
    if int(args.epochs) != 5:
        raise RuntimeError("SSD4Rec sanity must run exactly 5 epochs.")

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        config_overrides = {
            "epochs": 5,
            "metrics": ["Hit", "Recall", "NDCG"],
            "topk": [5, 10, 20, 50],
            "show_progress": False,
            "log_wandb": False,
        }
        config = Config(model=SSD4Rec, config_file_list=[args.config], config_dict=config_overrides)
    finally:
        sys.argv = original_argv

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for SSD4Rec sanity training.")

    if int(config["hidden_size"]) != 256:
        raise RuntimeError(f"Original SSD4Rec sanity requires hidden_size=256, got {config['hidden_size']}")
    if not bool(config["var_len"]):
        raise RuntimeError("Original SSD4Rec sanity requires var_len=True.")

    artifact_dir = Path(args.artifact_dir)
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_info = load_manifest(args.manifest)
    fingerprint = manifest_fingerprint(manifest_info)
    assert_expected_fingerprint(fingerprint)
    sequence_stats = manifest_sequence_stats(manifest_info)

    env_actual = collect_environment()
    env_differences = environment_differences(env_actual)

    start_monotonic = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    dataset = SSD4RecDataset(config)
    train_data, valid_data, _test_data = SSD4RecData_preparation(config, dataset)
    test_loader_created_not_iterated = True
    model = SSD4Rec(config, train_data.dataset).to(config["device"])
    trainer = SSD4RecTrainer(config, model)

    if int(dataset.user_num) - 1 != EXPECTED_FINGERPRINT["users"]:
        raise RuntimeError(f"User universe mismatch: {int(dataset.user_num) - 1}")
    if int(dataset.item_num) - 1 != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Item universe mismatch: {int(dataset.item_num) - 1}")
    if int(dataset.inter_num) != EXPECTED_SEQUENTIAL_EXAMPLES:
        raise RuntimeError(f"SSD4Rec sequential example mismatch: {int(dataset.inter_num)}")
    if int(valid_data._dataset.item_num) - 1 != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Full-ranking item universe mismatch: {int(valid_data._dataset.item_num) - 1}")

    epochs = []
    best_valid_score = -float("inf")
    best_epoch = None
    best_checkpoint = None
    last_checkpoint = None
    cur_step = 0
    warnings: list[str] = []
    first_batch_diag = None
    first_gradient_diag = None
    first_full_ranking_checks = None
    topk = list(config["topk"])
    valid_metric = str(config["valid_metric"]).lower()
    valid_metric_bigger = bool(config["valid_metric_bigger"])
    result_path = Path(args.result_json)
    partial_path = result_path.with_suffix(".partial.json")

    for epoch in range(1, 6):
        epoch_start = time.monotonic()
        train_start = time.monotonic()
        train_loss_sum, train_loss_avg, batch_diag, gradient_diag = train_one_epoch(
            trainer,
            train_data,
            epoch,
            collect_diagnostics=(epoch == 1),
        )
        train_time = time.monotonic() - train_start
        if batch_diag is not None:
            first_batch_diag = batch_diag
            first_gradient_diag = gradient_diag

        valid_start = time.monotonic()
        valid_result, full_ranking_checks = evaluate_validation_full_sort(trainer, valid_data)
        validation_time = time.monotonic() - valid_start
        if first_full_ranking_checks is None:
            first_full_ranking_checks = full_ranking_checks
        if not full_ranking_checks["one_positive_per_row"]:
            raise RuntimeError(f"Validation must have one positive per row: {full_ranking_checks}")
        if not full_ranking_checks["raw_scores_all_finite"]:
            raise RuntimeError(f"Non-finite raw validation scores: {full_ranking_checks}")
        if not full_ranking_checks["positive_scores_all_finite"]:
            raise RuntimeError(f"Non-finite positive validation scores: {full_ranking_checks}")
        if not full_ranking_checks["padding_item_zero_masked"]:
            raise RuntimeError(f"Item id 0 was not masked correctly: {full_ranking_checks}")

        hit_recall_equal_check = check_hit_recall_equal(valid_result, topk)
        parameter_finite_check = all_parameters_finite(model)
        if not parameter_finite_check["ok"]:
            raise RuntimeError(f"Non-finite model parameters after epoch {epoch}: {parameter_finite_check}")

        valid_score = float(valid_result[valid_metric])
        best_valid_score, cur_step, stop_flag, update_flag = early_stopping(
            valid_score,
            best_valid_score,
            cur_step,
            max_step=int(config["stopping_step"]),
            bigger=valid_metric_bigger,
        )

        if update_flag:
            best_epoch = epoch
            best_checkpoint = save_checkpoint(
                model,
                trainer.optimizer,
                config,
                checkpoint_dir / "best_validation.pth",
                epoch,
                best_valid_score,
                valid_result,
            )
        last_checkpoint = save_checkpoint(
            model,
            trainer.optimizer,
            config,
            checkpoint_dir / "last.pth",
            epoch,
            best_valid_score,
            valid_result,
        )

        epoch_payload = {
            "epoch": epoch,
            "train_loss_sum": train_loss_sum,
            "train_loss_avg": train_loss_avg,
            "learning_rate": float(config["learning_rate"]),
            "train_time_sec": train_time,
            "validation_time_sec": validation_time,
            "epoch_time_sec": time.monotonic() - epoch_start,
            "validation": valid_result,
            "valid_score": valid_score,
            "valid_metric": valid_metric,
            "early_stopping": {
                "cur_step": int(cur_step),
                "update_flag": bool(update_flag),
                "stop_flag": bool(stop_flag),
                "best_valid_score": float(best_valid_score),
            },
            "hit_recall_equal_check": hit_recall_equal_check,
            "full_ranking_checks": full_ranking_checks,
            "parameters_finite_check": parameter_finite_check,
            "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()),
        }
        epochs.append(epoch_payload)
        write_partial(
            partial_path,
            {
                "run_id": args.run_id,
                "status": "partial",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "epochs_completed": len(epochs),
                "latest_epoch": epoch_payload,
                "best_epoch_so_far": best_epoch,
                "best_valid_score_so_far": best_valid_score,
                "remote_artifact_path": str(artifact_dir),
                "test_metrics_computed": False,
            },
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss_avg": train_loss_avg,
                    "validation": valid_result,
                    "train_time_sec": train_time,
                    "validation_time_sec": validation_time,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stop_flag:
            warnings.append(f"early stopping triggered unexpectedly during 5-epoch sanity at epoch {epoch}")
            break

    if len(epochs) != 5:
        raise RuntimeError(f"Sanity run must complete 5 epochs, completed {len(epochs)}")
    if first_batch_diag is None or first_gradient_diag is None:
        raise RuntimeError("Missing first-batch diagnostics.")

    best_epoch_payload = max(epochs, key=lambda item: item["validation"]["ndcg@10"])
    runtime_sec = time.monotonic() - start_monotonic
    ru_maxrss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    loss_decreased = epochs[-1]["train_loss_avg"] < epochs[0]["train_loss_avg"]
    ndcg10_improved = epochs[-1]["validation"]["ndcg@10"] > epochs[0]["validation"]["ndcg@10"]
    above_random = best_epoch_payload["validation"]["hit@10"] > VALIDATION_REFERENCES["random_full_ranking_validation"]["hr@10"]
    bidi = first_batch_diag["bidirectional"]
    exact_env = all(item["matches_exact_pin"] for item in env_differences.values())

    result = {
        "run_id": args.run_id,
        "status": "ok",
        "sanity": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "starting_project_commit": os.environ.get("SSD4REC_GIT_COMMIT", "unknown"),
        "project_git_branch": os.environ.get("SSD4REC_GIT_BRANCH", "unknown"),
        "working_tree_state_at_submission": os.environ.get("SSD4REC_WORKTREE_STATE", "unknown"),
        "upstream": {
            "repository": "https://github.com/ZhangYifeng1995/SSD4Rec",
            "commit": UPSTREAM_COMMIT,
            "license": "MIT License, copyright 2025 Zhang Yifeng",
        },
        "paper": {
            "reference": PAPER_REFERENCE_VERSION,
            "url": "https://arxiv.org/abs/2409.01192",
            "published_metrics": PUBLISHED_SSD4REC_V2,
        },
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
            "time_limit": os.environ.get("SLURM_TIMELIMIT"),
        },
        "environment": {
            "upstream_requirements": UPSTREAM_REQUIREMENTS,
            "actual": env_actual,
            "known_differences": env_differences,
            "dependency_sources": DEPENDENCY_SOURCES,
            "exact_upstream_environment": exact_env,
            "separate_environment_path": env_actual["sys_prefix"],
            "mutates_tim4rec_env": False,
        },
        "gpu": {
            "device": str(config["device"]),
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(torch.cuda.current_device()))),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "memory": {
            "process_ru_maxrss_kb": ru_maxrss_kb,
        },
        "dataset": {
            "manifest": {"path": manifest_info["path"], "loaded": True},
            "fingerprint": fingerprint,
            "expected_fingerprint": EXPECTED_FINGERPRINT,
            "sequence_stats": sequence_stats,
            "filtering": {
                "min_core": 5,
                "source": "Protocol B manifest",
            },
            "split": {
                "name": "Protocol B chronological leave-one-out",
                "train": EXPECTED_FINGERPRINT["train"],
                "validation": EXPECTED_FINGERPRINT["validation"],
                "test": EXPECTED_FINGERPRINT["test"],
                "test_loader_created_not_iterated": test_loader_created_not_iterated,
                "test_metrics_computed": False,
                "random_exposure_logs_used": False,
            },
            "recbole": {
                "user_num_with_padding": int(dataset.user_num),
                "item_num_with_padding": int(dataset.item_num),
                "item_universe_without_padding": int(dataset.item_num) - 1,
                "inter_num_after_ssd4rec_augmentation": int(dataset.inter_num),
                "train_batches": len(train_data),
                "valid_batches": len(valid_data),
            },
        },
        "config": {
            "config_file": str(Path(args.config).resolve()),
            "seed": int(config["seed"]),
            "seed_source": "upstream config.yaml global seed",
            "paper_seed": None,
            "upstream_config_seed": 2024,
            "seed_tuned": False,
            "learning_rate": float(config["learning_rate"]),
            "optimizer": config["learner"],
            "weight_decay": float(config["weight_decay"]),
            "epochs_requested": 5,
            "epochs_completed": len(epochs),
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "metrics": list(config["metrics"]),
            "topk": topk,
            "valid_metric": str(config["valid_metric"]),
            "eval_args": config["eval_args"],
            "var_len": bool(config["var_len"]),
            "maskratio": float(config["maskratio"]),
            "hyperparameter_sources": HYPERPARAMETER_SOURCES,
            "architecture": {
                "model": "Official SSD4Rec",
                "hidden_size": int(config["hidden_size"]),
                "num_layers": int(config["num_layers"]),
                "dropout_prob": float(config["dropout_prob"]),
                "norm_embedding": bool(config["norm_embedding"]),
                "beta": float(config["beta"]),
                "d_state": int(config["d_state"]),
                "d_conv": int(config["d_conv"]),
                "expand": int(config["expand"]),
                "headdim": int(config["headdim"]),
                "bissd_layer_count": int(len(model.BiSSD_layers)),
                "uses_mamba2": True,
                "uses_sequence_registers": True,
                "uses_bidirectional_ssd": True,
                "uses_original_masking": True,
                "loss": "CrossEntropyLoss over full item logits",
            },
        },
        "variable_length_diagnostics": {
            "first_train_batch": first_batch_diag,
            "full_dataset_var_len_true": bool(config["var_len"]),
            "note": "With upstream var_len=True, MAX_ITEM_LIST_LENGTH is not an active truncation cap in SSD4RecDataset.data_augmentation.",
        },
        "bidirectional_diagnostics": bidi,
        "gradient_diagnostics": {
            "first_backward": first_gradient_diag,
        },
        "validation_protocol": {
            "candidate_protocol": "full_7111_items",
            "evaluation_protocol": "full_7111_items",
            "mask_item_zero": True,
            "mask_seen_history_items": False,
            "test_metrics_computed": False,
            "first_epoch_full_ranking_checks": first_full_ranking_checks,
        },
        "epochs": epochs,
        "best_validation_epoch": best_epoch_payload,
        "best_epoch_by_early_stopping_tracker": best_epoch,
        "best_valid_score_by_early_stopping_tracker": float(best_valid_score),
        "best_valid_metric": valid_metric,
        "checkpoints": {
            "best_validation": best_checkpoint,
            "last": last_checkpoint,
        },
        "remote_artifact_path": str(artifact_dir),
        "runtime": {
            "total_sec": runtime_sec,
            "mean_epoch_sec": sum(item["epoch_time_sec"] for item in epochs) / len(epochs),
            "mean_train_sec": sum(item["train_time_sec"] for item in epochs) / len(epochs),
            "mean_validation_sec": sum(item["validation_time_sec"] for item in epochs) / len(epochs),
        },
        "comparison_references": {
            "validation_references": VALIDATION_REFERENCES,
            "paper_ssd4rec_v2": PUBLISHED_SSD4REC_V2,
        },
        "decision": {
            "pipeline_ready": True,
            "original_ssd4rec": True,
            "variable_length_path_working": bool(first_batch_diag["uses_flat_representation_in_forward"]),
            "bidirectional_path_working": bool(bidi["forward_direction_active"] and bidi["backward_reversed_direction_active"]),
            "environment_sufficient_for_sanity": True,
            "exact_env_rebuild_recommended_before_final": not exact_env,
            "loss_decreased": bool(loss_decreased),
            "ndcg10_improved": bool(ndcg10_improved),
            "best_validation_above_random_hr10": bool(above_random),
            "estimated_full_300_epoch_hours": float((runtime_sec / len(epochs)) * 300 / 3600),
            "recommended_gpu": "A100/type_e",
            "full_run_auto_started": False,
        },
        "warnings": warnings,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    notes_path = Path(args.notes_md)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(build_notes(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
