#!/usr/bin/env python
"""Validation-only Optuna tuning for MultitaskTiM4Rec loss configuration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import optuna
import torch
import yaml
from optuna.trial import TrialState
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.data.utils import get_dataloader
from recbole.trainer import Trainer
from recbole.utils import init_seed


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tim4rec import TiM4Rec  # noqa: E402
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS  # noqa: E402
from experiments.multitask_tim4rec.train import (  # noqa: E402
    EXPECTED_FINGERPRINT,
    EXPECTED_IDENTITY_HASH,
    all_gradient_check,
    backbone_parameters,
    check_hit_recall_equal,
    count_parameters,
    evaluate_auxiliary,
    evaluate_full_sort_with_checks,
    first_batch,
    format_float,
    grad_norm,
    inspect_eval_loader,
    load_json,
    load_target_stats,
    metric_subset,
    named_head_parameters,
    sha256_file,
    tensor_to_float,
)


COMMON_TARGETS = ("is_click", "long_view")
RARE_TARGETS = ("is_like", "is_profile_enter")
TRIAL_STATE_COMPLETE = "COMPLETE"
TRIAL_STATE_PRUNED = "PRUNED"
TRIAL_STATE_FAIL = "FAIL"


@dataclass
class DataBundle:
    base_config: Config
    full_dataset: Any
    train_dataset: Any
    valid_dataset: Any
    validation_only_summary: dict[str, Any]
    loader_inspection: dict[str, Any]
    target_stats: dict[str, dict[str, float]]
    base_params: dict[str, int]
    multitask_params: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments/multitask_tim4rec_optuna/config.yaml"))
    parser.add_argument("--search-space", default=str(ROOT / "experiments/multitask_tim4rec_optuna/search_space.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--target-complete", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path}"


def git_value(args: list[str], default: str = "unknown") -> str:
    env_map = {
        ("rev-parse", "HEAD"): "MULTITASK_OPTUNA_GIT_COMMIT",
        ("rev-parse", "--abbrev-ref", "HEAD"): "MULTITASK_OPTUNA_GIT_BRANCH",
    }
    env_key = env_map.get(tuple(args))
    if env_key and os.environ.get(env_key):
        return str(os.environ[env_key])
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def slurm_info() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
        "hostname": socket.gethostname(),
    }


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "recbole": version("recbole"),
        "optuna": optuna.__version__,
        "sqlalchemy": version("SQLAlchemy"),
        "alembic": version("alembic"),
        "mamba_ssm": version("mamba-ssm"),
        "causal_conv1d": version("causal-conv1d"),
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
    }


def merge_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def assert_protocol_config(optuna_config: dict[str, Any]) -> None:
    protocol = optuna_config["protocol"]
    expected = {
        "users": EXPECTED_FINGERPRINT["users"],
        "items": EXPECTED_FINGERPRINT["items"],
        "interactions": EXPECTED_FINGERPRINT["interactions"],
        "train": EXPECTED_FINGERPRINT["train"],
        "validation": EXPECTED_FINGERPRINT["validation"],
        "test": EXPECTED_FINGERPRINT["test"],
        "identity_hash": EXPECTED_IDENTITY_HASH,
    }
    observed = {key: protocol[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"Protocol config mismatch: {observed}")
    test_policy = optuna_config["test_policy"]
    if any(bool(test_policy[key]) for key in ("load_test_dataset", "create_test_dataloader", "evaluate_test")):
        raise RuntimeError(f"Optuna search must keep test closed: {test_policy}")
    if int(test_policy["test_evaluation_count"]) != 0:
        raise RuntimeError(f"test_evaluation_count must be 0: {test_policy}")


def assert_validation_only_summary(summary: dict[str, Any]) -> None:
    if summary.get("forbidden_test_paths_loaded") != []:
        raise RuntimeError(f"Forbidden test paths were loaded during validation-only prep: {summary}")
    if bool(summary.get("test_path_passed_to_search")):
        raise RuntimeError(f"Test path passed to validation-only prep: {summary}")
    if int(summary["rows"]["test"]) != 0 or int(summary["test_rows_in_inter_file"]) != 0:
        raise RuntimeError(f"Validation-only RecBole .inter contains test rows: {summary['rows']}")
    if int(summary.get("test_rows_in_benchmark_file", 0)) != 0:
        raise RuntimeError(f"Validation-only benchmark test file is not empty: {summary.get('test_rows_in_benchmark_file')}")
    if int(summary["rows"]["train"]) != EXPECTED_FINGERPRINT["train"]:
        raise RuntimeError(f"Train rows changed: {summary['rows']}")
    if int(summary["rows"]["validation"]) != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Validation rows changed: {summary['rows']}")
    sequential = summary.get("sequential_examples", {})
    expected_train_examples = EXPECTED_FINGERPRINT["train"] - EXPECTED_FINGERPRINT["users"]
    if int(sequential.get("train", -1)) != expected_train_examples:
        raise RuntimeError(f"Sequential train examples changed: {sequential}")
    if int(sequential.get("validation", -1)) != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Sequential validation examples changed: {sequential}")
    if int(summary["items_sidecar_rows"]) != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Item sidecar does not preserve full universe: {summary['items_sidecar_rows']}")
    loaded = json.dumps(summary.get("loaded_source_paths", {}), ensure_ascii=False).lower()
    if "test.parquet" in loaded:
        raise RuntimeError(f"Validation-only prep loaded test parquet: {loaded}")


def recbole_overrides(optuna_config: dict[str, Any], artifact_root: Path, sampled: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides = dict(optuna_config["recbole_overrides"])
    overrides["checkpoint_dir"] = str(artifact_root / "recbole_checkpoints")
    overrides["epochs"] = int(optuna_config["trial"]["max_epochs"])
    overrides["stopping_step"] = int(optuna_config["trial"]["early_stopping_patience"])
    overrides["final_test_evaluation_count"] = 0
    overrides["test_evaluation_count"] = 0
    if sampled:
        overrides["learning_rate"] = float(sampled["learning_rate"])
        overrides["weight_decay"] = float(sampled["weight_decay"])
        overrides["dropout_prob"] = float(sampled["dropout_prob"])
    return overrides


def build_config(optuna_config: dict[str, Any], artifact_root: Path, sampled: dict[str, Any] | None = None) -> Config:
    base_config = project_path(optuna_config["source"]["base_config"])
    return Config(
        model=MultitaskTiM4Rec,
        config_file_list=[str(base_config)],
        config_dict=recbole_overrides(optuna_config, artifact_root, sampled),
    )


def validation_source_ids(summary: dict[str, Any]) -> set[int]:
    path = Path(summary["validation_source_row_ids_path"])
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def create_loaders(config: Config, train_dataset: Any, valid_dataset: Any) -> tuple[Any, FullSortEvalDataLoader]:
    train_loader = get_dataloader(config, "train")(config, train_dataset, None, shuffle=config["shuffle"])
    valid_loader = get_dataloader(config, "valid")(config, valid_dataset, None, shuffle=False)
    if not isinstance(valid_loader, FullSortEvalDataLoader):
        raise RuntimeError(f"Expected FullSortEvalDataLoader, got {type(valid_loader).__name__}")
    return train_loader, valid_loader


def load_data_bundle(optuna_config: dict[str, Any], artifact_root: Path) -> DataBundle:
    summary_path = Path(optuna_config["validation_only_data"]["summary_json"])
    if not summary_path.exists():
        raise FileNotFoundError(f"Validation-only dataset summary is missing: {summary_path}")
    summary = load_json(summary_path)
    assert_validation_only_summary(summary)

    target_stats = load_target_stats(project_path(optuna_config["source"]["target_statistics"]))
    base_config = build_config(optuna_config, artifact_root)
    init_seed(base_config["seed"] + base_config["local_rank"], base_config["reproducibility"])

    full_dataset = create_dataset(base_config)
    built = full_dataset.build()
    if len(built) == 2:
        train_dataset, valid_dataset = built
    elif len(built) == 3:
        train_dataset, valid_dataset, unused_dataset = built
        if len(unused_dataset) != 0:
            raise RuntimeError(f"Validation-only RecBole split created non-empty unused split: {len(unused_dataset)}")
    else:
        raise RuntimeError(f"Expected train/valid validation-only splits from RecBole, got {len(built)}")

    train_loader, valid_loader = create_loaders(base_config, train_dataset, valid_dataset)
    expected_ids = validation_source_ids(summary)
    inspection = inspect_eval_loader(valid_loader, int(valid_loader._dataset.item_num), expected_ids)
    if not inspection["one_positive_per_row"]:
        raise RuntimeError(f"Validation split must have one positive per row: {inspection}")
    if not inspection["positive_targets_within_item_universe"]:
        raise RuntimeError(f"Validation positives outside item universe: {inspection}")
    if int(valid_loader._dataset.item_num) - 1 != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Full-ranking item universe changed: {int(valid_loader._dataset.item_num) - 1}")
    if len(train_dataset) != EXPECTED_FINGERPRINT["train"] - EXPECTED_FINGERPRINT["users"]:
        raise RuntimeError(f"Sequential train examples changed: {len(train_dataset)}")
    if len(valid_dataset) != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Validation examples changed: {len(valid_dataset)}")

    device = base_config["device"]
    init_seed(base_config["seed"] + base_config["local_rank"], base_config["reproducibility"])
    mt_model = MultitaskTiM4Rec(base_config, train_dataset).to(device)
    base_model = TiM4Rec(base_config, train_dataset).to(device)
    base_params = count_parameters(base_model)
    multitask_params = count_parameters(mt_model)
    del mt_model, base_model, train_loader, valid_loader
    torch.cuda.empty_cache()

    return DataBundle(
        base_config=base_config,
        full_dataset=full_dataset,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        validation_only_summary=summary,
        loader_inspection=inspection,
        target_stats=target_stats,
        base_params=base_params,
        multitask_params=multitask_params,
    )


def suggest_float(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> float:
    return float(trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))))


def sample_trial_params(trial: optuna.Trial, search_space: dict[str, Any], target_stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    specs = search_space["parameters"]
    raw_params = {name: suggest_float(trial, name, spec) for name, spec in specs.items()}
    raw_weights = {
        "is_click": raw_params["w_click_raw"],
        "long_view": raw_params["w_long_view_raw"],
        "is_like": raw_params["w_like_raw"],
        "is_profile_enter": raw_params["w_profile_raw"],
    }
    mean_weight = sum(raw_weights.values()) / len(raw_weights)
    task_weights = {target: value / mean_weight for target, value in raw_weights.items()}
    alpha_by_target = {
        "is_click": raw_params["alpha_common"],
        "long_view": raw_params["alpha_common"],
        "is_like": raw_params["alpha_rare"],
        "is_profile_enter": raw_params["alpha_rare"],
    }
    raw_pos_weights = {target: float(target_stats[target]["negative_positive_ratio"]) for target in TARGETS}
    effective_pos_weights = {
        target: raw_pos_weights[target] ** alpha_by_target[target]
        for target in TARGETS
    }
    lambda_aux = raw_params["lambda_aux"]
    effective_loss_multipliers = {target: lambda_aux * task_weights[target] for target in TARGETS}
    effective_positive_multipliers = {
        target: effective_loss_multipliers[target] * effective_pos_weights[target]
        for target in TARGETS
    }
    return {
        "raw_params": raw_params,
        "lambda_aux": lambda_aux,
        "learning_rate": raw_params["learning_rate"],
        "weight_decay": raw_params["weight_decay"],
        "dropout_prob": raw_params["dropout_prob"],
        "head_lr_multiplier": raw_params["head_lr_multiplier"],
        "head_learning_rate": raw_params["learning_rate"] * raw_params["head_lr_multiplier"],
        "raw_task_weights": raw_weights,
        "normalized_task_weights": task_weights,
        "task_weight_normalization": "mean_equals_one",
        "alpha_common": raw_params["alpha_common"],
        "alpha_rare": raw_params["alpha_rare"],
        "alpha_by_target": alpha_by_target,
        "raw_pos_weights": raw_pos_weights,
        "effective_pos_weights": effective_pos_weights,
        "effective_loss_multipliers": effective_loss_multipliers,
        "effective_positive_multipliers": effective_positive_multipliers,
    }


def pos_weight_tensors(effective_pos_weights: dict[str, float], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        target: torch.tensor([float(effective_pos_weights[target])], dtype=torch.float32, device=device)
        for target in TARGETS
    }


def head_parameter_ids(model: MultitaskTiM4Rec) -> set[int]:
    result: set[int] = set()
    for params in named_head_parameters(model).values():
        for _name, param in params:
            result.add(id(param))
    return result


def optimizer_for_trial(model: MultitaskTiM4Rec, sampled: dict[str, Any]) -> torch.optim.Optimizer:
    head_ids = head_parameter_ids(model)
    backbone_params = [param for param in model.parameters() if param.requires_grad and id(param) not in head_ids]
    head_params = [param for param in model.parameters() if param.requires_grad and id(param) in head_ids]
    if not backbone_params or not head_params:
        raise RuntimeError("Expected both backbone and head parameter groups.")
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": float(sampled["learning_rate"]), "name": "backbone"},
            {"params": head_params, "lr": float(sampled["head_learning_rate"]), "name": "auxiliary_heads"},
        ],
        weight_decay=float(sampled["weight_decay"]),
    )


def compute_tuned_losses(
    model: MultitaskTiM4Rec,
    interaction: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    seq_output = model.shared_representation(interaction)
    pos_items = interaction[model.POS_ITEM_ID]
    rank_logits = model.ranking_logits_from_representation(seq_output)
    rank_loss = model.loss_fct(rank_logits, pos_items)

    aux_logits = model.auxiliary_logits_from_representation(seq_output)
    aux_losses = {}
    weighted_aux = torch.zeros((), dtype=rank_loss.dtype, device=rank_loss.device)
    unweighted_aux = torch.zeros((), dtype=rank_loss.dtype, device=rank_loss.device)
    for target, logits in aux_logits.items():
        labels = interaction[target].float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weights[target].to(logits.device),
        )
        aux_losses[target] = loss
        unweighted_aux = unweighted_aux + loss
        weighted_aux = weighted_aux + float(sampled["normalized_task_weights"][target]) * loss
    total = rank_loss + float(sampled["lambda_aux"]) * weighted_aux
    return {
        "total": total,
        "rank": rank_loss,
        "aux_sum": unweighted_aux,
        "weighted_aux_sum": weighted_aux,
        **{f"{target}_loss": loss for target, loss in aux_losses.items()},
        **{
            f"{target}_scaled_contribution": float(sampled["lambda_aux"])
            * float(sampled["normalized_task_weights"][target])
            * loss
            for target, loss in aux_losses.items()
        },
    }


def train_one_epoch_tuned(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    device: torch.device,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.train()
    keys = [
        "total",
        "rank",
        "aux_sum",
        "weighted_aux_sum",
        "is_click_loss",
        "long_view_loss",
        "is_like_loss",
        "is_profile_enter_loss",
        "is_click_scaled_contribution",
        "long_view_scaled_contribution",
        "is_like_scaled_contribution",
        "is_profile_enter_scaled_contribution",
    ]
    sums = {key: 0.0 for key in keys}
    examples = 0
    batches = 0
    for interaction in train_data:
        interaction = interaction.to(device)
        batch_size = len(interaction)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_tuned_losses(model, interaction, sampled, pos_weights)
        loss = losses["total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss in batch {batches}: {tensor_to_float(loss)}")
        loss.backward()
        if not all_gradient_check(model)["all_finite"]:
            raise RuntimeError(f"Non-finite gradients in train batch {batches}")
        optimizer.step()
        for key in keys:
            sums[key] += tensor_to_float(losses[key]) * batch_size
        examples += batch_size
        batches += 1
    if examples == 0:
        raise RuntimeError("No training examples.")
    result = {key: value / examples for key, value in sums.items()}
    result["auxiliary_scaled_contribution"] = float(sampled["lambda_aux"]) * result["weighted_aux_sum"]
    result["auxiliary_rank_ratio"] = result["auxiliary_scaled_contribution"] / result["rank"]
    result["per_task_rank_ratio"] = {
        target: result[f"{target}_scaled_contribution"] / result["rank"]
        for target in TARGETS
    }
    result["batches"] = batches
    result["examples"] = examples
    return result


def early_gradient_diagnostic(
    model: MultitaskTiM4Rec,
    interaction: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    rank_losses = compute_tuned_losses(model, interaction, sampled, pos_weights)
    rank_losses["rank"].backward()
    ranking_norm = grad_norm(backbone_parameters(model))
    ranking_all_finite = all_gradient_check(model)["all_finite"]

    model.zero_grad(set_to_none=True)
    aux_losses = compute_tuned_losses(model, interaction, sampled, pos_weights)
    scaled_aux = float(sampled["lambda_aux"]) * aux_losses["weighted_aux_sum"]
    scaled_aux.backward()
    auxiliary_norm = grad_norm(backbone_parameters(model))
    auxiliary_all_finite = all_gradient_check(model)["all_finite"]

    ratio = None
    if ranking_norm is not None and ranking_norm > 0 and auxiliary_norm is not None:
        ratio = auxiliary_norm / ranking_norm

    model.zero_grad(set_to_none=True)
    return {
        "batch_size": len(interaction),
        "definition": "backbone gradient norm from lambda_aux * weighted auxiliary loss divided by backbone gradient norm from ranking loss",
        "ranking_gradient_norm": ranking_norm,
        "aggregate_auxiliary_gradient_norm": auxiliary_norm,
        "aux_gradient_ratio": ratio,
        "ranking_gradient_finite": ranking_all_finite,
        "auxiliary_gradient_finite": auxiliary_all_finite,
        "first_batch_losses": {key: tensor_to_float(value) for key, value in aux_losses.items()},
    }


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in (5, 10, 20, 50):
        hit = metrics.get(f"HR@{k}", metrics.get(f"hit@{k}", metrics.get(f"hr@{k}")))
        recall = metrics.get(f"Recall@{k}", metrics.get(f"recall@{k}"))
        ndcg = metrics.get(f"NDCG@{k}", metrics.get(f"ndcg@{k}"))
        if hit is None or recall is None or ndcg is None:
            raise KeyError(f"Missing @{k} metrics in {metrics}")
        result[f"HR@{k}"] = float(hit)
        result[f"Recall@{k}"] = float(recall)
        result[f"NDCG@{k}"] = float(ndcg)
    return result


def compact_validation(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "HR@10": float(metrics["HR@10"]),
        "HR@20": float(metrics["HR@20"]),
        "HR@50": float(metrics["HR@50"]),
        "NDCG@10": float(metrics["NDCG@10"]),
        "NDCG@20": float(metrics["NDCG@20"]),
        "NDCG@50": float(metrics["NDCG@50"]),
    }


def trial_json_path(artifact_root: Path, trial_number: int) -> Path:
    return artifact_root / "trials" / f"trial_{trial_number:04d}.json"


def trial_summary(result: dict[str, Any]) -> dict[str, Any]:
    best = result.get("best_validation_metrics") or {}
    return {
        "trial_number": result["trial_number"],
        "state": result["state"],
        "value": result.get("value"),
        "best_epoch": result.get("best_epoch"),
        "actual_epochs": result.get("actual_epochs"),
        "runtime_sec": result.get("runtime_sec"),
        "validation_metrics": best,
        "validation_compact": compact_validation(best) if best else {},
        "params": result["params"],
        "normalized_task_weights": result["normalized_task_weights"],
        "effective_pos_weights": result["effective_pos_weights"],
        "effective_loss_multipliers": result["effective_loss_multipliers"],
        "effective_positive_multipliers": result["effective_positive_multipliers"],
        "head_lr_multiplier": result["params"]["head_lr_multiplier"],
        "gradient_diagnostic": result.get("gradient_diagnostic"),
        "auxiliary_metrics": result.get("best_auxiliary_metrics"),
        "best_epoch_losses": result.get("best_epoch_losses"),
        "stop_reason": result.get("stop_reason"),
        "gpu": result.get("gpu"),
    }


def train_trial(
    trial: optuna.Trial,
    optuna_config: dict[str, Any],
    search_space: dict[str, Any],
    data: DataBundle,
    artifact_root: Path,
) -> float:
    trial_dir = artifact_root / "trials" / f"trial_{trial.number:04d}_artifacts"
    trial_dir.mkdir(parents=True, exist_ok=True)
    sampled = sample_trial_params(trial, search_space, data.target_stats)
    trial_config = build_config(optuna_config, trial_dir, sampled)
    init_seed(trial_config["seed"] + trial_config["local_rank"], trial_config["reproducibility"])
    if tuple(trial_config["multitask_targets"]) != TARGETS:
        raise RuntimeError(f"Task set changed: {trial_config['multitask_targets']}")
    if not bool(trial_config["is_time"]):
        raise RuntimeError("TiM4Rec is_time must stay True.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for MultitaskTiM4Rec Optuna search.")

    train_data, valid_data = create_loaders(trial_config, data.train_dataset, data.valid_dataset)
    device = trial_config["device"]
    torch.cuda.reset_peak_memory_stats()
    init_seed(trial_config["seed"] + trial_config["local_rank"], trial_config["reproducibility"])
    model = MultitaskTiM4Rec(trial_config, train_data.dataset).to(device)
    optimizer = optimizer_for_trial(model, sampled)
    trainer = Trainer(trial_config, model)
    trainer.optimizer = optimizer
    pos_weights = pos_weight_tensors(sampled["effective_pos_weights"], device)

    first = first_batch(train_data, device)
    gradient = early_gradient_diagnostic(model, first, sampled, pos_weights)
    if not bool(gradient["ranking_gradient_finite"]) or not bool(gradient["auxiliary_gradient_finite"]):
        raise RuntimeError(f"Non-finite early gradients: {gradient}")

    topk = list(trial_config["topk"])
    max_epochs = int(optuna_config["trial"]["max_epochs"])
    patience = int(optuna_config["trial"]["early_stopping_patience"])
    min_delta = float(optuna_config["trial"].get("early_stopping_min_delta", 0.0))
    best_score = -float("inf")
    best_epoch = None
    best_snapshot: dict[str, Any] | None = None
    no_improve = 0
    history: list[dict[str, Any]] = []
    stop_reason = "max_epochs_reached"
    started = time.monotonic()

    try:
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.monotonic()
            train_start = time.monotonic()
            losses = train_one_epoch_tuned(model, optimizer, train_data, device, sampled, pos_weights)
            train_time = time.monotonic() - train_start

            validation_start = time.monotonic()
            valid_result, checks = evaluate_full_sort_with_checks(trainer, valid_data, train_data)
            aux_metrics = evaluate_auxiliary(model, valid_data, device)
            validation_time = time.monotonic() - validation_start
            check_hit_recall_equal(valid_result, topk)
            if not checks["raw_scores_all_finite"] or not checks["positive_scores_all_finite"]:
                raise RuntimeError(f"Non-finite validation scores: {checks}")

            metrics = normalize_metrics(metric_subset(valid_result))
            ndcg10 = float(metrics["NDCG@10"])
            improved = ndcg10 > best_score + min_delta
            if improved:
                best_score = ndcg10
                best_epoch = epoch
                no_improve = 0
                best_snapshot = {
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "auxiliary_metrics": aux_metrics,
                    "losses": losses,
                    "full_ranking_checks": checks,
                    "validation_time_sec": validation_time,
                }
            else:
                no_improve += 1

            record = {
                "epoch": epoch,
                "validation_ndcg10": ndcg10,
                "validation_hr10": float(metrics["HR@10"]),
                "validation_ndcg20": float(metrics["NDCG@20"]),
                "validation_ndcg50": float(metrics["NDCG@50"]),
                "losses": losses,
                "train_time_sec": float(train_time),
                "validation_time_sec": float(validation_time),
                "epoch_time_sec": float(time.monotonic() - epoch_start),
                "improved": bool(improved),
                "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()),
                "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()),
            }
            history.append(record)
            trial.report(ndcg10, step=epoch)
            trial.set_user_attr("last_epoch", record)
            trial.set_user_attr("test_evaluation_count", 0)
            if trial.should_prune():
                stop_reason = "optuna_median_pruned"
                result = build_trial_result(
                    trial,
                    TRIAL_STATE_PRUNED,
                    sampled,
                    gradient,
                    best_snapshot,
                    best_epoch,
                    best_score,
                    history,
                    started,
                    stop_reason,
                )
                save_json(trial_json_path(artifact_root, trial.number), result)
                trial.set_user_attr("summary", trial_summary(result))
                raise optuna.exceptions.TrialPruned()
            if no_improve >= patience:
                stop_reason = f"early_stopping_no_validation_ndcg10_improvement_{patience}"
                break
        if best_snapshot is None or best_epoch is None:
            raise RuntimeError("Trial finished without validation evaluation.")
        result = build_trial_result(
            trial,
            TRIAL_STATE_COMPLETE,
            sampled,
            gradient,
            best_snapshot,
            best_epoch,
            best_score,
            history,
            started,
            stop_reason,
        )
        save_json(trial_json_path(artifact_root, trial.number), result)
        trial.set_user_attr("summary", trial_summary(result))
        trial.set_user_attr("validation_hr10", result["best_validation_metrics"]["HR@10"])
        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("actual_epochs", int(len(history)))
        trial.set_user_attr("aux_gradient_ratio", gradient["aux_gradient_ratio"])
        return float(best_score)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception:
        best = best_snapshot if best_snapshot is not None else {}
        result = build_trial_result(
            trial,
            TRIAL_STATE_FAIL,
            sampled,
            gradient,
            best,
            best_epoch,
            best_score if math.isfinite(best_score) else None,
            history,
            started,
            "failed",
        )
        save_json(trial_json_path(artifact_root, trial.number), result)
        trial.set_user_attr("summary", trial_summary(result))
        raise
    finally:
        del model, trainer, optimizer, train_data, valid_data
        torch.cuda.empty_cache()


def build_trial_result(
    trial: optuna.Trial,
    state: str,
    sampled: dict[str, Any],
    gradient: dict[str, Any],
    best_snapshot: dict[str, Any] | None,
    best_epoch: int | None,
    best_score: float | None,
    history: list[dict[str, Any]],
    started: float,
    stop_reason: str,
) -> dict[str, Any]:
    best_snapshot = best_snapshot or {}
    best_metrics = best_snapshot.get("validation_metrics") or {}
    result = {
        "trial_number": int(trial.number),
        "state": state,
        "value": None if best_score is None else float(best_score),
        "objective": "validation_full_ranking_NDCG@10",
        "params": {
            "lambda_aux": sampled["lambda_aux"],
            "learning_rate": sampled["learning_rate"],
            "weight_decay": sampled["weight_decay"],
            "dropout_prob": sampled["dropout_prob"],
            "head_lr_multiplier": sampled["head_lr_multiplier"],
            "head_learning_rate": sampled["head_learning_rate"],
            "alpha_common": sampled["alpha_common"],
            "alpha_rare": sampled["alpha_rare"],
            **sampled["raw_params"],
        },
        "raw_task_weights": sampled["raw_task_weights"],
        "normalized_task_weights": sampled["normalized_task_weights"],
        "task_weight_normalization": sampled["task_weight_normalization"],
        "raw_pos_weights": sampled["raw_pos_weights"],
        "effective_pos_weights": sampled["effective_pos_weights"],
        "effective_loss_multipliers": sampled["effective_loss_multipliers"],
        "effective_positive_multipliers": sampled["effective_positive_multipliers"],
        "best_epoch": best_epoch,
        "actual_epochs": len(history),
        "stop_reason": stop_reason,
        "runtime_sec": float(time.monotonic() - started),
        "mean_epoch_sec": None if not history else float(sum(item["epoch_time_sec"] for item in history) / len(history)),
        "best_validation_metrics": best_metrics,
        "best_validation_compact": compact_validation(best_metrics) if best_metrics else {},
        "best_auxiliary_metrics": best_snapshot.get("auxiliary_metrics"),
        "best_epoch_losses": best_snapshot.get("losses"),
        "full_ranking_checks": best_snapshot.get("full_ranking_checks"),
        "gradient_diagnostic": gradient,
        "history": history,
        "gpu": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(torch.cuda.current_device()))) if torch.cuda.is_available() else None,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        },
        "memory": {
            "process_ru_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "test_evaluation_count": 0,
        "test_dataset_loaded": False,
        "test_dataloader_created": False,
    }
    return result


def storage_for(optuna_config: dict[str, Any]) -> tuple[optuna.storages.RDBStorage, Path, str]:
    storage_path = Path(optuna_config["study_storage"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    url = sqlite_url(storage_path)
    storage = optuna.storages.RDBStorage(
        url=url,
        heartbeat_interval=int(optuna_config["study"]["heartbeat_interval_sec"]),
        grace_period=int(optuna_config["study"]["grace_period_sec"]),
    )
    return storage, storage_path, url


def create_or_load_study(optuna_config: dict[str, Any]) -> tuple[optuna.Study, Path, str, dict[str, Any]]:
    storage, storage_path, url = storage_for(optuna_config)
    sampler = optuna.samplers.TPESampler(seed=int(optuna_config["study"]["sampler_seed"]))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(optuna_config["study"]["n_startup_trials"]),
        n_warmup_steps=int(optuna_config["study"]["n_warmup_steps"]),
        interval_steps=int(optuna_config["study"]["interval_steps"]),
    )
    study = optuna.create_study(
        study_name=optuna_config["study_name"],
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    stale = {"attempted": False, "ok": None, "error": None}
    if hasattr(optuna.storages, "fail_stale_trials"):
        stale["attempted"] = True
        try:
            optuna.storages.fail_stale_trials(study)
            stale["ok"] = True
        except Exception as exc:  # pragma: no cover - depends on storage internals
            stale["ok"] = False
            stale["error"] = repr(exc)
    return study, storage_path, url, stale


def state_counts(study: optuna.Study) -> dict[str, int]:
    counts = {state.name: 0 for state in TrialState}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    return counts


def complete_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]


def summary_from_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    summary = trial.user_attrs.get("summary")
    if summary:
        return dict(summary)
    return {
        "trial_number": int(trial.number),
        "state": trial.state.name,
        "value": None if trial.value is None else float(trial.value),
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "actual_epochs": trial.user_attrs.get("actual_epochs"),
        "params": trial.params,
    }


def top_trials(study: optuna.Study, n: int = 10) -> list[dict[str, Any]]:
    ranked = sorted(complete_trials(study), key=lambda trial: float(trial.value), reverse=True)
    rows = []
    for rank, trial in enumerate(ranked[:n], start=1):
        summary = summary_from_trial(trial)
        params = summary.get("params", trial.params)
        weights = summary.get("normalized_task_weights", {})
        rows.append(
            {
                "rank": rank,
                "trial": int(trial.number),
                "NDCG@10": float(trial.value),
                "HR@10": summary.get("validation_metrics", {}).get("HR@10"),
                "best_epoch": summary.get("best_epoch"),
                "lambda_aux": params.get("lambda_aux"),
                "task_weights": weights,
                "alpha_common": params.get("alpha_common"),
                "alpha_rare": params.get("alpha_rare"),
                "learning_rate": params.get("learning_rate"),
                "weight_decay": params.get("weight_decay"),
                "dropout_prob": params.get("dropout_prob"),
                "head_lr_multiplier": params.get("head_lr_multiplier"),
                "summary": summary,
            }
        )
    return rows


def parameter_importance(study: optuna.Study) -> dict[str, float]:
    if len(complete_trials(study)) < 2:
        return {}
    try:
        return {key: float(value) for key, value in optuna.importance.get_param_importances(study).items()}
    except Exception as exc:
        return {"__error__": repr(exc)}


def validation_refs(optuna_config: dict[str, Any]) -> dict[str, Any]:
    fixed = load_json(project_path(optuna_config["source"]["fixed_multitask_json"]))
    tim = load_json(project_path(optuna_config["source"]["tim4rec_json"]))
    return {
        "fixed_multitask": {
            "run_id": fixed["run_id"],
            "validation_metrics": normalize_metrics(fixed["best_validation_metrics"]),
        },
        "tim4rec": {
            "run_id": tim["run_id"],
            "validation_metrics": normalize_metrics(tim["best_validation_metrics"]),
        },
    }


def compare_validation(base: dict[str, float], fixed: dict[str, float], tuned: dict[str, float]) -> dict[str, Any]:
    def row(name: str, metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "model": name,
            "HR@10": metrics["HR@10"],
            "HR@20": metrics["HR@20"],
            "HR@50": metrics["HR@50"],
            "NDCG@10": metrics["NDCG@10"],
            "NDCG@20": metrics["NDCG@20"],
            "NDCG@50": metrics["NDCG@50"],
        }

    fixed_delta = fixed["NDCG@10"] - base["NDCG@10"]
    tuned_delta = tuned["NDCG@10"] - base["NDCG@10"]
    tuned_vs_fixed = tuned["NDCG@10"] - fixed["NDCG@10"]
    if tuned_delta >= 0:
        status = "removed"
    elif tuned_vs_fixed > 0:
        status = "reduced"
    else:
        status = "remained"
    return {
        "table": [
            row("TiM4Rec", base),
            row("MultitaskTiM4Rec fixed", fixed),
            row("MultitaskTiM4Rec tuned", tuned),
        ],
        "fixed_delta_ndcg10_vs_tim4rec": fixed_delta,
        "tuned_delta_ndcg10_vs_tim4rec": tuned_delta,
        "tuned_delta_ndcg10_vs_fixed": tuned_vs_fixed,
        "negative_transfer_status": status,
        "test_metrics_used": False,
    }


def effective_task_contribution(summary: dict[str, Any]) -> dict[str, Any]:
    losses = summary.get("best_epoch_losses") or {}
    rank = float(losses.get("rank", 0.0) or 0.0)
    result: dict[str, Any] = {
        "definition": "lambda_aux * normalized_task_weight * best_epoch_aux_loss; ratio divides by best_epoch ranking loss",
        "scaled_loss_contribution": {},
        "scaled_loss_to_rank_ratio": {},
        "effective_positive_multipliers": summary.get("effective_positive_multipliers", {}),
    }
    for target in TARGETS:
        value = float(losses.get(f"{target}_scaled_contribution", 0.0) or 0.0)
        result["scaled_loss_contribution"][target] = value
        result["scaled_loss_to_rank_ratio"][target] = None if rank <= 0 else value / rank
    return result


def assert_no_test_metrics(summary: dict[str, Any]) -> None:
    forbidden = {"final_test", "final_test_metrics", "test_metrics", "test_result"}

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).lower()
                if key_lower in forbidden:
                    raise RuntimeError(f"Forbidden test metric key in summary: {'.'.join(path + (str(key),))}")
                walk(child, path + (str(key),))
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, path + (str(i),))

    walk(summary)


def build_search_summary(
    optuna_config: dict[str, Any],
    search_space: dict[str, Any],
    data: DataBundle | None,
    study: optuna.Study,
    storage_path: Path,
    storage_url: str,
    stale: dict[str, Any],
    mode: str,
    invocation_runtime_sec: float,
) -> dict[str, Any]:
    completed = complete_trials(study)
    top10 = top_trials(study, 10)
    best_summary = top10[0]["summary"] if top10 else None
    refs = validation_refs(optuna_config)
    tuned_metrics = best_summary["validation_metrics"] if best_summary else {}
    comparison = (
        compare_validation(refs["tim4rec"]["validation_metrics"], refs["fixed_multitask"]["validation_metrics"], tuned_metrics)
        if best_summary
        else None
    )
    importances = parameter_importance(study)
    trial_summaries = [summary_from_trial(trial) for trial in study.trials]
    total_trial_runtime = sum(float(item.get("runtime_sec") or 0.0) for item in trial_summaries)
    mean_complete_runtime = None
    complete_runtimes = [
        float(item.get("runtime_sec"))
        for item in trial_summaries
        if item.get("state") == TRIAL_STATE_COMPLETE and item.get("runtime_sec") is not None
    ]
    if complete_runtimes:
        mean_complete_runtime = sum(complete_runtimes) / len(complete_runtimes)
    gpu = None
    if best_summary:
        gpu = best_summary.get("gpu")

    summary = {
        "run_id": optuna_config["run_id"] if mode == "search" else optuna_config["smoke_run_id"],
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "starting_commit": optuna_config["base_git_commit"],
        },
        "source_files": {
            "experiments/multitask_tim4rec_optuna/optuna_search.py": sha256_file(
                ROOT / "experiments" / "multitask_tim4rec_optuna" / "optuna_search.py"
            ),
            "experiments/multitask_tim4rec_optuna/prepare_validation_only.py": sha256_file(
                ROOT / "experiments" / "multitask_tim4rec_optuna" / "prepare_validation_only.py"
            ),
            "experiments/multitask_tim4rec_optuna/config.yaml": sha256_file(
                ROOT / "experiments" / "multitask_tim4rec_optuna" / "config.yaml"
            ),
            "experiments/multitask_tim4rec_optuna/search_space.yaml": sha256_file(
                ROOT / "experiments" / "multitask_tim4rec_optuna" / "search_space.yaml"
            ),
            "slurm/multitask_tim4rec_optuna.sh": sha256_file(ROOT / "slurm" / "multitask_tim4rec_optuna.sh"),
        },
        "environment": environment_info(),
        "slurm": slurm_info(),
        "optuna": {
            "version": optuna.__version__,
            "study_name": optuna_config["study_name"],
            "storage": storage_url,
            "storage_path": str(storage_path),
            "sampler": "TPESampler",
            "sampler_seed": int(optuna_config["study"]["sampler_seed"]),
            "pruner": "MedianPruner",
            "n_startup_trials": int(optuna_config["study"]["n_startup_trials"]),
            "n_warmup_steps": int(optuna_config["study"]["n_warmup_steps"]),
            "interval_steps": int(optuna_config["study"]["interval_steps"]),
            "n_jobs": 1,
            "stale_running_handling": stale,
        },
        "study_state_counts": state_counts(study),
        "target_complete_trials": int(optuna_config["study"]["target_complete_trials"]),
        "search_space": search_space,
        "trial_policy": optuna_config["trial"],
        "data": {
            "protocol": optuna_config["protocol"],
            "validation_only_summary": data.validation_only_summary if data else None,
            "loader_inspection": data.loader_inspection if data else None,
            "model_parameters": {
                "base": data.base_params if data else None,
                "multitask": data.multitask_params if data else None,
                "delta_total": (data.multitask_params["total"] - data.base_params["total"]) if data else None,
            },
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
        },
        "best_trial": None
        if not best_summary
        else {
            "trial": top10[0]["trial"],
            "value": top10[0]["NDCG@10"],
            "params": best_summary["params"],
            "best_epoch": best_summary["best_epoch"],
            "actual_epochs": best_summary["actual_epochs"],
            "validation_metrics": best_summary["validation_metrics"],
            "normalized_task_weights": best_summary["normalized_task_weights"],
            "effective_pos_weights": best_summary["effective_pos_weights"],
            "head_lr_multiplier": best_summary["head_lr_multiplier"],
            "gradient_diagnostic": best_summary["gradient_diagnostic"],
            "auxiliary_metrics": best_summary["auxiliary_metrics"],
            "task_contribution": effective_task_contribution(best_summary),
        },
        "top10_trials": [
            {key: value for key, value in row.items() if key != "summary"}
            for row in top10
        ],
        "parameter_importance": importances,
        "validation_references": refs,
        "fixed_vs_tuned_vs_tim4rec_validation": comparison,
        "runtime": {
            "invocation_runtime_sec": invocation_runtime_sec,
            "sum_recorded_trial_runtime_sec": total_trial_runtime,
            "mean_complete_trial_runtime_sec": mean_complete_runtime,
            "complete_gpu_hours_estimate": None if mean_complete_runtime is None else mean_complete_runtime * len(completed) / 3600.0,
            "gpu": gpu,
        },
        "trial_summaries": trial_summaries,
        "test_safety": {
            "test_evaluation_count": 0,
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
            "test_path_passed_to_search": False,
            "objective_uses_test": False,
            "pruning_uses_test": False,
            "best_trial_selection_uses_test": False,
            "final_test_metrics_present": False,
        },
        "results_csv_updated": False,
    }
    if comparison:
        summary["decision"] = {
            "tuning_exceeds_base_tim4rec_on_validation": comparison["tuned_delta_ndcg10_vs_tim4rec"] > 0,
            "validation_ndcg10_delta_vs_tim4rec": comparison["tuned_delta_ndcg10_vs_tim4rec"],
            "validation_ndcg10_delta_vs_fixed": comparison["tuned_delta_ndcg10_vs_fixed"],
            "negative_transfer": comparison["negative_transfer_status"],
            "enough_for_locked_final_test": comparison["tuned_delta_ndcg10_vs_tim4rec"] > 0,
            "fixed_task_weights_insufficient_signal": comparison["tuned_delta_ndcg10_vs_tim4rec"] <= 0,
        }
    assert_no_test_metrics(summary)
    return summary


def build_notes(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['run_id']}",
        "",
        "## Test safety",
        "",
        "- `test_evaluation_count = 0`.",
        "- Test dataset не загружался, test dataloader не создавался.",
        "- Objective/pruning/best trial selection используют только full-ranking validation NDCG@10.",
        "",
        "## Study",
        "",
        f"- Optuna: `{summary['optuna']['version']}`.",
        f"- Study: `{summary['optuna']['study_name']}`.",
        f"- Sampler: `{summary['optuna']['sampler']}(seed={summary['optuna']['sampler_seed']})`.",
        f"- Pruner: `{summary['optuna']['pruner']}`.",
        f"- State counts: `{summary['study_state_counts']}`.",
        "",
    ]
    best = summary.get("best_trial")
    if best:
        metrics = best["validation_metrics"]
        lines += [
            "## Best trial",
            "",
            f"- Trial: `{best['trial']}`.",
            f"- Best epoch: `{best['best_epoch']}`.",
            f"- NDCG@10: `{metrics['NDCG@10']:.6f}`.",
            f"- HR@10: `{metrics['HR@10']:.6f}`.",
            "",
            "## Top 10",
            "",
            "| rank | trial | NDCG@10 | HR@10 | best_epoch | lambda_aux | lr | weight_decay | dropout | head_lr_mult |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary["top10_trials"]:
            lines.append(
                f"| {row['rank']} | {row['trial']} | {row['NDCG@10']:.6f} | {row['HR@10']:.6f} | "
                f"{row['best_epoch']} | {row['lambda_aux']:.6g} | {row['learning_rate']:.6g} | "
                f"{row['weight_decay']:.6g} | {row['dropout_prob']:.4f} | {row['head_lr_multiplier']:.4f} |"
            )
        comparison = summary["fixed_vs_tuned_vs_tim4rec_validation"]
        lines += [
            "",
            "## Validation comparison",
            "",
            "| model | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in comparison["table"]:
            lines.append(
                f"| {row['model']} | {row['HR@10']:.4f} | {row['HR@20']:.4f} | {row['HR@50']:.4f} | "
                f"{row['NDCG@10']:.4f} | {row['NDCG@20']:.4f} | {row['NDCG@50']:.4f} |"
            )
        lines += [
            "",
            "## Negative transfer",
            "",
            f"- Status: `{comparison['negative_transfer_status']}`.",
            f"- Tuned delta vs TiM4Rec validation NDCG@10: `{comparison['tuned_delta_ndcg10_vs_tim4rec']:.6f}`.",
            f"- Tuned delta vs fixed validation NDCG@10: `{comparison['tuned_delta_ndcg10_vs_fixed']:.6f}`.",
            "",
            "## Parameter importance",
            "",
            "```json",
            json.dumps(summary["parameter_importance"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
    return "\n".join(lines) + "\n"


def write_best_params(optuna_config: dict[str, Any], summary: dict[str, Any]) -> None:
    best = summary.get("best_trial")
    path = ROOT / "experiments" / "multitask_tim4rec_optuna" / "best_params.yaml"
    if not best:
        save_yaml(path, {"run_id": optuna_config["run_id"], "status": "no_complete_trials", "test_evaluation_count": 0})
        return
    save_yaml(
        path,
        {
            "run_id": optuna_config["run_id"],
            "study_name": optuna_config["study_name"],
            "status": "completed",
            "source_run": optuna_config["base_run_id"],
            "trial_number": best["trial"],
            "objective": "validation_full_ranking_NDCG@10",
            "validation_metrics": best["validation_metrics"],
            "best_epoch": best["best_epoch"],
            "actual_epochs": best["actual_epochs"],
            "params": best["params"],
            "normalized_task_weights": best["normalized_task_weights"],
            "effective_pos_weights": best["effective_pos_weights"],
            "head_lr_multiplier": best["head_lr_multiplier"],
            "test_evaluation_count": 0,
            "test_dataset_loaded": False,
            "final_test_metrics_present": False,
        },
    )


def run_trials(
    optuna_config: dict[str, Any],
    search_space: dict[str, Any],
    data: DataBundle,
    study: optuna.Study,
    artifact_root: Path,
    target_complete: int | None,
    n_trials: int | None,
) -> None:
    def objective(trial: optuna.Trial) -> float:
        return train_trial(trial, optuna_config, search_space, data, artifact_root / optuna_config["run_id"])

    if target_complete is not None:
        max_total = int(optuna_config["study"]["max_total_trials"])
        while len(complete_trials(study)) < target_complete and len(study.trials) < max_total:
            before = state_counts(study)
            print(
                json.dumps(
                    {
                        "study": optuna_config["study_name"],
                        "target_complete": target_complete,
                        "state_counts_before": before,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            study.optimize(objective, n_trials=1, n_jobs=1, catch=(RuntimeError, FloatingPointError))
        if len(complete_trials(study)) < target_complete:
            raise RuntimeError(
                f"Study did not reach {target_complete} COMPLETE trials before max_total={max_total}: {state_counts(study)}"
            )
    else:
        study.optimize(objective, n_trials=int(n_trials or 1), n_jobs=1, catch=(RuntimeError, FloatingPointError))


def main() -> None:
    args = parse_args()
    optuna_config = load_yaml(Path(args.config))
    search_space = load_yaml(Path(args.search_space))
    assert_protocol_config(optuna_config)
    artifact_root = Path(optuna_config["remote_artifact_dir"])
    runs_dir = Path(optuna_config["compact_runs_dir"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    study, storage_path, storage_url, stale = create_or_load_study(optuna_config)
    started = time.monotonic()
    data: DataBundle | None = None
    if not args.summary_only:
        data = load_data_bundle(optuna_config, artifact_root / optuna_config["run_id"])
        if args.smoke:
            run_trials(optuna_config, search_space, data, study, artifact_root, target_complete=None, n_trials=1)
        else:
            target = int(args.target_complete or optuna_config["study"]["target_complete_trials"])
            run_trials(optuna_config, search_space, data, study, artifact_root, target_complete=target, n_trials=args.n_trials)
    else:
        data = load_data_bundle(optuna_config, artifact_root / optuna_config["run_id"])

    reloaded = optuna.load_study(study_name=optuna_config["study_name"], storage=storage_url)
    mode = "smoke" if args.smoke else "search"
    summary = build_search_summary(
        optuna_config,
        search_space,
        data,
        reloaded,
        storage_path,
        storage_url,
        stale,
        mode=mode,
        invocation_runtime_sec=float(time.monotonic() - started),
    )
    run_id = summary["run_id"]
    save_json(runs_dir / f"{run_id}.json", summary)
    (runs_dir / f"{run_id}_notes.md").write_text(build_notes(summary), encoding="utf-8")
    if mode == "search":
        write_best_params(optuna_config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False, default=str), flush=True)


if __name__ == "__main__":
    main()
