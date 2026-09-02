"""Stage 3 auxiliary-task ablations and gradient diagnostics.

This runner intentionally uses only the validation-only RecBole dataset prepared
for Protocol B. It never loads or evaluates the TEST split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.dataloader import FullSortEvalDataLoader, TrainDataLoader
from recbole.data.utils import get_dataloader
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from torch import nn

from experiments.multitask_tim4rec.model import TARGETS, MultitaskTiM4Rec
from experiments.multitask_tim4rec.train import (
    EXPECTED_FINGERPRINT,
    EXPECTED_IDENTITY_HASH,
    check_hit_recall_equal,
    evaluate_full_sort_with_checks,
    inspect_eval_loader,
    load_json,
    load_target_stats,
    metric_subset,
    tensor_to_float,
)
from experiments.multitask_tim4rec_optuna.optuna_search import (
    assert_protocol_config,
    assert_validation_only_summary,
    validation_source_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_AUX_TARGETS = tuple(TARGETS)
BINARY_AUDIT_TARGETS = (
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "strong_positive",
    "explicit_positive",
    "deep_engagement",
)
CONTINUOUS_AUDIT_FIELDS = (
    "play_time_ms",
    "duration_ms",
    "play_ratio",
    "profile_stay_time",
    "comment_stay_time",
)


@dataclass
class DataBundle:
    base_config: Config
    full_dataset: Any
    train_dataset: Any
    valid_dataset: Any
    validation_only_summary: dict[str, Any]
    loader_inspection: dict[str, Any]
    target_stats: dict[str, dict[str, float]]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_path(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_ready(value.item())
        return json_ready(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = project_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with project_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_metric_keys(metrics: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        lower = key.lower()
        if lower.startswith("hit@"):
            normalized[f"HR@{lower.split('@', 1)[1]}"] = float(value)
        elif lower.startswith("recall@"):
            normalized[f"Recall@{lower.split('@', 1)[1]}"] = float(value)
        elif lower.startswith("ndcg@"):
            normalized[f"NDCG@{lower.split('@', 1)[1]}"] = float(value)
        else:
            normalized[key.upper()] = float(value)
    return normalized


def rank_loss(model: MultitaskTiM4Rec, interaction: Any) -> torch.Tensor:
    return model.base_loss(interaction)


def active_auxiliary_losses(
    model: MultitaskTiM4Rec,
    interaction: Any,
    active_targets: Sequence[str],
    pos_weights: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    if not active_targets:
        return {}
    representation = model.shared_representation(interaction)
    logits = model.auxiliary_logits_from_representation(representation)
    losses: dict[str, torch.Tensor] = {}
    for target in active_targets:
        labels = interaction[target].float()
        pos_weight = torch.tensor(pos_weights[target], device=labels.device, dtype=labels.dtype)
        losses[target] = nn.functional.binary_cross_entropy_with_logits(
            logits[target],
            labels,
            pos_weight=pos_weight,
        )
    return losses


def multitask_loss(
    model: MultitaskTiM4Rec,
    interaction: Any,
    active_targets: Sequence[str],
    lambda_aux: float,
    aux_loss_weights: Mapping[str, float],
    pos_weights: Mapping[str, float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    primary = rank_loss(model, interaction)
    aux_losses = active_auxiliary_losses(model, interaction, active_targets, pos_weights)
    if aux_losses:
        aux_total = primary.new_zeros(())
        for target, value in aux_losses.items():
            aux_total = aux_total + float(aux_loss_weights[target]) * value
    else:
        aux_total = primary.new_zeros(())
    return primary + float(lambda_aux) * aux_total, primary, aux_losses


def head_prefix(target: str) -> str:
    mapping = {
        "is_click": "click_head.",
        "long_view": "long_view_head.",
        "is_like": "like_head.",
        "is_profile_enter": "profile_enter_head.",
    }
    return mapping[target]


def shared_backbone_parameters(model: MultitaskTiM4Rec) -> list[torch.nn.Parameter]:
    head_prefixes = tuple(head_prefix(target) for target in CURRENT_AUX_TARGETS)
    params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith(head_prefixes)
    ]
    if not params:
        raise RuntimeError("No shared backbone parameters found.")
    return params


def optimizer_for_model(
    model: MultitaskTiM4Rec,
    active_targets: Sequence[str],
    learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    active_head_params: list[torch.nn.Parameter] = []
    for target in active_targets:
        module = getattr(model, head_prefix(target).split(".")[0])
        active_head_params.extend(param for param in module.parameters() if param.requires_grad)

    all_head_ids: set[int] = set()
    for target in CURRENT_AUX_TARGETS:
        module = getattr(model, head_prefix(target).split(".")[0])
        all_head_ids.update(id(param) for param in module.parameters())
    backbone_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in all_head_ids
    ]
    param_groups: list[dict[str, Any]] = [
        {"params": backbone_params, "lr": learning_rate, "weight_decay": weight_decay}
    ]
    if active_head_params:
        param_groups.append(
            {"params": active_head_params, "lr": head_learning_rate, "weight_decay": weight_decay}
        )
    return torch.optim.Adam(param_groups)


def flatten_gradients(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for param in parameters:
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            chunks.append(param.grad.detach().reshape(-1))
    return torch.cat(chunks)


def rng_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(payload["cpu"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all(payload["cuda"])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = torch.linalg.vector_norm(a)
    b_norm = torch.linalg.vector_norm(b)
    if float(a_norm) == 0.0 or float(b_norm) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / (a_norm * b_norm))


def gradient_diagnostic(
    model: MultitaskTiM4Rec,
    interaction: Any,
    active_targets: Sequence[str],
    pos_weights: Mapping[str, float],
) -> dict[str, Any]:
    if not active_targets:
        return {}

    was_training = model.training
    rng = rng_snapshot()
    model.train()
    backbone_params = shared_backbone_parameters(model)
    grads: dict[str, torch.Tensor] = {}
    task_norms: dict[str, float] = {}

    try:
        for task in ("primary", *active_targets):
            model.zero_grad(set_to_none=True)
            if task == "primary":
                loss = rank_loss(model, interaction)
            else:
                loss = active_auxiliary_losses(model, interaction, [task], pos_weights)[task]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite diagnostic loss for {task}: {float(loss)}")
            loss.backward()
            flat = flatten_gradients(backbone_params)
            grads[task] = flat
            task_norms[task] = float(torch.linalg.vector_norm(flat))
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng(rng)
        model.train(was_training)

    primary_norm = task_norms["primary"]
    task_summary: dict[str, Any] = {"primary": {"norm": primary_norm}}
    for target in active_targets:
        cos_value = cosine(grads["primary"], grads[target])
        task_summary[target] = {
            "norm": task_norms[target],
            "norm_ratio_to_primary": task_norms[target] / primary_norm if primary_norm else float("nan"),
            "cosine_to_primary": cos_value,
            "conflict_with_primary": bool(math.isfinite(cos_value) and cos_value < 0.0),
        }

    pairwise: dict[str, dict[str, Any]] = {}
    for idx, left in enumerate(active_targets):
        for right in active_targets[idx + 1 :]:
            cos_value = cosine(grads[left], grads[right])
            pairwise[f"{left}|{right}"] = {
                "cosine": cos_value,
                "conflict": bool(math.isfinite(cos_value) and cos_value < 0.0),
            }

    return {"tasks": task_summary, "pairwise_auxiliary": pairwise}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def summarize_numbers(values: Iterable[Any]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if finite_float(value) is not None]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    array = np.asarray(clean, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_gradient_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    by_pair: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        diagnostic = record.get("diagnostic") or {}
        tasks = diagnostic.get("tasks") or {}
        for target, payload in tasks.items():
            if target == "primary":
                continue
            by_target[target]["aux_norm"].append(payload.get("norm"))
            by_target[target]["norm_ratio_to_primary"].append(payload.get("norm_ratio_to_primary"))
            by_target[target]["cosine_to_primary"].append(payload.get("cosine_to_primary"))
            by_target[target]["conflict_with_primary"].append(bool(payload.get("conflict_with_primary")))
        for pair, payload in (diagnostic.get("pairwise_auxiliary") or {}).items():
            by_pair[pair]["cosine"].append(payload.get("cosine"))
            by_pair[pair]["conflict"].append(bool(payload.get("conflict")))

    target_summary: dict[str, Any] = {}
    for target, values in by_target.items():
        conflicts = values["conflict_with_primary"]
        target_summary[target] = {
            "aux_norm": summarize_numbers(values["aux_norm"]),
            "norm_ratio_to_primary": summarize_numbers(values["norm_ratio_to_primary"]),
            "cosine_to_primary": summarize_numbers(values["cosine_to_primary"]),
            "conflict_fraction_with_primary": (
                float(sum(1 for item in conflicts if item) / len(conflicts)) if conflicts else None
            ),
            "diagnostic_batches": len(conflicts),
        }

    pair_summary: dict[str, Any] = {}
    for pair, values in by_pair.items():
        conflicts = values["conflict"]
        pair_summary[pair] = {
            "cosine": summarize_numbers(values["cosine"]),
            "conflict_fraction": (
                float(sum(1 for item in conflicts if item) / len(conflicts)) if conflicts else None
            ),
            "diagnostic_batches": len(conflicts),
        }

    return {
        "per_auxiliary_vs_primary": target_summary,
        "auxiliary_pairwise": pair_summary,
    }


def should_measure_gradient(epoch: int, batch_idx: int, diag_cfg: Mapping[str, Any]) -> bool:
    if not diag_cfg.get("enabled", True):
        return False
    batches_per_epoch = int(diag_cfg.get("batches_per_epoch", 1))
    if batch_idx >= batches_per_epoch:
        return False
    if epoch == 1 and diag_cfg.get("first_epoch", True):
        return True
    every_n = int(diag_cfg.get("every_n_epochs", 5))
    return every_n > 0 and epoch % every_n == 0


def train_one_epoch(
    model: MultitaskTiM4Rec,
    train_data: TrainDataLoader,
    optimizer: torch.optim.Optimizer,
    active_targets: Sequence[str],
    lambda_aux: float,
    aux_loss_weights: Mapping[str, float],
    pos_weights: Mapping[str, float],
    epoch: int,
    diag_cfg: Mapping[str, Any],
    max_train_batches: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.train()
    total_loss = 0.0
    total_primary = 0.0
    total_aux = defaultdict(float)
    batches = 0
    diagnostics: list[dict[str, Any]] = []

    for batch_idx, interaction in enumerate(train_data):
        if max_train_batches is not None and batch_idx >= max_train_batches:
            break
        interaction = interaction.to(model.device)

        if should_measure_gradient(epoch, batch_idx, diag_cfg):
            diagnostics.append(
                {
                    "epoch": epoch,
                    "batch_index": batch_idx,
                    "diagnostic": gradient_diagnostic(model, interaction, active_targets, pos_weights),
                }
            )

        optimizer.zero_grad(set_to_none=True)
        loss, primary, aux_losses = multitask_loss(
            model,
            interaction,
            active_targets,
            lambda_aux,
            aux_loss_weights,
            pos_weights,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch={epoch} batch={batch_idx}")
        loss.backward()
        optimizer.step()

        total_loss += tensor_to_float(loss)
        total_primary += tensor_to_float(primary)
        for target, value in aux_losses.items():
            total_aux[target] += tensor_to_float(value)
        batches += 1

    if batches == 0:
        raise RuntimeError("No training batches were processed.")

    summary = {
        "loss": total_loss / batches,
        "primary_loss": total_primary / batches,
        "batches": float(batches),
    }
    for target, value in total_aux.items():
        summary[f"{target}_bce"] = value / batches
    return summary, diagnostics


@torch.no_grad()
def evaluate_auxiliary(
    model: MultitaskTiM4Rec,
    valid_data: FullSortEvalDataLoader,
    active_targets: Sequence[str],
) -> dict[str, float]:
    if not active_targets:
        return {}

    model.eval()
    logits_by_target: dict[str, list[torch.Tensor]] = defaultdict(list)
    labels_by_target: dict[str, list[torch.Tensor]] = defaultdict(list)

    for batch in valid_data:
        interaction = batch[0] if isinstance(batch, (tuple, list)) else batch
        interaction = interaction.to(model.device)
        logits = model.auxiliary_logits(interaction)
        for target in active_targets:
            logits_by_target[target].append(logits[target].detach().cpu())
            labels_by_target[target].append(interaction[target].detach().float().cpu())

    metrics: dict[str, float] = {}
    for target in active_targets:
        logits = torch.cat(logits_by_target[target])
        labels = torch.cat(labels_by_target[target])
        bce = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
        probs = torch.sigmoid(logits)
        preds = probs >= 0.5
        positives = labels == 1
        negatives = labels == 0
        tp = torch.logical_and(preds, positives).sum().item()
        fp = torch.logical_and(preds, negatives).sum().item()
        fn = torch.logical_and(~preds, positives).sum().item()
        tn = torch.logical_and(~preds, negatives).sum().item()
        metrics[f"{target}_bce"] = float(bce)
        metrics[f"{target}_accuracy"] = float((tp + tn) / max(tp + tn + fp + fn, 1))
        metrics[f"{target}_positive_rate_pred"] = float(preds.float().mean().item())
        metrics[f"{target}_positive_rate_true"] = float(labels.float().mean().item())
    return metrics


def create_loaders(config: Config, train_dataset: Any, valid_dataset: Any) -> tuple[TrainDataLoader, FullSortEvalDataLoader]:
    train_loader = get_dataloader(config, "train")(config, train_dataset, None, shuffle=config["shuffle"])
    valid_loader = get_dataloader(config, "valid")(config, valid_dataset, None, shuffle=False)
    if not isinstance(train_loader, TrainDataLoader):
        raise RuntimeError(f"Expected TrainDataLoader, got {type(train_loader).__name__}")
    if not isinstance(valid_loader, FullSortEvalDataLoader):
        raise RuntimeError(f"Expected FullSortEvalDataLoader, got {type(valid_loader).__name__}")
    return train_loader, valid_loader


def load_data_bundle(stage_cfg: Mapping[str, Any]) -> DataBundle:
    optuna_cfg = load_yaml(stage_cfg["source"]["validation_only_config"])
    assert_protocol_config(optuna_cfg)

    summary_path = project_path(optuna_cfg["validation_only_data"]["summary_json"])
    if not summary_path.exists():
        raise FileNotFoundError(f"Validation-only dataset summary is missing: {summary_path}")
    summary = load_json(summary_path)
    assert_validation_only_summary(summary)

    recbole_overrides = dict(optuna_cfg.get("recbole_overrides", {}))
    recbole_overrides["benchmark_filename"] = ["train", "valid"]
    recbole_overrides["final_test_evaluation_count"] = 0
    recbole_overrides["test_evaluation_count"] = 0
    recbole_overrides["multitask_targets"] = list(CURRENT_AUX_TARGETS)
    recbole_overrides["epochs"] = int(stage_cfg["training"]["max_epochs"])
    recbole_overrides["train_batch_size"] = int(stage_cfg["training"]["train_batch_size"])
    recbole_overrides["eval_batch_size"] = int(stage_cfg["training"]["eval_batch_size"])
    recbole_overrides["seed"] = int(stage_cfg["training"]["seed"])
    recbole_overrides["checkpoint_dir"] = str(project_path(stage_cfg["outputs"]["artifact_dir"]))

    config = Config(
        model=MultitaskTiM4Rec,
        config_file_list=[str(project_path(optuna_cfg["source"]["base_config"]))],
        config_dict=recbole_overrides,
    )
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    init_logger(config)

    full_dataset = create_dataset(config)
    built = full_dataset.build()
    if len(built) == 2:
        train_dataset, valid_dataset = built
    elif len(built) == 3:
        train_dataset, valid_dataset, unused_dataset = built
        if len(unused_dataset) != 0:
            raise RuntimeError(f"Validation-only RecBole split created non-empty unused split: {len(unused_dataset)}")
    else:
        raise RuntimeError(f"Expected train/valid validation-only splits from RecBole, got {len(built)}")

    train_data, valid_data = create_loaders(config, train_dataset, valid_dataset)
    expected_validation_ids = validation_source_ids(summary)
    inspection = inspect_eval_loader(valid_data, int(valid_data._dataset.item_num), expected_validation_ids)
    if not inspection["one_positive_per_row"]:
        raise RuntimeError(f"Validation split must have one positive per row: {inspection}")
    if not inspection["positive_targets_within_item_universe"]:
        raise RuntimeError(f"Validation positives outside item universe: {inspection}")
    if int(valid_data._dataset.item_num) - 1 != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Full-ranking item universe changed: {int(valid_data._dataset.item_num) - 1}")
    if len(train_dataset) != EXPECTED_FINGERPRINT["train"] - EXPECTED_FINGERPRINT["users"]:
        raise RuntimeError(f"Sequential train examples changed: {len(train_dataset)}")
    if len(valid_dataset) != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Validation examples changed: {len(valid_dataset)}")

    target_stats = load_target_stats(project_path(stage_cfg["source"]["target_statistics"]))

    return DataBundle(
        base_config=config,
        full_dataset=full_dataset,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        validation_only_summary=summary,
        loader_inspection=inspection,
        target_stats=target_stats,
    )


def run_config_for_key(stage_cfg: Mapping[str, Any], run_key: str) -> dict[str, Any]:
    try:
        run_cfg = dict(stage_cfg["runs"][run_key])
    except KeyError as exc:
        keys = ", ".join(sorted(stage_cfg["runs"].keys()))
        raise KeyError(f"Unknown run key {run_key!r}. Available: {keys}") from exc
    active_targets = tuple(run_cfg.get("active_targets", []))
    unexpected = sorted(set(active_targets) - set(CURRENT_AUX_TARGETS))
    if unexpected:
        raise ValueError(f"Unsupported active targets for current model: {unexpected}")
    run_cfg["active_targets"] = list(active_targets)
    return run_cfg


def build_run_recbole_config(
    stage_cfg: Mapping[str, Any],
    base_config: Config,
    run_cfg: Mapping[str, Any],
    max_epochs_override: int | None,
) -> Config:
    optuna_cfg = load_yaml(stage_cfg["source"]["validation_only_config"])
    overrides = dict(optuna_cfg.get("recbole_overrides", {}))
    overrides["benchmark_filename"] = ["train", "valid"]
    overrides["multitask_targets"] = list(run_cfg["active_targets"])
    overrides["seed"] = int(stage_cfg["training"]["seed"])
    overrides["epochs"] = int(
        max_epochs_override
        or run_cfg.get("max_epochs")
        or stage_cfg["training"]["max_epochs"]
    )
    overrides["stopping_step"] = int(stage_cfg["training"]["early_stopping_patience"])
    overrides["train_batch_size"] = int(stage_cfg["training"]["train_batch_size"])
    overrides["eval_batch_size"] = int(stage_cfg["training"]["eval_batch_size"])
    overrides["learning_rate"] = float(stage_cfg["optimization"]["learning_rate"])
    overrides["weight_decay"] = float(stage_cfg["optimization"]["weight_decay"])
    overrides["dropout_prob"] = float(stage_cfg["optimization"]["dropout_prob"])
    overrides["checkpoint_dir"] = str(project_path(stage_cfg["outputs"]["artifact_dir"]) / run_cfg["run_id"])
    overrides["final_test_evaluation_count"] = 0
    overrides["test_evaluation_count"] = 0
    overrides["load_best_model"] = False

    config = Config(
        model=MultitaskTiM4Rec,
        config_file_list=[str(project_path(optuna_cfg["source"]["base_config"]))],
        config_dict=overrides,
    )
    # Preserve stable values expected by RecBole internals.
    config["data_path"] = base_config["data_path"]
    config["dataset"] = base_config["dataset"]
    return config


def loss_weights_for_run(stage_cfg: Mapping[str, Any], run_cfg: Mapping[str, Any]) -> dict[str, float]:
    mode = run_cfg.get("loss_weight_mode", "uniform_single_aux")
    active_targets = run_cfg["active_targets"]
    if mode == "none":
        return {}
    if mode == "uniform_single_aux":
        return {
            target: float(stage_cfg["optimization"]["single_auxiliary_loss_weight"])
            for target in active_targets
        }
    if mode == "tuned_task_weights":
        tuned = stage_cfg["optimization"]["tuned_task_weights"]
        return {target: float(tuned[target]) for target in active_targets}
    raise ValueError(f"Unsupported loss_weight_mode: {mode}")


def run_ablation(
    stage_cfg: Mapping[str, Any],
    run_key: str,
    max_epochs_override: int | None = None,
    max_train_batches_override: int | None = None,
) -> dict[str, Any]:
    run_cfg = run_config_for_key(stage_cfg, run_key)
    data = load_data_bundle(stage_cfg)
    config = build_run_recbole_config(stage_cfg, data.base_config, run_cfg, max_epochs_override)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    train_data, valid_data = create_loaders(config, data.train_dataset, data.valid_dataset)

    model = MultitaskTiM4Rec(config, train_data.dataset).to(config["device"])
    optimizer = optimizer_for_model(
        model,
        run_cfg["active_targets"],
        float(stage_cfg["optimization"]["learning_rate"]),
        float(stage_cfg["optimization"]["head_learning_rate"]),
        float(stage_cfg["optimization"]["weight_decay"]),
    )
    trainer = Trainer(config, model)
    trainer.optimizer = optimizer
    aux_weights = loss_weights_for_run(stage_cfg, run_cfg)
    pos_weights = {
        target: float(stage_cfg["optimization"]["effective_pos_weights"][target])
        for target in run_cfg["active_targets"]
    }

    max_epochs = int(config["epochs"])
    max_train_batches = max_train_batches_override
    if max_train_batches is None and run_cfg.get("max_train_batches") is not None:
        max_train_batches = int(run_cfg["max_train_batches"])
    patience = int(stage_cfg["training"]["early_stopping_patience"])
    diag_cfg = stage_cfg["training"].get("gradient_diagnostics", {})

    best_score = -float("inf")
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_aux_metrics: dict[str, float] = {}
    bad_epochs = 0
    histories: list[dict[str, Any]] = []
    gradient_records: list[dict[str, Any]] = []

    run_id = run_cfg["run_id"]
    run_path = project_path(stage_cfg["outputs"]["runs_dir"]) / f"{run_id}.json"
    partial_path = run_path.with_suffix(".partial.json")

    for epoch in range(1, max_epochs + 1):
        train_summary, epoch_diagnostics = train_one_epoch(
            model,
            train_data,
            optimizer,
            run_cfg["active_targets"],
            float(run_cfg.get("lambda_aux", stage_cfg["optimization"]["lambda_aux"])),
            aux_weights,
            pos_weights,
            epoch,
            diag_cfg,
            max_train_batches=max_train_batches,
        )
        gradient_records.extend(epoch_diagnostics)

        valid_result, full_ranking_checks = evaluate_full_sort_with_checks(trainer, valid_data, train_data)
        check_hit_recall_equal(valid_result, list(config["topk"]))
        if not full_ranking_checks["raw_scores_all_finite"] or not full_ranking_checks["positive_scores_all_finite"]:
            raise RuntimeError(f"Non-finite validation scores: {full_ranking_checks}")
        ranking_metrics = normalize_metric_keys(metric_subset(valid_result))
        aux_metrics = evaluate_auxiliary(model, valid_data, run_cfg["active_targets"])
        ndcg10 = float(ranking_metrics["NDCG@10"])
        improved = ndcg10 > best_score
        if improved:
            best_score = ndcg10
            best_epoch = epoch
            best_metrics = ranking_metrics
            best_aux_metrics = aux_metrics
            bad_epochs = 0
        else:
            bad_epochs += 1

        histories.append(
            {
                "epoch": epoch,
                "train": train_summary,
                "validation_ranking": ranking_metrics,
                "full_ranking_checks": full_ranking_checks,
                "validation_auxiliary": aux_metrics,
                "is_best": improved,
            }
        )

        partial_payload = {
            "run_id": run_id,
            "run_key": run_key,
            "status": "RUNNING",
            "updated_at_utc": now_utc(),
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "test_evaluation_count": 0,
            "history": histories,
        }
        save_json(partial_path, partial_payload)

        if bad_epochs >= patience:
            break

    gradient_summary = aggregate_gradient_records(gradient_records)
    payload = {
        "run_id": run_id,
        "run_key": run_key,
        "status": "COMPLETE",
        "created_at_utc": now_utc(),
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "remote_head": git_value("rev-parse", "origin/exp/moo-8families-benchmark"),
        },
        "protocol": {
            "dataset": stage_cfg["protocol"]["dataset"],
            "split": stage_cfg["protocol"]["split"],
            "evaluation_split": "validation",
            "test_evaluation_count": 0,
            "test_dataset_loaded": False,
            "test_metrics_present": False,
            "expected_identity_hash": EXPECTED_IDENTITY_HASH,
            "expected_fingerprint": EXPECTED_FINGERPRINT,
        },
        "data": {
            "dataset": config["dataset"],
            "benchmark_filename": list(config["benchmark_filename"]),
            "train_interactions": len(data.train_dataset),
            "validation_interactions": len(data.valid_dataset),
            "loader_inspection": data.loader_inspection,
            "target_stats_source": rel_path(stage_cfg["source"]["target_statistics"]),
        },
        "objective": {
            "primary": stage_cfg["protocol"]["primary_objective"],
            "active_auxiliary_targets": list(run_cfg["active_targets"]),
            "lambda_aux": float(run_cfg.get("lambda_aux", stage_cfg["optimization"]["lambda_aux"])),
            "auxiliary_loss_weights": aux_weights,
            "effective_pos_weights": pos_weights,
            "loss_weight_mode": run_cfg.get("loss_weight_mode"),
        },
        "optimization": {
            "seed": int(config["seed"]),
            "max_epochs": max_epochs,
            "actual_epochs": len(histories),
            "early_stopping_patience": patience,
            "learning_rate": float(stage_cfg["optimization"]["learning_rate"]),
            "head_learning_rate": float(stage_cfg["optimization"]["head_learning_rate"]),
            "weight_decay": float(stage_cfg["optimization"]["weight_decay"]),
            "dropout_prob": float(stage_cfg["optimization"]["dropout_prob"]),
            "max_train_batches": max_train_batches,
        },
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "best_validation_auxiliary_metrics": best_aux_metrics,
        "gradient_diagnostics": {
            "policy": dict(diag_cfg),
            "rng_restored_before_training_step": True,
            "raw_records": gradient_records,
            "summary": gradient_summary,
        },
        "history": histories,
        "test_evaluation_count": 0,
    }
    save_json(run_path, payload)
    if partial_path.exists():
        partial_path.unlink()
    return payload


def read_stats_rows(path: str | Path) -> list[dict[str, str]]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stats_by_split_target(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
    return {(row["scope"], row["field"]): row for row in rows}


def float_from_row(row: Mapping[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def missing_rate(row: Mapping[str, str] | None) -> float | None:
    if row is None:
        return None
    rows = float_from_row(row, "rows")
    missing = float_from_row(row, "missing")
    if rows in (None, 0.0) or missing is None:
        return None
    return missing / rows


def derive_binary_columns(frame: Any) -> Any:
    import polars as pl

    additions = []
    if {"is_like", "is_follow", "is_comment", "is_forward"}.issubset(frame.columns):
        additions.append(
            (
                (pl.col("is_like") == 1)
                | (pl.col("is_follow") == 1)
                | (pl.col("is_comment") == 1)
                | (pl.col("is_forward") == 1)
            )
            .cast(pl.Int8)
            .alias("explicit_positive")
        )
    if {"long_view", "is_like", "is_follow", "is_comment", "is_forward"}.issubset(frame.columns):
        additions.append(
            (
                (pl.col("long_view") == 1)
                | (pl.col("is_like") == 1)
                | (pl.col("is_follow") == 1)
                | (pl.col("is_comment") == 1)
                | (pl.col("is_forward") == 1)
            )
            .cast(pl.Int8)
            .alias("deep_engagement")
        )
    if {"is_like", "is_follow", "is_comment", "is_forward", "is_profile_enter"}.issubset(frame.columns):
        additions.append(
            (
                (pl.col("is_like") == 1)
                | (pl.col("is_follow") == 1)
                | (pl.col("is_comment") == 1)
                | (pl.col("is_forward") == 1)
                | (pl.col("is_profile_enter") == 1)
            )
            .cast(pl.Int8)
            .alias("strong_positive")
        )
    if additions:
        frame = frame.with_columns(additions)
    return frame


def binary_relationships(frame: Any, binary_targets: Sequence[str]) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    available = [target for target in binary_targets if target in frame.columns]
    total = frame.height
    for idx, left in enumerate(available):
        left_values = frame[left].fill_null(0).cast(bool)
        left_pos = int(left_values.sum())
        for right in available[idx + 1 :]:
            right_values = frame[right].fill_null(0).cast(bool)
            right_pos = int(right_values.sum())
            both = int((left_values & right_values).sum())
            left_only = left_pos - both
            right_only = right_pos - both
            neither = total - both - left_only - right_only
            denominator = math.sqrt(
                max((both + left_only) * (right_only + neither) * (both + right_only) * (left_only + neither), 0)
            )
            phi = ((both * neither - left_only * right_only) / denominator) if denominator else float("nan")
            union = both + left_only + right_only
            relationships.append(
                {
                    "left": left,
                    "right": right,
                    "n": total,
                    "both_positive": both,
                    "p_left": left_pos / total if total else float("nan"),
                    "p_right": right_pos / total if total else float("nan"),
                    "p_right_given_left": both / left_pos if left_pos else float("nan"),
                    "p_left_given_right": both / right_pos if right_pos else float("nan"),
                    "jaccard": both / union if union else float("nan"),
                    "phi": phi,
                }
            )
    return relationships


def continuous_binary_relationships(
    frame: Any,
    continuous_fields: Sequence[str],
    binary_targets: Sequence[str],
) -> list[dict[str, Any]]:
    import polars as pl

    rows: list[dict[str, Any]] = []
    binaries = [target for target in binary_targets if target in frame.columns]
    continuous = [field for field in continuous_fields if field in frame.columns]
    for field in continuous:
        for target in binaries:
            data = frame.select([pl.col(field), pl.col(target).fill_null(0).cast(pl.Int8)]).drop_nulls(field)
            if data.height == 0:
                continue
            pos = data.filter(pl.col(target) == 1)
            neg = data.filter(pl.col(target) == 0)
            rows.append(
                {
                    "continuous": field,
                    "binary": target,
                    "n": data.height,
                    "mean_if_positive": float(pos[field].mean()) if pos.height else float("nan"),
                    "mean_if_negative": float(neg[field].mean()) if neg.height else float("nan"),
                    "median_if_positive": float(pos[field].median()) if pos.height else float("nan"),
                    "median_if_negative": float(neg[field].median()) if neg.height else float("nan"),
                    "pearson": float(data.select(pl.corr(field, target)).item())
                    if data[target].n_unique() > 1
                    else float("nan"),
                }
            )
    return rows


def target_audit(stage_cfg: Mapping[str, Any]) -> dict[str, Any]:
    rows = read_stats_rows(stage_cfg["source"]["target_statistics"])
    indexed = stats_by_split_target(rows)
    candidates = stage_cfg["candidate_targets"]
    audit_rows: list[dict[str, Any]] = []

    for target, metadata in candidates.items():
        train_row = indexed.get(("train", target))
        valid_row = indexed.get(("validation", target))
        audit_rows.append(
            {
                "target": target,
                "display_name": metadata["display_name"],
                "type": metadata["type"],
                "currently_used": bool(metadata["currently_used"]),
                "stage3_ablation_enabled": bool(metadata["ablation_enabled"]),
                "train_observations": int(float_from_row(train_row, "rows") or 0),
                "train_positive_count": int(float_from_row(train_row, "positives") or 0)
                if metadata["type"] == "binary"
                else None,
                "train_positive_rate": float_from_row(train_row, "positive_rate")
                if metadata["type"] == "binary"
                else None,
                "validation_positive_rate": float_from_row(valid_row, "positive_rate")
                if metadata["type"] == "binary"
                else None,
                "missing_rate_train": missing_rate(train_row),
                "basic_distribution_train": {
                    key: float_from_row(train_row, key)
                    for key in ("mean", "std", "median", "p90", "p95", "p99", "max")
                    if float_from_row(train_row, key) is not None
                },
                "construction": metadata["construction"],
                "leakage_note": metadata["leakage_note"],
            }
        )

    protocol_dir = project_path(stage_cfg["source"]["protocol_b_multitask_dir"])
    train_parquet = protocol_dir / "train.parquet"
    relationship_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []
    relationship_source = ""
    if train_parquet.exists():
        import polars as pl

        columns = sorted(
            set(BINARY_AUDIT_TARGETS)
            | set(CONTINUOUS_AUDIT_FIELDS)
            | {"is_follow", "is_comment", "is_forward", "is_hate"}
        )
        available_columns = pl.scan_parquet(train_parquet).collect_schema().names()
        read_columns = [column for column in columns if column in available_columns]
        frame = pl.read_parquet(train_parquet, columns=read_columns)
        frame = derive_binary_columns(frame)
        relationship_rows = binary_relationships(frame, BINARY_AUDIT_TARGETS)
        continuous_rows = continuous_binary_relationships(
            frame,
            CONTINUOUS_AUDIT_FIELDS,
            BINARY_AUDIT_TARGETS[:8],
        )
        relationship_source = rel_path(train_parquet)
    else:
        relationship_source = f"missing: {rel_path(train_parquet)}"

    payload = {
        "run_id": stage_cfg["outputs"]["target_audit_run_id"],
        "status": "COMPLETE",
        "created_at_utc": now_utc(),
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "remote_head": git_value("rev-parse", "origin/exp/moo-8families-benchmark"),
        },
        "protocol": {
            "dataset": stage_cfg["protocol"]["dataset"],
            "split": stage_cfg["protocol"]["split"],
            "evaluation_split": "train/validation descriptive audit only",
            "test_evaluation_count": 0,
            "test_dataset_loaded": False,
            "test_metrics_present": False,
        },
        "target_statistics_source": rel_path(stage_cfg["source"]["target_statistics"]),
        "target_statistics_sha256": sha256_file(stage_cfg["source"]["target_statistics"]),
        "relationship_source": relationship_source,
        "target_audit": audit_rows,
        "binary_relationships_train": relationship_rows,
        "continuous_binary_relationships_train": continuous_rows,
        "candidate_selection": {
            "single_auxiliary_ablation_targets": list(CURRENT_AUX_TARGETS),
            "not_ablationed_now": {
                "is_follow": "Very low train prevalence and not in current model scope.",
                "is_comment": "Very low train prevalence and not in current model scope.",
                "is_forward": "Very low train prevalence and not in current model scope.",
                "is_hate": "Extreme rarity, negative-feedback semantics, and not in current model scope.",
                "play_time_ms": "Continuous post-exposure signal; requires a separate objective design.",
                "play_ratio": "Continuous post-exposure signal; requires a separate objective design.",
            },
        },
        "test_evaluation_count": 0,
    }
    output_path = project_path(stage_cfg["outputs"]["runs_dir"]) / f"{payload['run_id']}.json"
    save_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/stage3_auxiliary_analysis/config.yaml")
    parser.add_argument("--mode", choices=("target-audit", "ablation"), required=True)
    parser.add_argument("--run-key", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    args = parser.parse_args(argv)

    stage_cfg = load_yaml(args.config)
    if args.mode == "target-audit":
        target_audit(stage_cfg)
        return 0
    if args.run_key is None:
        parser.error("--run-key is required for ablation mode")
    run_ablation(
        stage_cfg,
        args.run_key,
        max_epochs_override=args.max_epochs,
        max_train_batches_override=args.max_train_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
