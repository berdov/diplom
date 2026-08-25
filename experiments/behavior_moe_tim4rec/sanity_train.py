#!/usr/bin/env python
"""5-epoch validation sanity training for plain Behavior-MoE TiM4Rec."""

from __future__ import annotations

import argparse
import csv
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from recbole.config import Config
from recbole.trainer import Trainer
from recbole.utils import init_seed


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tim4rec import TiM4Rec  # noqa: E402
from experiments.behavior_moe_tim4rec.model import (  # noqa: E402
    EXPERTS,
    ROUTING_TASKS,
    BehaviorMoETiM4Rec,
)
from experiments.behavior_moe_tim4rec.smoke_test import (  # noqa: E402
    TASK_LABELS,
    count_by_group,
)
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS  # noqa: E402
from experiments.multitask_tim4rec.train import (  # noqa: E402
    EXPECTED_FINGERPRINT,
    EXPECTED_IDENTITY_HASH,
    all_gradient_check,
    check_hit_recall_equal,
    count_parameters,
    evaluate_auxiliary,
    evaluate_full_sort_with_checks,
    first_batch,
    load_json,
    load_target_stats,
    metric_subset,
    sha256_file,
    tensor_to_float,
)
from experiments.multitask_tim4rec_optuna.optuna_search import (  # noqa: E402
    assert_protocol_config,
    assert_validation_only_summary,
    compact_validation,
    create_loaders,
    load_data_bundle,
    load_yaml,
    normalize_metrics,
    optimizer_for_trial,
    pos_weight_tensors,
    project_path,
    recbole_overrides,
)
from experiments.multitask_tim4rec_optuna.run_locked_tuned import sampled_from_locked_params  # noqa: E402


RUN_ID = "behavior_moe_sanity_001"
DEFAULT_REMOTE_ROOT = Path("/home/daryumin/iberdov/diplom/experiments/behavior_moe_tim4rec")
SMOKE_RUN = ROOT / "experiments" / "behavior_moe_tim4rec" / "runs" / "behavior_moe_smoke_001.json"
REFERENCE_RUNS = {
    "tim4rec_sanity_001": ROOT / "experiments" / "tim4rec_baseline" / "runs" / "tim4rec_sanity_001.json",
    "multitask_tim4rec_sanity_001": ROOT
    / "experiments"
    / "multitask_tim4rec"
    / "runs"
    / "multitask_tim4rec_sanity_001.json",
    "multitask_tim4rec_tuned_001": ROOT
    / "experiments"
    / "multitask_tim4rec_optuna"
    / "runs"
    / "multitask_tim4rec_tuned_001.json",
}
ROUTING_PAIRS = (
    ("ranking", "click"),
    ("ranking", "long_view"),
    ("ranking", "like"),
    ("ranking", "profile"),
    ("click", "long_view"),
    ("like", "profile"),
)
SEMANTIC_EXPECTATIONS = {
    "click": "interest",
    "long_view": "consumption",
    "like": "positive",
    "profile": "positive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments" / "behavior_moe_tim4rec" / "config.yaml"))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--routing-csv", default=None)
    parser.add_argument("--diagnostic-epochs", default="1,3,5")
    parser.add_argument("--routing-diagnostic-batches", type=int, default=5)
    parser.add_argument("--routing-example-limit", type=int, default=8)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def git_value(args: list[str], default: str = "unknown") -> str:
    env_map = {
        ("rev-parse", "HEAD"): "BEHAVIOR_MOE_GIT_COMMIT",
        ("rev-parse", "--abbrev-ref", "HEAD"): "BEHAVIOR_MOE_GIT_BRANCH",
        ("config", "--get", "remote.origin.url"): "BEHAVIOR_MOE_GIT_REMOTE",
    }
    env_key = env_map.get(tuple(args))
    if env_key and os.environ.get(env_key):
        return str(os.environ[env_key])
    try:
        value = subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return value or default
    except Exception:
        return default


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "recbole": version("recbole"),
        "optuna": version("optuna"),
        "mamba_ssm": version("mamba-ssm"),
        "causal_conv1d": version("causal-conv1d"),
        "pyyaml": yaml.__version__,
    }


def slurm_info() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "constraint": os.environ.get("SLURM_JOB_CONSTRAINT") or os.environ.get("BEHAVIOR_MOE_SLURM_CONSTRAINT"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "hostname": socket.gethostname(),
    }


def gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "device": "cpu"}
    current = torch.cuda.current_device()
    return {
        "available": True,
        "device": "cuda",
        "name": torch.cuda.get_device_name(current),
        "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(current)),
        "device_count": torch.cuda.device_count(),
    }


def source_hashes() -> dict[str, str]:
    paths = [
        "experiments/behavior_moe_tim4rec/model.py",
        "experiments/behavior_moe_tim4rec/smoke_test.py",
        "experiments/behavior_moe_tim4rec/sanity_train.py",
        "experiments/behavior_moe_tim4rec/config.yaml",
        "experiments/multitask_tim4rec/model.py",
        "experiments/multitask_tim4rec/train.py",
        "experiments/multitask_tim4rec_optuna/optuna_search.py",
        "experiments/multitask_tim4rec_optuna/run_locked_tuned.py",
        "experiments/multitask_tim4rec_optuna/best_params.yaml",
        "slurm/behavior_moe_tim4rec_sanity.sh",
    ]
    return {path: sha256_file(ROOT / path) for path in paths if (ROOT / path).exists()}


def run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else DEFAULT_REMOTE_ROOT / args.run_id
    runs_dir = ROOT / "experiments" / "behavior_moe_tim4rec" / "runs"
    result_json = Path(args.result_json) if args.result_json else runs_dir / f"{args.run_id}.json"
    notes = Path(args.notes) if args.notes else runs_dir / f"{args.run_id}_notes.md"
    routing_csv = Path(args.routing_csv) if args.routing_csv else runs_dir / f"{args.run_id}_routing.csv"
    return artifact_dir, result_json, notes, routing_csv


def parse_diagnostic_epochs(value: str) -> set[int]:
    epochs = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not epochs:
        raise RuntimeError("At least one diagnostic epoch is required.")
    return epochs


def build_behavior_config(
    optuna_config: dict[str, Any],
    artifact_dir: Path,
    sampled: dict[str, Any],
    moe_config: dict[str, Any],
    epochs: int,
) -> Config:
    overrides = recbole_overrides(optuna_config, artifact_dir / "recbole", sampled)
    overrides.update(
        {
            "behavior_moe": moe_config,
            "epochs": int(epochs),
            "stopping_step": int(epochs) + 1,
            "final_test_evaluation_count": 0,
            "test_evaluation_count": 0,
            "metrics": ["Hit", "Recall", "NDCG"],
            "topk": [5, 10, 20, 50],
            "valid_metric": "NDCG@10",
            "show_progress": False,
            "log_wandb": False,
        }
    )
    return Config(
        model=BehaviorMoETiM4Rec,
        config_file_list=[str(project_path(optuna_config["source"]["base_config"]))],
        config_dict=overrides,
    )


def float_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    keys = [
        "total",
        "rank",
        "aux_sum",
        "weighted_aux_sum",
        "load_balance_loss",
        "load_balance_contribution",
        "is_click_loss",
        "long_view_loss",
        "is_like_loss",
        "is_profile_enter_loss",
        "is_click_scaled_contribution",
        "long_view_scaled_contribution",
        "is_like_scaled_contribution",
        "is_profile_enter_scaled_contribution",
    ]
    return {key: tensor_to_float(losses[key]) for key in keys}


def train_one_epoch_behavior(
    model: BehaviorMoETiM4Rec,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    device: torch.device,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    load_balance_weight: float,
) -> dict[str, Any]:
    model.train()
    keys = [
        "total",
        "rank",
        "aux_sum",
        "weighted_aux_sum",
        "load_balance_loss",
        "load_balance_contribution",
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
        losses = model.calculate_multitask_loss(
            interaction,
            lambda_aux=float(sampled["lambda_aux"]),
            pos_weights=pos_weights,
            task_weights=sampled["normalized_task_weights"],
            load_balance_weight=load_balance_weight,
        )
        loss_values = float_losses(losses)
        if not all(math.isfinite(value) for value in loss_values.values()):
            raise RuntimeError(f"Non-finite Behavior-MoE losses in batch {batches}: {loss_values}")
        losses["total"].backward()
        grad_check = all_gradient_check(model)
        if not bool(grad_check["all_finite"]):
            raise RuntimeError(f"Non-finite gradients in train batch {batches}: {grad_check}")
        optimizer.step()
        for key in keys:
            sums[key] += loss_values[key] * batch_size
        examples += batch_size
        batches += 1
    if examples == 0:
        raise RuntimeError("No training examples.")
    result = {key: value / examples for key, value in sums.items()}
    result["auxiliary_scaled_contribution"] = float(sampled["lambda_aux"]) * result["weighted_aux_sum"]
    result["auxiliary_rank_ratio"] = result["auxiliary_scaled_contribution"] / result["rank"] if result["rank"] else None
    result["per_task_rank_ratio"] = {
        target: result[f"{target}_scaled_contribution"] / result["rank"] if result["rank"] else None
        for target in TARGETS
    }
    result["batches"] = batches
    result["examples"] = examples
    return result


def module_gradient_norms(modules: torch.nn.ModuleDict) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for module_name, module in modules.items():
        sq_norm = 0.0
        tensors = 0
        missing = 0
        nonfinite = 0
        for param in module.parameters():
            grad = param.grad
            if grad is None:
                missing += 1
                continue
            tensors += 1
            nonfinite += int(not torch.isfinite(grad).all().item())
            sq_norm += float(torch.linalg.vector_norm(grad.detach().float()).cpu().item()) ** 2
        result[TASK_LABELS.get(module_name, module_name)] = {
            "gradient_tensors": tensors,
            "missing_gradient_tensors": missing,
            "nonfinite_gradient_tensors": nonfinite,
            "gradient_norm": math.sqrt(sq_norm),
            "receives_gradient": tensors > 0 and missing == 0 and nonfinite == 0 and sq_norm > 0.0,
        }
    return result


def fixed_gradient_diagnostic(
    model: BehaviorMoETiM4Rec,
    interaction: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    load_balance_weight: float,
    epoch: int,
) -> dict[str, Any]:
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    losses = model.calculate_multitask_loss(
        interaction,
        lambda_aux=float(sampled["lambda_aux"]),
        pos_weights=pos_weights,
        task_weights=sampled["normalized_task_weights"],
        load_balance_weight=load_balance_weight,
    )
    losses["total"].backward()
    all_finite = bool(all_gradient_check(model)["all_finite"])
    experts = module_gradient_norms(model.experts)
    routers = module_gradient_norms(model.router_heads)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return {
        "epoch": int(epoch),
        "batch_size": len(interaction),
        "losses": float_losses(losses),
        "experts": experts,
        "routers": routers,
        "all_gradients_finite": all_finite,
        "all_experts_receive_gradients": all(item["receives_gradient"] for item in experts.values()),
        "all_routers_receive_gradients": all(item["receives_gradient"] for item in routers.values()),
    }


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    eps = 1e-12
    left = left.clamp_min(eps)
    right = right.clamp_min(eps)
    middle = 0.5 * (left + right)
    kl_left = torch.sum(left * torch.log(left / middle))
    kl_right = torch.sum(right * torch.log(right / middle))
    return float((0.5 * (kl_left + kl_right)).cpu().item())


def task_distances(mean_probs: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    vectors = {
        task: torch.tensor([values[expert] for expert in EXPERTS], dtype=torch.float64)
        for task, values in mean_probs.items()
    }
    rows = []
    all_pairs = list(ROUTING_PAIRS)
    names = list(vectors)
    all_pairs.extend((names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names)) if (names[i], names[j]) not in all_pairs)
    seen = set()
    for left, right in all_pairs:
        key = (left, right)
        if key in seen or left not in vectors or right not in vectors:
            continue
        seen.add(key)
        diff = vectors[left] - vectors[right]
        denom = torch.linalg.vector_norm(vectors[left]) * torch.linalg.vector_norm(vectors[right])
        cosine_similarity = float(torch.dot(vectors[left], vectors[right]) / denom) if denom > 0 else None
        rows.append(
            {
                "left": left,
                "right": right,
                "required_pair": key in ROUTING_PAIRS,
                "l1": float(diff.abs().sum().item()),
                "l2": float(torch.linalg.vector_norm(diff).item()),
                "js_divergence": js_divergence(vectors[left], vectors[right]),
                "cosine_similarity": cosine_similarity,
                "cosine_distance": None if cosine_similarity is None else 1.0 - cosine_similarity,
            }
        )
    rows.sort(key=lambda item: (not item["required_pair"], -item["l1"]))
    return rows


@torch.no_grad()
def routing_diagnostics(
    model: BehaviorMoETiM4Rec,
    valid_data: Any,
    max_batches: int,
    example_limit: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    expert_count = len(EXPERTS)
    log_experts = math.log(expert_count)
    prob_sums = {task: torch.zeros(expert_count, device=device) for task in ROUTING_TASKS}
    entropy_sums = {task: 0.0 for task in ROUTING_TASKS}
    rows = 0
    batches = 0
    examples: list[dict[str, Any]] = []
    for batched_data in valid_data:
        interaction = batched_data[0].to(device)
        seq_output = model.shared_representation(interaction)
        weights = model.routing_weights_from_shared(seq_output)
        batch_size = len(interaction)
        for task, task_weights in weights.items():
            prob_sums[task] += task_weights.sum(dim=0)
            probs = task_weights.clamp_min(1e-12)
            entropy_sums[task] += float((-(probs * probs.log()).sum(dim=-1)).sum().cpu().item())
        while len(examples) < example_limit and len(examples) < rows + batch_size:
            local_idx = len(examples) - rows
            record: dict[str, Any] = {"example_index": len(examples)}
            for field in ("user_id", model.POS_ITEM_ID, "source_row_id"):
                if field in interaction.interaction:
                    record[f"{field}_internal"] = int(interaction[field][local_idx].detach().cpu().item())
            record["routing"] = {
                TASK_LABELS[task]: {
                    expert: float(weights[task][local_idx, expert_idx].detach().cpu().item())
                    for expert_idx, expert in enumerate(EXPERTS)
                }
                for task in ROUTING_TASKS
            }
            examples.append(record)
        rows += batch_size
        batches += 1
        if batches >= max_batches:
            break
    if rows == 0:
        raise RuntimeError("No validation rows for routing diagnostics.")
    mean_probs = {
        TASK_LABELS[task]: {
            expert: float((prob_sums[task] / rows)[idx].detach().cpu().item())
            for idx, expert in enumerate(EXPERTS)
        }
        for task in ROUTING_TASKS
    }
    entropy = {
        TASK_LABELS[task]: {
            "mean_entropy": entropy_sums[task] / rows,
            "normalized_entropy": (entropy_sums[task] / rows) / log_experts,
            "max_expert_share": max(mean_probs[TASK_LABELS[task]].values()),
            "min_expert_share": min(mean_probs[TASK_LABELS[task]].values()),
            "experts_above_5pct": sum(value >= 0.05 for value in mean_probs[TASK_LABELS[task]].values()),
        }
        for task in ROUTING_TASKS
    }
    global_usage = {
        expert: sum(mean_probs[TASK_LABELS[task]][expert] for task in ROUTING_TASKS) / len(ROUTING_TASKS)
        for expert in EXPERTS
    }
    distances = task_distances(mean_probs)
    collapse = {
        "thresholds": {
            "expert_collapse_global_max_gt": 0.90,
            "dead_expert_global_min_lt": 0.01,
            "shared_domination_global_gt": 0.70,
            "all_task_same_max_l1_lt": 0.001,
        },
        "expert_collapse": max(global_usage.values()) > 0.90,
        "dead_expert": min(global_usage.values()) < 0.01,
        "shared_expert_domination": global_usage["shared"] > 0.70,
        "all_task_same_routing": (max(item["l1"] for item in distances) if distances else 0.0) < 0.001,
        "minimum_experts_used_per_task": min(item["experts_above_5pct"] for item in entropy.values()),
    }
    semantic = {}
    for task, expected_expert in SEMANTIC_EXPECTATIONS.items():
        values = mean_probs[task]
        top_expert = max(values, key=values.get)
        semantic[task] = {
            "expected_expert": expected_expert,
            "expected_share": values[expected_expert],
            "top_expert": top_expert,
            "top_share": values[top_expert],
            "matches_expected_top": top_expert == expected_expert,
            "gap_expected_minus_mean_other": values[expected_expert]
            - (sum(value for expert, value in values.items() if expert != expected_expert) / (len(values) - 1)),
        }
    rank_values = mean_probs["ranking"]
    rank_top = max(rank_values, key=rank_values.get)
    semantic["ranking"] = {
        "top_expert": rank_top,
        "top_share": rank_values[rank_top],
        "mixture": max(rank_values.values()) < 0.50,
    }
    model.train(was_training)
    return {
        "batches": batches,
        "rows": rows,
        "experts": list(EXPERTS),
        "mean_probabilities": mean_probs,
        "entropy": entropy,
        "global_expert_utilization": global_usage,
        "task_routing_distances": distances,
        "required_task_routing_distances": [
            item for item in distances if item["required_pair"]
        ],
        "strongest_task_difference": max(distances, key=lambda item: item["l1"]) if distances else None,
        "collapse_checks": collapse,
        "semantic_specialization": semantic,
        "routing_examples": examples,
    }


def write_routing_csv(path: Path, epochs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "task",
                "expert",
                "mean_probability",
                "mean_entropy",
                "normalized_entropy",
                "global_expert_utilization",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for epoch in epochs:
            routing = epoch["routing"]
            for task, values in routing["mean_probabilities"].items():
                for expert, probability in values.items():
                    writer.writerow(
                        {
                            "epoch": epoch["epoch"],
                            "task": task,
                            "expert": expert,
                            "mean_probability": probability,
                            "mean_entropy": routing["entropy"][task]["mean_entropy"],
                            "normalized_entropy": routing["entropy"][task]["normalized_entropy"],
                            "global_expert_utilization": routing["global_expert_utilization"][expert],
                        }
                    )


def reference_epoch5(path: Path, run_id: str) -> dict[str, Any]:
    payload = load_json(path)
    if run_id == "multitask_tim4rec_tuned_001":
        metrics = payload["validation_reproduction"]["reproduced_validation"]
        return {
            "run_id": run_id,
            "run_type": "full_budget_validation_reference",
            "epoch": payload["validation_reproduction"].get("best_epoch"),
            "actual_epochs": payload["validation_reproduction"].get("actual_epochs"),
            "validation_metrics": normalize_metrics(metrics),
            "limitation": "Full-budget tuned reference; не 5-epoch sanity.",
        }
    epochs = payload.get("epochs", [])
    if len(epochs) < 5:
        raise RuntimeError(f"Reference has fewer than 5 epochs: {path}")
    epoch5 = next((item for item in epochs if int(item["epoch"]) == 5), epochs[-1])
    validation = epoch5.get("validation") or epoch5.get("validation_metrics")
    return {
        "run_id": run_id,
        "run_type": "5_epoch_sanity_reference",
        "epoch": 5,
        "validation_metrics": normalize_metrics(validation),
        "train_time_sec": epoch5.get("train_time_sec"),
        "validation_time_sec": epoch5.get("validation_time_sec"),
        "gpu": payload.get("gpu"),
        "slurm": payload.get("slurm"),
    }


def load_reference_metrics() -> dict[str, Any]:
    return {
        run_id: reference_epoch5(path, run_id)
        for run_id, path in REFERENCE_RUNS.items()
    }


def parameter_count_summary(config: Config, train_dataset: Any, behavior_model: BehaviorMoETiM4Rec) -> dict[str, Any]:
    device = config["device"]
    base_model = TiM4Rec(config, train_dataset).to(device)
    multitask_model = MultitaskTiM4Rec(config, train_dataset).to(device)
    base = count_parameters(base_model)
    multitask = count_parameters(multitask_model)
    behavior = count_parameters(behavior_model)
    groups = count_by_group(behavior_model)
    del base_model, multitask_model
    torch.cuda.empty_cache()
    return {
        "tim4rec": base,
        "tuned_multitask": multitask,
        "behavior_moe": behavior,
        "behavior_moe_groups": groups,
        "delta_vs_tuned_multitask": int(behavior["total"] - multitask["total"]),
        "relative_increase_vs_tuned_multitask_pct": (behavior["total"] - multitask["total"]) / multitask["total"] * 100.0,
    }


def metric_table(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['HR@5']:.4f} | {metrics['HR@10']:.4f} | {metrics['HR@20']:.4f} | {metrics['HR@50']:.4f} | "
        f"{metrics['Recall@5']:.4f} | {metrics['Recall@10']:.4f} | {metrics['Recall@20']:.4f} | {metrics['Recall@50']:.4f} | "
        f"{metrics['NDCG@5']:.4f} | {metrics['NDCG@10']:.4f} | {metrics['NDCG@20']:.4f} | {metrics['NDCG@50']:.4f}"
    )


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_notes(result: dict[str, Any]) -> str:
    final_epoch = result["epochs"][-1]
    best = result["best_validation"]
    lines = [
        "# Behavior-MoE TiM4Rec sanity 001",
        "",
        "## Цель",
        "",
        "Проверить 5-epoch trajectory для plain Behavior-MoE без load balancing, Optuna, изменения backbone и доступа к test.",
        "",
        "## Архитектура",
        "",
        f"- Experts: `{', '.join(result['architecture']['experts'])}`.",
        f"- Router: `{result['architecture']['router']}`.",
        f"- Residual: `{result['architecture']['residual']}`.",
        f"- Load balancing: `{result['architecture']['load_balance']['enabled']}`.",
        f"- Residual scale: `{result['architecture']['behavior_moe']['residual_scale']}` fixed.",
        "",
        "## Данные",
        "",
        f"- Protocol B identity hash: `{result['dataset']['identity_hash_expected']}`.",
        f"- Train rows: `{result['dataset']['fingerprint_expected']['train']}`.",
        f"- Validation rows: `{result['dataset']['fingerprint_expected']['validation']}`.",
        f"- Test rows в validation-only benchmark: `{result['dataset']['validation_only_summary']['rows']['test']}`.",
        "",
        "## Обучение",
        "",
        f"- Epochs: `{result['actual_epochs']}`.",
        f"- Train batches: `{result['dataset']['loader']['train_batches']}`.",
        f"- Batch size: `{result['config']['train_batch_size']}`.",
        f"- Best epoch by NDCG@10: `{result['best_epoch']}`.",
        f"- Epoch 5 NDCG@10: `{final_epoch['validation_metrics']['NDCG@10']:.6f}`.",
        "",
        "| epoch | total | ranking | click | long_view | like | profile |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in result["epochs"]:
        losses = epoch["losses"]
        lines.append(
            f"| {epoch['epoch']} | {losses['total']:.4f} | {losses['rank']:.4f} | "
            f"{losses['is_click_loss']:.4f} | {losses['long_view_loss']:.4f} | "
            f"{losses['is_like_loss']:.4f} | {losses['is_profile_enter_loss']:.4f} |"
        )
    lines += [
        "",
        "## Validation trajectory",
        "",
        "| epoch | HR@5 | HR@10 | HR@20 | HR@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in result["epochs"]:
        lines.append(f"| {epoch['epoch']} | {metric_table(epoch['validation_metrics'])} |")
    lines += [
        "",
        "## Auxiliary tasks",
        "",
        "| epoch | task | ROC-AUC | PR-AUC | BCE | positive rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for epoch in result["epochs"]:
        for target, metrics in epoch["auxiliary_validation"].items():
            lines.append(
                f"| {epoch['epoch']} | `{target}` | {fmt(metrics['roc_auc'])} | {fmt(metrics['pr_auc'])} | "
                f"{fmt(metrics['bce_loss'])} | {fmt(metrics['positive_rate'])} |"
            )
    lines += [
        "",
        "## Routing trajectory",
        "",
    ]
    for epoch in result["epochs"]:
        lines += [
            f"Epoch {epoch['epoch']}:",
            "",
            "| task | interest | consumption | positive | shared |",
            "|---|---:|---:|---:|---:|",
        ]
        for task in ("ranking", "click", "long_view", "like", "profile"):
            values = epoch["routing"]["mean_probabilities"][task]
            lines.append(
                f"| {task} | {values['interest']:.4f} | {values['consumption']:.4f} | "
                f"{values['positive']:.4f} | {values['shared']:.4f} |"
            )
        lines.append("")
    lines += [
        "## Entropy",
        "",
        "| epoch | ranking | click | long_view | like | profile |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in result["epochs"]:
        entropy = epoch["routing"]["entropy"]
        lines.append(
            f"| {epoch['epoch']} | {entropy['ranking']['normalized_entropy']:.4f} | "
            f"{entropy['click']['normalized_entropy']:.4f} | {entropy['long_view']['normalized_entropy']:.4f} | "
            f"{entropy['like']['normalized_entropy']:.4f} | {entropy['profile']['normalized_entropy']:.4f} |"
        )
    lines += [
        "",
        "## Expert utilization",
        "",
        "| epoch | interest | consumption | positive | shared | collapse | dead expert | shared domination |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for epoch in result["epochs"]:
        util = epoch["routing"]["global_expert_utilization"]
        collapse = epoch["routing"]["collapse_checks"]
        lines.append(
            f"| {epoch['epoch']} | {util['interest']:.4f} | {util['consumption']:.4f} | "
            f"{util['positive']:.4f} | {util['shared']:.4f} | `{collapse['expert_collapse']}` | "
            f"`{collapse['dead_expert']}` | `{collapse['shared_expert_domination']}` |"
        )
    lines += [
        "",
        "## Specialization",
        "",
        f"- Strongest final pair: `{final_epoch['routing']['strongest_task_difference']['left']}` vs "
        f"`{final_epoch['routing']['strongest_task_difference']['right']}`, "
        f"L1 `{final_epoch['routing']['strongest_task_difference']['l1']:.6f}`.",
        f"- Specialization L1 trend: `{result['specialization']['mean_required_pair_l1_by_epoch']}`.",
        f"- Semantic matches at epoch 5: `{result['specialization']['epoch5_semantic_matches']}`.",
        "",
        "## Gradient diagnostics",
        "",
        "| epoch | all experts | all routers | expert max norm | router max norm |",
        "|---:|---|---|---:|---:|",
    ]
    for diag in result["gradient_diagnostics"]:
        expert_max = max(item["gradient_norm"] for item in diag["experts"].values())
        router_max = max(item["gradient_norm"] for item in diag["routers"].values())
        lines.append(
            f"| {diag['epoch']} | `{diag['all_experts_receive_gradients']}` | "
            f"`{diag['all_routers_receive_gradients']}` | {expert_max:.6f} | {router_max:.6f} |"
        )
    lines += [
        "",
        "## Comparison with TiM4Rec / Multitask",
        "",
        "| run | type | epoch | HR@10 | NDCG@10 | NDCG@20 | NDCG@50 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run_id in ("tim4rec_sanity_001", "multitask_tim4rec_sanity_001", "behavior_moe_sanity_001", "multitask_tim4rec_tuned_001"):
        comp = result["baseline_comparison"][run_id]
        metrics = comp["validation_metrics"]
        lines.append(
            f"| `{run_id}` | {comp['run_type']} | {comp.get('epoch', 'n/a')} | "
            f"{metrics['HR@10']:.4f} | {metrics['NDCG@10']:.4f} | {metrics['NDCG@20']:.4f} | {metrics['NDCG@50']:.4f} |"
        )
    lines += [
        "",
        "## Cost",
        "",
        f"- Mean train epoch: `{result['cost']['mean_train_epoch_time_sec']:.3f}` sec.",
        f"- Mean validation: `{result['cost']['mean_validation_time_sec']:.3f}` sec.",
        f"- Peak VRAM: `{result['gpu']['peak_allocated_bytes']}` bytes.",
        f"- Process MaxRSS: `{result['memory']['process_ru_maxrss_kb']}` KB.",
        f"- Params overhead: `{result['parameter_counts']['delta_vs_tuned_multitask']}` "
        f"(`{result['parameter_counts']['relative_increase_vs_tuned_multitask_pct']:.2f}%`).",
        "",
        "## Risks",
        "",
        f"- Collapse: `{result['risk_checks']['router_collapse']}`.",
        f"- Dead expert: `{result['risk_checks']['dead_expert']}`.",
        f"- Shared domination: `{result['risk_checks']['shared_expert_domination']}`.",
        f"- NDCG@10 падение против multitask sanity epoch 5: `{result['comparison']['delta_epoch5_ndcg10_vs_multitask_sanity']:.6f}`.",
        "",
        "## Decision",
        "",
        result["decision"]["summary"],
    ]
    return "\n".join(lines)


def compute_specialization(epochs: list[dict[str, Any]]) -> dict[str, Any]:
    mean_required_l1 = {}
    for epoch in epochs:
        required = epoch["routing"]["required_task_routing_distances"]
        mean_required_l1[str(epoch["epoch"])] = sum(item["l1"] for item in required) / len(required)
    epoch5 = epochs[-1]["routing"]["semantic_specialization"]
    return {
        "mean_required_pair_l1_by_epoch": mean_required_l1,
        "mean_required_pair_l1_delta_epoch5_minus_epoch1": mean_required_l1[str(epochs[-1]["epoch"])]
        - mean_required_l1[str(epochs[0]["epoch"])],
        "epoch5_semantic_matches": {
            task: details.get("matches_expected_top")
            for task, details in epoch5.items()
            if task in SEMANTIC_EXPECTATIONS
        },
        "epoch5_ranking_uses_mixture": bool(epoch5["ranking"]["mixture"]),
    }


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise RuntimeError(f"This script is locked to run_id={RUN_ID}, got {args.run_id}")
    if int(args.epochs) != 5:
        raise RuntimeError(f"This sanity must run exactly 5 epochs, got {args.epochs}")
    if int(args.routing_diagnostic_batches) <= 0:
        raise RuntimeError("routing_diagnostic_batches must be positive.")

    artifact_dir, result_json, notes_path, routing_csv = run_paths(args)
    partial_json = result_json.with_suffix(".partial.json")
    if not args.allow_overwrite:
        for path in (result_json, notes_path, routing_csv, partial_json):
            if path.exists():
                raise RuntimeError(f"Refusing to overwrite existing run artifact: {path}")
        if artifact_dir.exists() and any(artifact_dir.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty artifact dir: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_started = datetime.now(timezone.utc)
    started = time.monotonic()
    behavior_config = load_yaml(Path(args.config))
    optuna_config = load_yaml(project_path(behavior_config["base"]["optuna_config"]))
    best_params = load_yaml(project_path(behavior_config["base"]["best_params"]))
    assert_protocol_config(optuna_config)
    summary = load_json(Path(optuna_config["validation_only_data"]["summary_json"]))
    assert_validation_only_summary(summary)
    smoke_ref = load_json(SMOKE_RUN)
    moe_config = behavior_config["architecture"]["behavior_moe"]
    if smoke_ref["architecture"]["behavior_moe"] != moe_config:
        raise RuntimeError("Behavior-MoE architecture differs from behavior_moe_smoke_001.")
    if float(moe_config.get("load_balance_weight", 0.0)) != 0.0:
        raise RuntimeError(f"Load balancing must stay OFF, got behavior_moe.load_balance_weight={moe_config}")
    if bool(behavior_config["architecture"]["load_balance"]["enabled"]):
        raise RuntimeError("Load balancing config must stay disabled.")

    data = load_data_bundle(optuna_config, artifact_dir / "data_probe")
    sampled = sampled_from_locked_params(best_params, data.target_stats)
    config = build_behavior_config(optuna_config, artifact_dir, sampled, moe_config, int(args.epochs))
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    if tuple(config["multitask_targets"]) != TARGETS:
        raise RuntimeError(f"Task set changed: {config['multitask_targets']}")
    if not bool(config["is_time"]):
        raise RuntimeError("TiM4Rec is_time must remain True.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Behavior-MoE sanity.")
    if data.validation_only_summary.get("identity_hash") != EXPECTED_IDENTITY_HASH:
        raise RuntimeError(f"Identity hash mismatch: {data.validation_only_summary.get('identity_hash')}")
    if int(data.validation_only_summary["rows"]["test"]) != 0:
        raise RuntimeError(f"Validation-only benchmark unexpectedly has test rows: {data.validation_only_summary}")

    train_data, valid_data = create_loaders(config, data.train_dataset, data.valid_dataset)
    device = config["device"]
    pos_weights = pos_weight_tensors(sampled["effective_pos_weights"], device)
    diagnostic_epochs = parse_diagnostic_epochs(args.diagnostic_epochs)
    fixed_diag_batch = first_batch(train_data, device)

    torch.cuda.reset_peak_memory_stats()
    model = BehaviorMoETiM4Rec(config, train_data.dataset).to(device)
    optimizer = optimizer_for_trial(model, sampled)
    trainer = Trainer(config, model)
    trainer.optimizer = optimizer
    param_counts = parameter_count_summary(config, train_data.dataset, model)
    load_balance_weight = 0.0

    epochs: list[dict[str, Any]] = []
    gradient_diagnostics: list[dict[str, Any]] = []
    best_epoch = None
    best_score = -float("inf")
    best_snapshot: dict[str, Any] | None = None
    topk = list(config["topk"])

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.monotonic()
        train_start = time.monotonic()
        losses = train_one_epoch_behavior(
            model,
            optimizer,
            train_data,
            device,
            sampled,
            pos_weights,
            load_balance_weight,
        )
        train_time = time.monotonic() - train_start

        valid_start = time.monotonic()
        valid_result, full_checks = evaluate_full_sort_with_checks(trainer, valid_data, train_data)
        auxiliary_validation = evaluate_auxiliary(model, valid_data, device)
        validation_time = time.monotonic() - valid_start
        hit_recall = check_hit_recall_equal(valid_result, topk)
        if not full_checks["raw_scores_all_finite"] or not full_checks["positive_scores_all_finite"]:
            raise RuntimeError(f"Non-finite validation scores: {full_checks}")

        routing = routing_diagnostics(
            model,
            valid_data,
            max_batches=int(args.routing_diagnostic_batches),
            example_limit=int(args.routing_example_limit),
        )
        if epoch in diagnostic_epochs:
            gradient_diagnostics.append(
                fixed_gradient_diagnostic(
                    model,
                    fixed_diag_batch,
                    sampled,
                    pos_weights,
                    load_balance_weight,
                    epoch,
                )
            )

        metrics = normalize_metrics(metric_subset(valid_result))
        valid_score = float(metrics["NDCG@10"])
        update_flag = valid_score > best_score
        if update_flag:
            best_epoch = epoch
            best_score = valid_score
            best_snapshot = {
                "epoch": epoch,
                "metrics": metrics,
                "compact_metrics": compact_validation(metrics),
                "auxiliary_validation": auxiliary_validation,
                "losses": losses,
                "routing": routing,
                "full_ranking_checks": full_checks,
                "validation_time_sec": validation_time,
            }

        epoch_result = {
            "epoch": epoch,
            "losses": losses,
            "validation": {key.lower(): value for key, value in metrics.items()},
            "validation_metrics": metrics,
            "auxiliary_validation": auxiliary_validation,
            "valid_score": valid_score,
            "valid_metric": "NDCG@10",
            "best_so_far": {"epoch": best_epoch, "NDCG@10": best_score},
            "hit_recall_equal_check": hit_recall,
            "full_ranking_checks": full_checks,
            "routing": routing,
            "train_time_sec": train_time,
            "validation_time_sec": validation_time,
            "epoch_time_sec": time.monotonic() - epoch_start,
            "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()),
        }
        epochs.append(epoch_result)
        save_json(
            partial_json,
            {
                "run_id": args.run_id,
                "status": "partial",
                "record_type": "sanity",
                "epochs_completed": len(epochs),
                "latest_epoch": epoch_result,
                "best_epoch_so_far": best_epoch,
                "best_valid_score_so_far": best_score,
                "test_evaluation_count": 0,
            },
        )
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "epoch": epoch,
                    "validation_ndcg10": metrics["NDCG@10"],
                    "validation_hr10": metrics["HR@10"],
                    "train_time_sec": train_time,
                    "validation_time_sec": validation_time,
                    "routing_mean_required_l1": sum(
                        item["l1"] for item in routing["required_task_routing_distances"]
                    )
                    / len(routing["required_task_routing_distances"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if best_snapshot is None or best_epoch is None:
        raise RuntimeError("No validation snapshot recorded.")
    if len(epochs) != 5:
        raise RuntimeError(f"Expected exactly 5 epochs, got {len(epochs)}")

    references = load_reference_metrics()
    behavior_epoch5_metrics = epochs[-1]["validation_metrics"]
    references[RUN_ID] = {
        "run_id": RUN_ID,
        "run_type": "5_epoch_sanity",
        "epoch": 5,
        "validation_metrics": behavior_epoch5_metrics,
        "train_time_sec": epochs[-1]["train_time_sec"],
        "validation_time_sec": epochs[-1]["validation_time_sec"],
    }
    specialization = compute_specialization(epochs)
    final_routing = epochs[-1]["routing"]
    collapse = final_routing["collapse_checks"]
    all_grad_ok = all(
        diag["all_gradients_finite"]
        and diag["all_experts_receive_gradients"]
        and diag["all_routers_receive_gradients"]
        for diag in gradient_diagnostics
    )
    aux_epoch5 = epochs[-1]["auxiliary_validation"]
    aux_not_catastrophic = all(
        metrics["roc_auc"] is not None
        and metrics["roc_auc"] >= 0.5
        and metrics["pr_auc"] is not None
        and metrics["pr_auc"] >= metrics["random_pr_auc_baseline"]
        for metrics in aux_epoch5.values()
    )
    ndcg_growth = epochs[-1]["validation_metrics"]["NDCG@10"] > epochs[0]["validation_metrics"]["NDCG@10"]
    delta_vs_multitask = (
        epochs[-1]["validation_metrics"]["NDCG@10"]
        - references["multitask_tim4rec_sanity_001"]["validation_metrics"]["NDCG@10"]
    )
    ranking_not_crashed = delta_vs_multitask > -0.01
    no_collapse = (
        not collapse["expert_collapse"]
        and not collapse["dead_expert"]
        and not collapse["shared_expert_domination"]
    )
    runtime_sec = time.monotonic() - started
    mean_train = sum(item["train_time_sec"] for item in epochs) / len(epochs)
    mean_valid = sum(item["validation_time_sec"] for item in epochs) / len(epochs)
    multitask_ref = references["multitask_tim4rec_sanity_001"]
    ref_train_time = multitask_ref.get("train_time_sec")
    runtime_overhead = mean_train / float(ref_train_time) if ref_train_time else None
    ref_gpu = (multitask_ref.get("gpu") or {}).get("name")
    current_gpu = gpu_info().get("name")
    hardware_differs = bool(ref_gpu and current_gpu and ref_gpu != current_gpu)
    pipeline_ready = bool(
        ndcg_growth
        and ranking_not_crashed
        and no_collapse
        and all_grad_ok
        and aux_not_catastrophic
        and specialization["mean_required_pair_l1_delta_epoch5_minus_epoch1"] > 0.0
    )

    result: dict[str, Any] = {
        "run_id": args.run_id,
        "status": "completed" if pipeline_ready else "completed_with_warnings",
        "record_type": "sanity",
        "execution_status": "completed",
        "sanity": True,
        "objective": "validation_full_ranking_NDCG@10",
        "created_at_utc": run_started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": git_value(["config", "--get", "remote.origin.url"]),
            "expected_start_commit": "924f18d759fdcdf656300f24758b4f376fb16e8d",
        },
        "source_files": source_hashes(),
        "environment": environment_info(),
        "slurm": slurm_info(),
        "gpu": gpu_info()
        | {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "memory": {
            "process_ru_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "artifact_dir": str(artifact_dir),
        "base_model": "MultitaskTiM4Rec",
        "base_run": "multitask_tim4rec_tuned_001",
        "model_name": "BehaviorMoETiM4Rec",
        "dataset": {
            "name": "KuaiRand",
            "protocol": "B",
            "fingerprint_expected": EXPECTED_FINGERPRINT,
            "identity_hash_expected": EXPECTED_IDENTITY_HASH,
            "validation_only_summary": data.validation_only_summary,
            "loader_inspection": data.loader_inspection,
            "loader": {
                "train_batches": len(train_data),
                "valid_batches": len(valid_data),
                "train_examples": len(data.train_dataset),
                "validation_examples": len(data.valid_dataset),
                "batch_size": int(config["train_batch_size"]),
                "full_train_used": True,
                "train_subset_used": False,
            },
        },
        "test_safety": {
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
            "test_evaluated": False,
            "test_evaluation_count": 0,
        },
        "test_evaluation_count": 0,
        "architecture": {
            "experts": list(EXPERTS),
            "expert_semantics": {
                "interest": "is_click",
                "consumption": "long_view",
                "positive": "is_like + is_profile_enter",
                "shared": "residual/general",
            },
            "expert_mlp": "Linear(hidden, hidden) -> GELU -> Dropout -> Linear(hidden, hidden)",
            "router": "separate learned Linear(hidden, 4) router head per task; softmax(logits / temperature)",
            "task_conditioned_routing": {
                "tasks": [TASK_LABELS[task] for task in ROUTING_TASKS],
                "current_behavior_labels_used_as_router_input": False,
            },
            "residual": "h_task = h + residual_scale * sum_e p(task,e|h) * expert_e(h)",
            "behavior_moe": moe_config,
            "load_balance": behavior_config["architecture"]["load_balance"] | {"runtime_weight": load_balance_weight},
        },
        "training_config": {
            "source": "multitask_tim4rec_optuna_v1 trial 110 / best_params.yaml",
            "epochs_requested": int(args.epochs),
            "epochs_completed": len(epochs),
            "early_stopping_enabled": False,
            "full_training_after_sanity_started": False,
            "optuna_started": False,
            "hard_topk_routing": False,
            "load_balancing_regularizer": False,
            "entropy_regularizer": False,
            "lambda_aux": sampled["lambda_aux"],
            "learning_rate": sampled["learning_rate"],
            "weight_decay": sampled["weight_decay"],
            "dropout_prob": sampled["dropout_prob"],
            "head_lr_multiplier": sampled["head_lr_multiplier"],
            "head_learning_rate": sampled["head_learning_rate"],
            "normalized_task_weights": sampled["normalized_task_weights"],
            "effective_pos_weights": sampled["effective_pos_weights"],
            "effective_loss_multipliers": sampled["effective_loss_multipliers"],
            "effective_positive_multipliers": sampled["effective_positive_multipliers"],
            "locked_param_diff_checks": sampled["locked_param_diff_checks"],
        },
        "config": {
            "seed": int(config["seed"]),
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "is_time": bool(config["is_time"]),
            "metrics": list(config["metrics"]),
            "topk": topk,
            "eval_args": config["eval_args"],
        },
        "parameter_counts": param_counts,
        "epochs": epochs,
        "actual_epochs": len(epochs),
        "best_epoch": best_epoch,
        "best_valid_score": best_score,
        "best_valid_metric": "NDCG@10",
        "best_validation": best_snapshot,
        "best_validation_metrics": best_snapshot["metrics"],
        "best_validation_compact": best_snapshot["compact_metrics"],
        "best_auxiliary_metrics": best_snapshot["auxiliary_validation"],
        "best_epoch_losses": best_snapshot["losses"],
        "routing_diagnostics": {
            "diagnostic_subset": {
                "split": "validation",
                "batches_per_epoch": int(args.routing_diagnostic_batches),
                "fixed_order": True,
                "raw_sensitive_data_stored": False,
            },
            "per_epoch": [{"epoch": item["epoch"], **item["routing"]} for item in epochs],
        },
        "routing_entropy": {
            str(item["epoch"]): item["routing"]["entropy"]
            for item in epochs
        },
        "task_routing_distances": {
            str(item["epoch"]): item["routing"]["required_task_routing_distances"]
            for item in epochs
        },
        "expert_utilization": {
            str(item["epoch"]): item["routing"]["global_expert_utilization"]
            for item in epochs
        },
        "specialization": specialization,
        "gradient_diagnostics": gradient_diagnostics,
        "baseline_comparison": references,
        "comparison": {
            "epoch5_ndcg10": epochs[-1]["validation_metrics"]["NDCG@10"],
            "delta_epoch5_ndcg10_vs_tim4rec_sanity": epochs[-1]["validation_metrics"]["NDCG@10"]
            - references["tim4rec_sanity_001"]["validation_metrics"]["NDCG@10"],
            "delta_epoch5_ndcg10_vs_multitask_sanity": delta_vs_multitask,
            "delta_best_ndcg10_vs_tuned_full_validation_reference": best_score
            - references["multitask_tim4rec_tuned_001"]["validation_metrics"]["NDCG@10"],
            "comparison_policy": "Главное сравнение sanity-to-sanity: epoch 5 vs epoch 5. Tuned fixed reference является full-budget validation reference.",
        },
        "runtime": {
            "total_sec": runtime_sec,
            "mean_epoch_sec": sum(item["epoch_time_sec"] for item in epochs) / len(epochs),
        },
        "cost": {
            "mean_train_epoch_time_sec": mean_train,
            "mean_validation_time_sec": mean_valid,
            "mean_epoch_time_sec": sum(item["epoch_time_sec"] for item in epochs) / len(epochs),
            "multitask_sanity_epoch5_train_time_sec": ref_train_time,
            "runtime_overhead_vs_multitask_sanity_epoch5_train_time": runtime_overhead,
            "hardware_differs_from_multitask_sanity": hardware_differs,
            "hardware_note": f"Behavior-MoE GPU={current_gpu}; Multitask sanity GPU={ref_gpu}.",
        },
        "risk_checks": {
            "training_stable": True,
            "ndcg10_grew_epoch5_vs_epoch1": ndcg_growth,
            "ranking_not_crashed_vs_multitask_sanity": ranking_not_crashed,
            "router_collapse": collapse["expert_collapse"],
            "dead_expert": collapse["dead_expert"],
            "shared_expert_domination": collapse["shared_expert_domination"],
            "all_task_same_routing": collapse["all_task_same_routing"],
            "all_experts_and_routers_receive_gradients": all_grad_ok,
            "auxiliary_not_catastrophic": aux_not_catastrophic,
            "parameter_overhead_acceptable": param_counts["relative_increase_vs_tuned_multitask_pct"] < 10.0,
        },
        "decision": {
            "pipeline_ready_for_full_run": pipeline_ready,
            "behavior_moe_improves_5_epoch_ranking_vs_multitask_sanity": delta_vs_multitask > 0.0,
            "real_task_specialization_observed": specialization["mean_required_pair_l1_delta_epoch5_minus_epoch1"] > 0.0,
            "load_balancing_regularizer_needed_now": False,
            "recommended_next_step": "full plain Behavior-MoE"
            if pipeline_ready and not collapse["expert_collapse"]
            else "analyze routing architecture before full run",
            "summary": (
                "5-epoch sanity прошёл стабильно: можно переходить к full plain Behavior-MoE без load balancing."
                if pipeline_ready
                else "5-epoch sanity завершён, но есть предупреждения; full run лучше не запускать до анализа trajectory."
            ),
        },
    }
    save_json(result_json, result)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(build_notes(result) + "\n", encoding="utf-8")
    write_routing_csv(routing_csv, epochs)
    if partial_json.exists():
        partial_json.unlink()


if __name__ == "__main__":
    main()
