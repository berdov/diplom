#!/usr/bin/env python
"""Smoke test for Behavior-MoE TiM4Rec on real KuaiRand train batches."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader
from recbole.utils import init_seed


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tim4rec import TiM4Rec  # noqa: E402
from experiments.behavior_moe_tim4rec.model import (  # noqa: E402
    STRUCTURED_ALLOWED_EXPERTS,
    ROUTING_TASKS,
    BehaviorMoETiM4Rec,
    StructuredBehaviorMoETiM4Rec,
)
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS  # noqa: E402
from experiments.multitask_tim4rec.train import (  # noqa: E402
    EXPECTED_FINGERPRINT,
    EXPECTED_IDENTITY_HASH,
    all_gradient_check,
    count_parameters,
    load_json,
    load_target_stats,
    sha256_file,
    tensor_to_float,
)
from experiments.multitask_tim4rec_optuna.optuna_search import (  # noqa: E402
    assert_validation_only_summary,
    compute_tuned_losses,
    create_loaders,
    load_yaml,
    optimizer_for_trial,
    pos_weight_tensors,
    project_path,
    recbole_overrides,
)
from experiments.multitask_tim4rec_optuna.run_locked_tuned import sampled_from_locked_params  # noqa: E402


RUN_ID = "behavior_moe_smoke_001"
STRUCTURED_RUN_ID = "structured_behavior_moe_smoke_001"
TASK_LABELS = {
    "rank": "ranking",
    "is_click": "click",
    "long_view": "long_view",
    "is_like": "like",
    "is_profile_enter": "profile",
}
TASK_DISPLAY_ORDER = ("ranking", "click", "long_view", "like", "profile")
HEAD_PREFIX = {
    "is_click": "click_head.",
    "long_view": "long_view_head.",
    "is_like": "like_head.",
    "is_profile_enter": "profile_enter_head.",
}
DEFAULT_OUTPUT = ROOT / "experiments" / "behavior_moe_tim4rec" / "runs" / f"{RUN_ID}.json"
DEFAULT_NOTES = ROOT / "experiments" / "behavior_moe_tim4rec" / "runs" / f"{RUN_ID}_notes.md"
GENERIC_REFERENCE_PATH = ROOT / "experiments" / "behavior_moe_tim4rec" / "runs" / "behavior_moe_smoke_001.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments" / "behavior_moe_tim4rec" / "config.yaml"))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--notes", default=str(DEFAULT_NOTES))
    parser.add_argument("--artifact-dir", default="/home/daryumin/iberdov/diplom/experiments/behavior_moe_tim4rec/behavior_moe_smoke_001")
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--routing-mode", choices=("generic", "structured"), default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def relative_path(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def git_value(args: list[str], default: str = "unknown") -> str:
    env_map = {
        ("rev-parse", "HEAD"): "BEHAVIOR_MOE_GIT_COMMIT",
        ("rev-parse", "--abbrev-ref", "HEAD"): "BEHAVIOR_MOE_GIT_BRANCH",
        ("config", "--get", "remote.origin.url"): "BEHAVIOR_MOE_GIT_REMOTE",
    }
    try:
        value = subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        if value:
            return value
    except Exception:
        pass
    env_key = env_map.get(tuple(args))
    return os.environ.get(env_key, default) if env_key else default


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


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def max_cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"max_allocated_bytes": 0, "max_reserved_bytes": 0}
    return {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def trimmed_mean(values: list[float]) -> float:
    if len(values) < 3:
        return mean(values)
    ordered = sorted(values)
    return mean(ordered[1:-1])


def resolve_variant(args: argparse.Namespace, behavior_config: dict[str, Any]) -> dict[str, Any]:
    routing_mode = args.routing_mode
    if routing_mode is None:
        if args.run_id == STRUCTURED_RUN_ID:
            routing_mode = "structured"
        else:
            routing_mode = str(behavior_config["architecture"]["behavior_moe"].get("routing_mode", "generic"))
    if routing_mode == "structured":
        smoke_config = dict(behavior_config.get("structured_smoke", behavior_config["smoke"]))
        moe_config = dict(behavior_config["architecture"]["structured_behavior_moe"])
        model_cls: type[BehaviorMoETiM4Rec] = StructuredBehaviorMoETiM4Rec
        default_artifact = f"/home/daryumin/iberdov/diplom/experiments/behavior_moe_tim4rec/{STRUCTURED_RUN_ID}"
    else:
        smoke_config = dict(behavior_config["smoke"])
        moe_config = dict(behavior_config["architecture"]["behavior_moe"])
        model_cls = BehaviorMoETiM4Rec
        default_artifact = f"/home/daryumin/iberdov/diplom/experiments/behavior_moe_tim4rec/{RUN_ID}"
    run_id = args.run_id
    if run_id == RUN_ID and routing_mode == "structured":
        run_id = STRUCTURED_RUN_ID
    output = Path(args.output)
    notes = Path(args.notes)
    if output == DEFAULT_OUTPUT and run_id != RUN_ID:
        output = ROOT / "experiments" / "behavior_moe_tim4rec" / "runs" / f"{run_id}.json"
    if notes == DEFAULT_NOTES and run_id != RUN_ID:
        notes = ROOT / "experiments" / "behavior_moe_tim4rec" / "runs" / f"{run_id}_notes.md"
    artifact_dir = Path(args.artifact_dir)
    if args.artifact_dir.endswith("behavior_moe_smoke_001") and run_id != RUN_ID:
        artifact_dir = Path(default_artifact)
    return {
        "run_id": run_id,
        "routing_mode": routing_mode,
        "smoke_config": smoke_config,
        "moe_config": moe_config,
        "model_cls": model_cls,
        "output": output,
        "notes": notes,
        "artifact_dir": artifact_dir,
    }


def enforce_artifact_policy(path: Path, notes: Path, *, allow_overwrite: bool) -> None:
    existing = [str(candidate) for candidate in (path, notes) if candidate.exists()]
    if existing and not allow_overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing smoke artifact(s): " + ", ".join(existing)
        )


def build_behavior_config(
    optuna_config: dict[str, Any],
    artifact_root: Path,
    sampled: dict[str, Any],
    moe_config: dict[str, Any],
    model_cls: type[BehaviorMoETiM4Rec],
) -> Config:
    base_config = project_path(optuna_config["source"]["base_config"])
    overrides = recbole_overrides(optuna_config, artifact_root, sampled)
    overrides["behavior_moe"] = moe_config
    overrides["final_test_evaluation_count"] = 0
    overrides["test_evaluation_count"] = 0
    return Config(
        model=model_cls,
        config_file_list=[str(base_config)],
        config_dict=overrides,
    )


def validation_only_train_dataset(config: Config, summary: dict[str, Any]) -> tuple[Any, Any, Any]:
    assert_validation_only_summary(summary)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    dataset = create_dataset(config)
    built = dataset.build()
    if len(built) == 2:
        train_dataset, valid_dataset = built
    elif len(built) == 3:
        train_dataset, valid_dataset, unused_dataset = built
        if len(unused_dataset) != 0:
            raise RuntimeError(f"Validation-only split created non-empty unused split: {len(unused_dataset)}")
    else:
        raise RuntimeError(f"Expected train/valid validation-only splits, got {len(built)}")
    expected_train_examples = EXPECTED_FINGERPRINT["train"] - EXPECTED_FINGERPRINT["users"]
    if len(train_dataset) != expected_train_examples:
        raise RuntimeError(f"Sequential train examples changed: {len(train_dataset)} != {expected_train_examples}")
    if len(valid_dataset) != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Validation examples changed: {len(valid_dataset)}")
    train_loader = get_dataloader(config, "train")(config, train_dataset, None, shuffle=config["shuffle"])
    return dataset, train_dataset, train_loader


def collect_batches(train_data: Any, device: torch.device, count: int) -> list[Any]:
    if count < 3 or count > 10:
        raise ValueError(f"Smoke batches must stay in [3, 10], got {count}")
    batches = []
    for interaction in train_data:
        batches.append(interaction.to(device))
        if len(batches) >= count:
            break
    if len(batches) != count:
        raise RuntimeError(f"Expected {count} train batches, got {len(batches)}")
    return batches


def load_tuned_checkpoint(model: torch.nn.Module, path: Path, device: torch.device, *, allow_moe_missing: bool) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint has no state_dict: {path}")
    if allow_moe_missing:
        loaded = model.load_state_dict(state_dict, strict=False)
        unexpected = list(loaded.unexpected_keys)
        missing = list(loaded.missing_keys)
        allowed_prefixes = ("experts.", "router_heads.")
        disallowed_missing = [name for name in missing if not name.startswith(allowed_prefixes)]
        if unexpected or disallowed_missing:
            raise RuntimeError({"unexpected": unexpected, "disallowed_missing": disallowed_missing[:20]})
        load_mode = "strict_false_extra_moe_params_initialized"
    else:
        model.load_state_dict(state_dict, strict=True)
        missing = []
        unexpected = []
        load_mode = "strict_true"
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "best_valid_score": checkpoint.get("best_valid_score"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "load_mode": load_mode,
        "missing_initialized_parameter_count": len(missing),
        "missing_initialized_parameters_sample": missing[:16],
        "unexpected_parameters": unexpected,
    }


def new_baseline_model(config: Config, train_dataset: Any, checkpoint_path: Path, sampled: dict[str, Any]) -> tuple[MultitaskTiM4Rec, torch.optim.Optimizer, dict[str, Any]]:
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    device = config["device"]
    model = MultitaskTiM4Rec(config, train_dataset).to(device)
    checkpoint_info = load_tuned_checkpoint(model, checkpoint_path, device, allow_moe_missing=False)
    model.train()
    return model, optimizer_for_trial(model, sampled), checkpoint_info


def new_moe_model(
    config: Config,
    train_dataset: Any,
    checkpoint_path: Path,
    sampled: dict[str, Any],
    model_cls: type[BehaviorMoETiM4Rec],
) -> tuple[BehaviorMoETiM4Rec, torch.optim.Optimizer, dict[str, Any]]:
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    device = config["device"]
    model = model_cls(config, train_dataset).to(device)
    checkpoint_info = load_tuned_checkpoint(model, checkpoint_path, device, allow_moe_missing=True)
    model.train()
    return model, optimizer_for_trial(model, sampled), checkpoint_info


def grouped_parameter_names(model: torch.nn.Module) -> dict[str, list[str]]:
    groups = {"backbone": [], "experts": [], "router": [], "auxiliary_heads": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("experts."):
            groups["experts"].append(name)
        elif name.startswith("router_heads."):
            groups["router"].append(name)
        elif any(name.startswith(prefix) for prefix in HEAD_PREFIX.values()):
            groups["auxiliary_heads"].append(name)
        else:
            groups["backbone"].append(name)
    return groups


def count_by_group(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    names = grouped_parameter_names(model)
    params = dict(model.named_parameters())
    return {
        group: {
            "parameter_tensors": len(group_names),
            "parameters": int(sum(params[name].numel() for name in group_names)),
        }
        for group, group_names in names.items()
    }


def snapshot_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def update_summary(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> dict[str, Any]:
    names = grouped_parameter_names(model)
    params = dict(model.named_parameters())
    groups: dict[str, Any] = {}
    for group, group_names in names.items():
        sq_norm = 0.0
        updated_tensors = 0
        for name in group_names:
            diff = params[name].detach() - before[name]
            norm = float(torch.linalg.vector_norm(diff.float()).cpu().item())
            sq_norm += norm * norm
            updated_tensors += int(norm > 0.0)
        groups[group] = {
            "parameter_tensors": len(group_names),
            "updated_tensors": updated_tensors,
            "all_tensors_updated": updated_tensors == len(group_names) if group_names else False,
            "update_norm": math.sqrt(sq_norm),
        }
    groups["all_trainable_parameters_finite"] = all(
        torch.isfinite(param.detach()).all().item()
        for param in model.parameters()
        if param.requires_grad
    )
    return groups


def gradient_group_summary(model: torch.nn.Module) -> dict[str, Any]:
    names = grouped_parameter_names(model)
    params = dict(model.named_parameters())
    groups: dict[str, Any] = {}
    nonfinite = []
    for group, group_names in names.items():
        grad_tensors = 0
        sq_norm = 0.0
        missing = []
        for name in group_names:
            grad = params[name].grad
            if grad is None:
                missing.append(name)
                continue
            grad_tensors += 1
            if not torch.isfinite(grad).all().item():
                nonfinite.append(name)
            norm = float(torch.linalg.vector_norm(grad.detach().float()).cpu().item())
            sq_norm += norm * norm
        groups[group] = {
            "parameter_tensors": len(group_names),
            "gradient_tensors": grad_tensors,
            "missing_gradient_tensors": len(missing),
            "missing_gradient_sample": missing[:8],
            "all_parameters_have_gradient": grad_tensors == len(group_names) if group_names else False,
            "gradient_norm": math.sqrt(sq_norm),
        }
    groups["all_finite"] = not nonfinite
    groups["nonfinite_gradient_sample"] = nonfinite[:8]
    return groups


def auxiliary_head_gradient_summary(model: BehaviorMoETiM4Rec) -> dict[str, Any]:
    result = {}
    for target, head in model.auxiliary_heads().items():
        sq_norm = 0.0
        tensors = 0
        missing = 0
        for param in head.parameters():
            if param.grad is None:
                missing += 1
                continue
            tensors += 1
            sq_norm += float(torch.linalg.vector_norm(param.grad.detach().float()).cpu().item()) ** 2
        result[target] = {
            "gradient_tensors": tensors,
            "missing_gradient_tensors": missing,
            "gradient_norm": math.sqrt(sq_norm),
            "receives_gradient": tensors > 0 and missing == 0 and sq_norm > 0.0,
        }
    return result


def module_grad_norm(module: torch.nn.Module) -> dict[str, Any]:
    sq_norm = 0.0
    tensors = 0
    missing = 0
    nonfinite = 0
    for param in module.parameters():
        if param.grad is None:
            missing += 1
            continue
        tensors += 1
        grad = param.grad.detach().float()
        nonfinite += int(not torch.isfinite(grad).all().item())
        sq_norm += float(torch.linalg.vector_norm(grad).cpu().item()) ** 2
    norm = math.sqrt(sq_norm)
    return {
        "gradient_tensors": tensors,
        "missing_gradient_tensors": missing,
        "gradient_norm": norm,
        "all_finite": nonfinite == 0,
        "receives_gradient": tensors > 0 and missing == 0 and nonfinite == 0 and norm > 0.0,
    }


def expert_gradient_norms(model: BehaviorMoETiM4Rec) -> dict[str, Any]:
    return {name: module_grad_norm(model.experts[name]) for name in model.expert_names}


def router_gradient_norms(model: BehaviorMoETiM4Rec) -> dict[str, Any]:
    return {TASK_LABELS[task]: module_grad_norm(model.router_heads[task]) for task in ROUTING_TASKS}


def expected_structured_sources() -> dict[str, set[str]]:
    return {
        "interest": {"ranking", "click"},
        "consumption": {"ranking", "long_view"},
        "engagement": {"ranking", "like", "profile"},
        "shared": {"ranking", "click", "long_view", "like", "profile"},
    }


def gradient_connectivity_probe(
    model: BehaviorMoETiM4Rec,
    batch: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
) -> dict[str, Any]:
    objective_to_loss = {
        "ranking": "rank",
        "click": "is_click_loss",
        "long_view": "long_view_loss",
        "like": "is_like_loss",
        "profile": "is_profile_enter_loss",
    }
    by_objective = {}
    for objective, loss_key in objective_to_loss.items():
        model.zero_grad(set_to_none=True)
        losses = model.calculate_multitask_loss(
            batch,
            lambda_aux=float(sampled["lambda_aux"]),
            pos_weights=pos_weights,
            task_weights=sampled["normalized_task_weights"],
            load_balance_weight=0.0,
        )
        losses[loss_key].backward()
        by_objective[objective] = {
            "loss": tensor_to_float(losses[loss_key]),
            "expert_gradients": expert_gradient_norms(model),
            "router_gradients": router_gradient_norms(model),
        }

    actual_by_expert = {
        expert: [
            objective
            for objective, row in by_objective.items()
            if row["expert_gradients"][expert]["receives_gradient"]
        ]
        for expert in model.expert_names
    }
    actual_by_router = {
        router: [
            objective
            for objective, row in by_objective.items()
            if row["router_gradients"][router]["receives_gradient"]
        ]
        for router in TASK_DISPLAY_ORDER
    }
    if model.routing_mode == "structured":
        expected_by_expert = expected_structured_sources()
        expected_by_router = {task: {task} for task in TASK_DISPLAY_ORDER}
        expert_connectivity_met = all(
            set(actual_by_expert.get(expert, [])) == expected
            for expert, expected in expected_by_expert.items()
        )
        router_connectivity_met = all(
            set(actual_by_router.get(router, [])) == expected
            for router, expected in expected_by_router.items()
        )
    else:
        expected_by_expert = {
            expert: set(objective_to_loss)
            for expert in model.expert_names
        }
        expected_by_router = {task: {task} for task in TASK_DISPLAY_ORDER}
        expert_connectivity_met = all(
            set(actual_by_expert.get(expert, [])) == expected
            for expert, expected in expected_by_expert.items()
        )
        router_connectivity_met = all(
            set(actual_by_router.get(router, [])) == expected
            for router, expected in expected_by_router.items()
        )
    model.zero_grad(set_to_none=True)
    return {
        "by_objective": by_objective,
        "expected_by_expert": {key: sorted(value) for key, value in expected_by_expert.items()},
        "actual_by_expert": actual_by_expert,
        "expected_by_router": {key: sorted(value) for key, value in expected_by_router.items()},
        "actual_by_router": actual_by_router,
        "expert_connectivity_met": expert_connectivity_met,
        "router_connectivity_met": router_connectivity_met,
        "expected_connectivity_met": expert_connectivity_met and router_connectivity_met,
    }


def float_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: tensor_to_float(value) for key, value in losses.items()}


def warmup_fixed_tuned(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
) -> None:
    optimizer.zero_grad(set_to_none=True)
    losses = compute_tuned_losses(model, batch, sampled, pos_weights)
    losses["total"].backward()
    optimizer.zero_grad(set_to_none=True)
    sync_cuda()


def warmup_behavior_moe(
    model: BehaviorMoETiM4Rec,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    load_balance_weight: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    losses = model.calculate_multitask_loss(
        batch,
        lambda_aux=float(sampled["lambda_aux"]),
        pos_weights=pos_weights,
        task_weights=sampled["normalized_task_weights"],
        load_balance_weight=load_balance_weight,
    )
    losses["total"].backward()
    optimizer.zero_grad(set_to_none=True)
    sync_cuda()


def run_fixed_tuned_steps(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    batches: list[Any],
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
) -> dict[str, Any]:
    warmup_fixed_tuned(model, optimizer, batches[0], sampled, pos_weights)
    reset_cuda_peak()
    step_times = []
    all_gradients_finite = True
    last_losses: dict[str, float] = {}
    start_total = time.monotonic()
    for batch in batches:
        sync_cuda()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = compute_tuned_losses(model, batch, sampled, pos_weights)
        if not all(math.isfinite(value) for value in float_losses(losses).values()):
            raise RuntimeError("Non-finite tuned fixed loss in smoke.")
        losses["total"].backward()
        grad_check = all_gradient_check(model)
        all_gradients_finite = all_gradients_finite and bool(grad_check["all_finite"])
        optimizer.step()
        step_times.append(time.monotonic() - start)
        sync_cuda()
        last_losses = float_losses(losses)
    return {
        "status": "completed",
        "steps": len(batches),
        "warmup_forward_backward_not_counted": True,
        "mean_step_time_sec": mean(step_times),
        "trimmed_mean_step_time_sec": trimmed_mean(step_times),
        "step_times_sec": step_times,
        "total_time_sec": time.monotonic() - start_total,
        "all_gradients_finite": all_gradients_finite,
        "last_losses": last_losses,
        "cuda_memory": max_cuda_memory(),
    }


@torch.no_grad()
def forward_probe(model: BehaviorMoETiM4Rec, batch: Any) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    seq_output = model.shared_representation(batch)
    representations = model.task_representations_from_shared(seq_output)
    rank_logits = model.scores_from_task_representation(representations["rank"])
    aux_logits = model.auxiliary_logits_from_representation(seq_output)
    routing = model.routing_weights_from_shared(seq_output)
    local_routing = model.local_routing_weights_from_shared(seq_output)
    result = {
        "batch_size": len(batch),
        "shared_representation_shape": list(seq_output.shape),
        "moe_representation_shapes": {TASK_LABELS[task]: list(value.shape) for task, value in representations.items()},
        "ranking_scores_shape": list(rank_logits.shape),
        "auxiliary_logit_shapes": {TASK_LABELS[target]: list(logits.shape) for target, logits in aux_logits.items()},
        "routing_weight_shapes": {TASK_LABELS[task]: list(value.shape) for task, value in routing.items()},
        "local_routing_weight_shapes": {TASK_LABELS[task]: list(value.shape) for task, value in local_routing.items()},
        "all_outputs_finite": bool(
            torch.isfinite(seq_output).all().item()
            and torch.isfinite(rank_logits).all().item()
            and all(torch.isfinite(value).all().item() for value in aux_logits.values())
            and all(torch.isfinite(value).all().item() for value in routing.values())
            and all(torch.isfinite(value).all().item() for value in local_routing.values())
        ),
    }
    model.train(was_training)
    return result


@torch.no_grad()
def routing_diagnostics(model: BehaviorMoETiM4Rec, batches: list[Any]) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    expert_names = tuple(model.expert_names)
    expert_count = len(expert_names)
    log_experts = math.log(expert_count)
    device = next(model.parameters()).device
    prob_sums = {task: torch.zeros(expert_count, device=device) for task in ROUTING_TASKS}
    local_prob_sums = {
        task: torch.zeros(len(model.allowed_expert_indices[task]), device=device)
        for task in ROUTING_TASKS
    }
    entropy_sums = {task: 0.0 for task in ROUTING_TASKS}
    local_entropy_sums = {task: 0.0 for task in ROUTING_TASKS}
    local_entropy_denominators = {
        task: math.log(max(len(model.allowed_expert_indices[task]), 2))
        for task in ROUTING_TASKS
    }
    forbidden_max_abs = {task: 0.0 for task in ROUTING_TASKS}
    local_sum_deviation_max = {task: 0.0 for task in ROUTING_TASKS}
    logits_stats = {
        task: {"sum": 0.0, "sum_sq": 0.0, "count": 0, "min": float("inf"), "max": -float("inf")}
        for task in ROUTING_TASKS
    }
    rows = 0
    for batch in batches:
        seq_output = model.shared_representation(batch)
        weights = model.routing_weights_from_shared(seq_output)
        local_weights = model.local_routing_weights_from_shared(seq_output)
        raw_logits = model.routing_logits_from_shared(seq_output, masked=False)
        for task, task_weights in weights.items():
            prob_sums[task] += task_weights.sum(dim=0)
            entropy = -(task_weights.clamp_min(1e-12) * task_weights.clamp_min(1e-12).log()).sum(dim=-1)
            entropy_sums[task] += float(entropy.sum().cpu().item())
            local_task_weights = local_weights[task]
            local_prob_sums[task] += local_task_weights.sum(dim=0)
            local_entropy = -(
                local_task_weights.clamp_min(1e-12)
                * local_task_weights.clamp_min(1e-12).log()
            ).sum(dim=-1)
            local_entropy_sums[task] += float(local_entropy.sum().cpu().item())
            local_sum_deviation = torch.abs(local_task_weights.sum(dim=-1) - 1.0).max()
            local_sum_deviation_max[task] = max(
                local_sum_deviation_max[task],
                float(local_sum_deviation.cpu().item()),
            )
            allowed = set(model.allowed_expert_indices[task])
            forbidden = [idx for idx in range(expert_count) if idx not in allowed]
            if forbidden:
                forbidden_abs = task_weights[:, forbidden].abs().max()
                forbidden_max_abs[task] = max(
                    forbidden_max_abs[task],
                    float(forbidden_abs.cpu().item()),
                )
            logits = raw_logits[task]
            logits_float = logits.detach().float()
            logits_stats[task]["sum"] += float(logits_float.sum().cpu().item())
            logits_stats[task]["sum_sq"] += float((logits_float ** 2).sum().cpu().item())
            logits_stats[task]["count"] += int(logits_float.numel())
            logits_stats[task]["min"] = min(logits_stats[task]["min"], float(logits_float.min().cpu().item()))
            logits_stats[task]["max"] = max(logits_stats[task]["max"], float(logits_float.max().cpu().item()))
        rows += len(batch)
    mean_probs = {
        TASK_LABELS[task]: {
            expert: float((prob_sums[task] / rows)[idx].cpu().item())
            for idx, expert in enumerate(expert_names)
        }
        for task in ROUTING_TASKS
    }
    local_mean_probs = {
        TASK_LABELS[task]: {
            expert: float((local_prob_sums[task] / rows)[idx].cpu().item())
            for idx, expert in enumerate(model.allowed_experts[task])
        }
        for task in ROUTING_TASKS
    }
    entropy = {
        TASK_LABELS[task]: {
            "mean_entropy": entropy_sums[task] / rows,
            "normalized_entropy": (entropy_sums[task] / rows) / log_experts,
            "local_mean_entropy": local_entropy_sums[task] / rows,
            "local_normalized_entropy": (local_entropy_sums[task] / rows) / local_entropy_denominators[task],
            "max_expert_share": max(mean_probs[TASK_LABELS[task]].values()),
            "min_expert_share": min(mean_probs[TASK_LABELS[task]].values()),
            "experts_above_5pct": sum(value >= 0.05 for value in mean_probs[TASK_LABELS[task]].values()),
        }
        for task in ROUTING_TASKS
    }
    global_usage = {
        expert: sum(mean_probs[TASK_LABELS[task]][expert] for task in ROUTING_TASKS) / len(ROUTING_TASKS)
        for expert in expert_names
    }
    allowed_masks = {
        TASK_LABELS[task]: list(model.allowed_experts[task])
        for task in ROUTING_TASKS
    }
    forbidden_masks = {
        TASK_LABELS[task]: [
            expert for expert in expert_names if expert not in model.allowed_experts[task]
        ]
        for task in ROUTING_TASKS
    }
    forbidden_checks = {
        TASK_LABELS[task]: {
            "forbidden_experts": forbidden_masks[TASK_LABELS[task]],
            "max_abs_forbidden_weight": forbidden_max_abs[task],
            "exact_zero": forbidden_max_abs[task] == 0.0,
            "local_weight_sum_max_abs_deviation": local_sum_deviation_max[task],
        }
        for task in ROUTING_TASKS
    }
    pairs = []
    task_names = [TASK_LABELS[task] for task in ROUTING_TASKS]
    vectors = {
        name: torch.tensor([mean_probs[name][expert] for expert in expert_names], dtype=torch.float64)
        for name in task_names
    }
    for i, left in enumerate(task_names):
        for right in task_names[i + 1:]:
            diff = vectors[left] - vectors[right]
            denom = torch.linalg.vector_norm(vectors[left]) * torch.linalg.vector_norm(vectors[right])
            cosine_similarity = float(torch.dot(vectors[left], vectors[right]) / denom) if denom > 0 else None
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "l1": float(diff.abs().sum().item()),
                    "l2": float(torch.linalg.vector_norm(diff).item()),
                    "cosine_similarity": cosine_similarity,
                    "cosine_distance": None if cosine_similarity is None else 1.0 - cosine_similarity,
                }
            )
    pairs.sort(key=lambda item: item["l1"], reverse=True)
    logits_summary = {}
    for task, stats in logits_stats.items():
        count = max(int(stats["count"]), 1)
        mean = stats["sum"] / count
        variance = max(stats["sum_sq"] / count - mean * mean, 0.0)
        logits_summary[TASK_LABELS[task]] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": stats["min"],
            "max": stats["max"],
        }
    collapse = {
        "expert_collapse": max(global_usage.values()) > 0.90 or min(global_usage.values()) < 0.01,
        "shared_expert_domination": global_usage["shared"] > 0.70,
        "all_auxiliary_shared_domination": all(
            mean_probs[task]["shared"] > 0.90
            for task in ("click", "long_view", "like", "profile")
        ),
        "all_task_same_routing": (pairs[0]["l1"] if pairs else 0.0) < 1e-3,
        "minimum_experts_used_per_task": min(item["experts_above_5pct"] for item in entropy.values()),
        "all_forbidden_paths_exact_zero": all(item["exact_zero"] for item in forbidden_checks.values()),
    }
    model.train(was_training)
    return {
        "batches": len(batches),
        "rows": rows,
        "routing_mode": model.routing_mode,
        "experts": list(expert_names),
        "allowed_experts": allowed_masks,
        "forbidden_experts": forbidden_masks,
        "mean_probabilities": mean_probs,
        "local_mean_probabilities": local_mean_probs,
        "entropy": entropy,
        "global_expert_utilization": global_usage,
        "task_routing_distances": pairs,
        "strongest_task_difference": pairs[0] if pairs else None,
        "router_logits": logits_summary,
        "forbidden_path_checks": forbidden_checks,
        "collapse_checks": collapse,
    }


def specialist_for_task(task: str, expert_names: tuple[str, ...]) -> str:
    if task == "click":
        return "interest"
    if task == "long_view":
        return "consumption"
    if task in {"like", "profile"}:
        return "engagement" if "engagement" in expert_names else "positive"
    raise KeyError(task)


def specialization_summary_from_routing(
    routing: dict[str, Any],
    *,
    routing_mode: str,
) -> dict[str, Any]:
    mean_probs = routing["mean_probabilities"]
    expert_names = tuple(routing["experts"])
    aux_tasks = ("click", "long_view", "like", "profile")
    shares = {}
    short_names = {
        "click": "click",
        "long_view": "long",
        "like": "like",
        "profile": "profile",
    }
    for task in aux_tasks:
        specialist = specialist_for_task(task, expert_names)
        shares[f"{short_names[task]}_{specialist}_share"] = mean_probs[task][specialist]
    shared_shares = {task: mean_probs[task]["shared"] for task in aux_tasks}
    average_share = mean(list(shares.values()))
    if routing_mode == "structured":
        uniform_baseline = mean([1.0 / len(routing["allowed_experts"][task]) for task in aux_tasks])
    else:
        uniform_baseline = 1.0 / len(expert_names)
    return {
        "task_specialist_shares": shares,
        "shared_shares": shared_shares,
        "average_specialist_share": average_share,
        "uniform_specialist_share_baseline": uniform_baseline,
        "specialization_above_uniform": average_share - uniform_baseline,
        "shared_domination": all(value > 0.90 for value in shared_shares.values()),
        "ranking_expert_distribution": mean_probs["ranking"],
    }


def generic_reference_summary(reference_path: Path) -> dict[str, Any]:
    if not reference_path.exists():
        return {"available": False, "path": relative_path(reference_path)}
    payload = load_json(reference_path)
    routing = payload.get("routing_diagnostics_final", {})
    specialization = (
        specialization_summary_from_routing(routing, routing_mode="generic")
        if routing
        else None
    )
    step_times = payload.get("behavior_moe_smoke", {}).get("step_times_sec", [])
    cost = payload.get("cost_summary", {}).get("behavior_moe", {})
    return {
        "available": True,
        "path": relative_path(reference_path),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "slurm": payload.get("slurm"),
        "gpu": payload.get("gpu"),
        "routing_mode": "generic",
        "routing_diagnostics_final": routing,
        "specialization_summary": specialization,
        "parameter_counts": payload.get("parameter_counts", {}),
        "step_time": {
            "mean_step_time_sec": cost.get("mean_step_time_sec"),
            "trimmed_mean_step_time_sec": trimmed_mean([float(value) for value in step_times]) if step_times else None,
            "step_times_sec": step_times,
            "max_allocated_bytes": cost.get("max_allocated_bytes"),
            "max_reserved_bytes": cost.get("max_reserved_bytes"),
        },
    }


def run_behavior_moe_steps(
    model: BehaviorMoETiM4Rec,
    optimizer: torch.optim.Optimizer,
    batches: list[Any],
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    load_balance_weight: float,
) -> dict[str, Any]:
    warmup_behavior_moe(model, optimizer, batches[0], sampled, pos_weights, load_balance_weight)
    reset_cuda_peak()
    model.train()
    before = snapshot_parameters(model)
    step_times = []
    losses_history = []
    all_gradients_finite = True
    last_gradient_summary: dict[str, Any] = {}
    last_head_summary: dict[str, Any] = {}
    start_total = time.monotonic()
    for step, batch in enumerate(batches, start=1):
        sync_cuda()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = model.calculate_multitask_loss(
            batch,
            lambda_aux=float(sampled["lambda_aux"]),
            pos_weights=pos_weights,
            task_weights=sampled["normalized_task_weights"],
            load_balance_weight=load_balance_weight,
        )
        loss_values = float_losses(losses)
        if not all(math.isfinite(value) for value in loss_values.values()):
            raise RuntimeError(f"Non-finite Behavior-MoE losses at step {step}: {loss_values}")
        losses["total"].backward()
        grad_check = all_gradient_check(model)
        last_gradient_summary = gradient_group_summary(model)
        last_head_summary = auxiliary_head_gradient_summary(model)
        all_gradients_finite = (
            all_gradients_finite
            and bool(grad_check["all_finite"])
            and bool(last_gradient_summary["all_finite"])
        )
        if not all_gradients_finite:
            raise RuntimeError(f"Non-finite Behavior-MoE gradients at step {step}: {grad_check}")
        optimizer.step()
        sync_cuda()
        step_times.append(time.monotonic() - start)
        losses_history.append({"step": step, **loss_values})
    updates = update_summary(model, before)
    return {
        "status": "completed",
        "steps": len(batches),
        "warmup_forward_backward_not_counted": True,
        "mean_step_time_sec": mean(step_times),
        "trimmed_mean_step_time_sec": trimmed_mean(step_times),
        "step_times_sec": step_times,
        "total_time_sec": time.monotonic() - start_total,
        "losses_history": losses_history,
        "last_losses": losses_history[-1],
        "all_losses_finite": all(math.isfinite(value) for row in losses_history for key, value in row.items() if key != "step"),
        "all_gradients_finite": all_gradients_finite,
        "last_gradient_summary": last_gradient_summary,
        "auxiliary_head_gradient_summary": last_head_summary,
        "parameter_updates": updates,
        "cuda_memory": max_cuda_memory(),
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
    result = {
        "tim4rec": base,
        "tuned_multitask": multitask,
        "behavior_moe": behavior,
        "behavior_moe_groups": groups,
        "delta_vs_tuned_multitask": int(behavior["total"] - multitask["total"]),
        "relative_increase_vs_tuned_multitask_pct": (behavior["total"] - multitask["total"]) / multitask["total"] * 100.0,
    }
    if behavior_model.routing_mode == "structured":
        result["structured_behavior_moe"] = behavior
        result["structured_behavior_moe_groups"] = groups
    return result


def source_hashes() -> dict[str, str]:
    files = [
        "experiments/behavior_moe_tim4rec/model.py",
        "experiments/behavior_moe_tim4rec/smoke_test.py",
        "experiments/behavior_moe_tim4rec/config.yaml",
        "experiments/build_results.py",
        "experiments/multitask_tim4rec/model.py",
        "experiments/multitask_tim4rec_optuna/best_params.yaml",
        "slurm/behavior_moe_tim4rec.sh",
    ]
    return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in files if (ROOT / rel).exists()}


def write_notes(path: Path, result: dict[str, Any]) -> None:
    final_routing = result["routing_diagnostics_final"]["mean_probabilities"]
    expert_names = result["architecture"]["experts"]
    rows = [
        "| task | " + " | ".join(expert_names) + " |",
        "|---|" + "|".join("---:" for _ in expert_names) + "|",
    ]
    for task in TASK_DISPLAY_ORDER:
        weights = final_routing[task]
        rows.append("| " + task + " | " + " | ".join(f"{weights[expert]:.4f}" for expert in expert_names) + " |")

    allowed_rows = [
        "| task | allowed experts | forbidden experts | exact zeros |",
        "|---|---|---|---:|",
    ]
    for task in TASK_DISPLAY_ORDER:
        allowed = ", ".join(result["routing_diagnostics_final"]["allowed_experts"][task])
        forbidden = ", ".join(result["routing_diagnostics_final"]["forbidden_experts"][task]) or "none"
        exact_zero = result["routing_diagnostics_final"]["forbidden_path_checks"][task]["exact_zero"]
        allowed_rows.append(f"| {task} | {allowed} | {forbidden} | {exact_zero} |")

    specialist = result["specialization_summary"]
    specialist_rows = [
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in specialist["task_specialist_shares"].items():
        specialist_rows.append(f"| {key} | {value:.4f} |")
    specialist_rows.extend(
        [
            f"| average_specialist_share | {specialist['average_specialist_share']:.4f} |",
            f"| uniform_specialist_share_baseline | {specialist['uniform_specialist_share_baseline']:.4f} |",
            f"| specialization_above_uniform | {specialist['specialization_above_uniform']:.4f} |",
        ]
    )

    collapse = result["routing_diagnostics_final"]["collapse_checks"]
    strongest = result["routing_diagnostics_final"]["strongest_task_difference"]
    updates = result["behavior_moe_smoke"]["parameter_updates"]
    title = (
        "# Structured Behavior-MoE smoke 001"
        if result["routing_mode"] == "structured"
        else "# Behavior-MoE smoke 001"
    )
    generic_ref = result.get("generic_moe_reference", {})
    generic_cost = generic_ref.get("step_time", {}) if generic_ref.get("available") else {}
    lines = [
        title,
        "",
        "## Цель",
        "",
        f"Проверить техническую работоспособность `{result['model']}` поверх tuned MultitaskTiM4Rec на реальных train batches KuaiRand Protocol B.",
        "",
        "## Архитектура",
        "",
        f"- Routing mode: `{result['routing_mode']}`.",
        f"- Experts: `{', '.join(result['architecture']['experts'])}`.",
        f"- Expert MLP: `{result['architecture']['expert_mlp']}`.",
        f"- Router: `{result['architecture']['router']}`.",
        f"- Residual: `{result['architecture']['residual']}`.",
        f"- Load balance на smoke: `{result['architecture']['load_balance']['enabled']}`.",
        "",
        "## Allowed expert masks",
        "",
        *allowed_rows,
        "",
        "## Smoke",
        "",
        f"- Batches: `{result['smoke']['batches']}`.",
        f"- Batch size: `{result['smoke']['batch_size']}`.",
        f"- Losses finite: `{result['behavior_moe_smoke']['all_losses_finite']}`.",
        f"- Gradients finite: `{result['behavior_moe_smoke']['all_gradients_finite']}`.",
        f"- Test evaluations: `{result['test_safety']['test_evaluation_count']}`.",
        "",
        "## Routing после smoke",
        "",
        *rows,
        "",
        "## Specialization score",
        "",
        *specialist_rows,
        f"- Shared domination: `{specialist['shared_domination']}`.",
        "",
        "## Collapse checks",
        "",
        f"- Expert collapse: `{collapse['expert_collapse']}`.",
        f"- Shared domination: `{collapse['shared_expert_domination']}`.",
        f"- All auxiliary shared domination: `{collapse['all_auxiliary_shared_domination']}`.",
        f"- Forbidden paths exact zero: `{collapse['all_forbidden_paths_exact_zero']}`.",
        f"- All-task same routing: `{collapse['all_task_same_routing']}`.",
        f"- Minimum experts used per task: `{collapse['minimum_experts_used_per_task']}`.",
        "",
        "## Specialization signal",
        "",
        f"- Strongest pair: `{strongest['left']}` vs `{strongest['right']}`.",
        f"- L1 distance: `{strongest['l1']:.6f}`.",
        f"- Cosine distance: `{strongest['cosine_distance']:.8f}`.",
        "",
        "## Gradients and updates",
        "",
        f"- Experts updated: `{updates['experts']['all_tensors_updated']}`.",
        f"- Router updated: `{updates['router']['all_tensors_updated']}`.",
        f"- Auxiliary heads updated: `{updates['auxiliary_heads']['all_tensors_updated']}`.",
        f"- Expected structured connectivity met: `{result['gradient_connectivity']['expected_connectivity_met']}`.",
        "",
        "## Cost",
        "",
        f"- Tuned fixed mean step: `{result['cost_summary']['tuned_fixed_reference']['mean_step_time_sec']:.6f}` sec.",
        f"- Tuned fixed trimmed mean step: `{result['cost_summary']['tuned_fixed_reference']['trimmed_mean_step_time_sec']:.6f}` sec.",
        f"- Behavior-MoE mean step: `{result['cost_summary']['behavior_moe']['mean_step_time_sec']:.6f}` sec.",
        f"- Behavior-MoE trimmed mean step: `{result['cost_summary']['behavior_moe']['trimmed_mean_step_time_sec']:.6f}` sec.",
        f"- Step-time overhead: `{result['cost_summary']['overhead_vs_tuned_fixed_step_time']:.4f}`.",
        f"- Peak VRAM: `{result['cost_summary']['behavior_moe']['max_allocated_bytes']}` bytes.",
    ]
    if generic_cost:
        lines.extend(
            [
                f"- Generic reference mean step: `{generic_cost.get('mean_step_time_sec')}` sec.",
                f"- Generic reference trimmed mean step: `{generic_cost.get('trimmed_mean_step_time_sec')}` sec.",
                f"- Generic reference GPU: `{generic_ref.get('gpu', {}).get('name')}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            result["decision"]["summary"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_started = datetime.now(timezone.utc)
    config_path = Path(args.config)
    behavior_config = load_yaml(config_path)
    variant = resolve_variant(args, behavior_config)
    enforce_artifact_policy(variant["output"], variant["notes"], allow_overwrite=args.allow_overwrite)
    optuna_config = load_yaml(project_path(behavior_config["base"]["optuna_config"]))
    best_params = load_yaml(project_path(behavior_config["base"]["best_params"]))
    summary = load_json(Path(optuna_config["validation_only_data"]["summary_json"]))
    assert_validation_only_summary(summary)
    target_stats = load_target_stats(project_path(optuna_config["source"]["target_statistics"]))
    sampled = sampled_from_locked_params(best_params, target_stats)
    artifact_dir = variant["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config = build_behavior_config(
        optuna_config,
        artifact_dir / "recbole",
        sampled,
        variant["moe_config"],
        variant["model_cls"],
    )
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Behavior-MoE smoke.")
    if tuple(config["multitask_targets"]) != TARGETS:
        raise RuntimeError(f"Task set changed: {config['multitask_targets']}")
    if not bool(config["is_time"]):
        raise RuntimeError("TiM4Rec is_time must remain True.")

    _dataset, train_dataset, train_data = validation_only_train_dataset(config, summary)
    device = config["device"]
    batches = collect_batches(train_data, device, int(args.batches or variant["smoke_config"]["batches"]))
    pos_weights = pos_weight_tensors(sampled["effective_pos_weights"], device)
    checkpoint_path = Path(behavior_config["base"]["tuned_checkpoint"])

    baseline_model, baseline_optimizer, baseline_checkpoint = new_baseline_model(config, train_dataset, checkpoint_path, sampled)
    tuned_fixed_smoke = run_fixed_tuned_steps(baseline_model, baseline_optimizer, batches, sampled, pos_weights)

    moe_model, moe_optimizer, moe_checkpoint = new_moe_model(
        config,
        train_dataset,
        checkpoint_path,
        sampled,
        variant["model_cls"],
    )
    param_counts = parameter_count_summary(config, train_dataset, moe_model)
    initial_probe = forward_probe(moe_model, batches[0])
    routing_initial = routing_diagnostics(moe_model, batches)
    lb_weight = float(variant["moe_config"].get("load_balance_weight", 0.0))
    moe_smoke = run_behavior_moe_steps(moe_model, moe_optimizer, batches, sampled, pos_weights, lb_weight)
    final_probe = forward_probe(moe_model, batches[0])
    routing_final = routing_diagnostics(moe_model, batches)
    specialization_initial = specialization_summary_from_routing(routing_initial, routing_mode=variant["routing_mode"])
    specialization_final = specialization_summary_from_routing(routing_final, routing_mode=variant["routing_mode"])
    gradient_connectivity = gradient_connectivity_probe(moe_model, batches[-1], sampled, pos_weights)
    generic_ref = generic_reference_summary(GENERIC_REFERENCE_PATH)

    overhead = moe_smoke["mean_step_time_sec"] / tuned_fixed_smoke["mean_step_time_sec"]
    trimmed_overhead = moe_smoke["trimmed_mean_step_time_sec"] / tuned_fixed_smoke["trimmed_mean_step_time_sec"]
    collapse = routing_final["collapse_checks"]
    gradients = moe_smoke["last_gradient_summary"]
    heads = moe_smoke["auxiliary_head_gradient_summary"]
    all_experts_grad = gradients["experts"]["all_parameters_have_gradient"] and gradients["experts"]["gradient_norm"] > 0.0
    router_grad = gradients["router"]["all_parameters_have_gradient"] and gradients["router"]["gradient_norm"] > 0.0
    all_heads_grad = all(item["receives_gradient"] for item in heads.values())
    task_signal = not collapse["all_task_same_routing"]
    structured_checks = (
        variant["routing_mode"] != "structured"
        or (
            collapse["all_forbidden_paths_exact_zero"]
            and gradient_connectivity["expected_connectivity_met"]
            and not collapse["all_auxiliary_shared_domination"]
            and specialization_final["specialization_above_uniform"] > 0.0
        )
    )
    ready_for_sanity = (
        final_probe["all_outputs_finite"]
        and moe_smoke["all_losses_finite"]
        and moe_smoke["all_gradients_finite"]
        and all_experts_grad
        and router_grad
        and all_heads_grad
        and not collapse["expert_collapse"]
        and param_counts["relative_increase_vs_tuned_multitask_pct"] < 10.0
        and structured_checks
    )
    next_run = (
        "5-epoch StructuredBehaviorMoE validation sanity"
        if variant["routing_mode"] == "structured" and ready_for_sanity
        else "analyze structured routing before 5-epoch sanity"
        if variant["routing_mode"] == "structured"
        else "plain Behavior-MoE"
        if not collapse["expert_collapse"]
        else "Behavior-MoE + minimal load balancing"
    )
    expert_semantics = {
        "interest": "is_click",
        "consumption": "long_view",
        "shared": "residual/general",
    }
    if variant["routing_mode"] == "structured":
        expert_semantics["engagement"] = "is_like + is_profile_enter"
    else:
        expert_semantics["positive"] = "is_like + is_profile_enter"
    model_name = "StructuredBehaviorMoE" if variant["routing_mode"] == "structured" else "BehaviorMoETiM4Rec"
    decision_summary = (
        "Structured smoke технически корректен: forward/backward/optimizer step проходят на real train batches, "
        "forbidden expert paths имеют exact zero, expected task-to-expert gradient connectivity подтверждена, "
        "load balancing выключен, test не использован. Это architecture probe; следующий шаг может быть 5-epoch validation sanity, "
        "но для диплома нужно честно сопоставить эту схему с PLE-style baseline."
        if variant["routing_mode"] == "structured" and ready_for_sanity
        else "Structured smoke выявил риск до 5-epoch sanity."
        if variant["routing_mode"] == "structured"
        else (
            "Smoke pipeline корректен: Behavior-MoE делает forward/backward/optimizer step на real train batches, "
            "router и experts получают gradients, routing не collapsed. Следующий sanity лучше запускать как plain Behavior-MoE без load balancing."
            if ready_for_sanity
            else "Smoke выявил риск, который нужно исправить до 5-epoch sanity."
        )
    )
    result: dict[str, Any] = {
        "run_id": variant["run_id"],
        "status": "diagnostic",
        "execution_status": "completed",
        "record_type": "sanity",
        "routing_mode": variant["routing_mode"],
        "actual_epochs": 0,
        "test_evaluation_count": 0,
        "created_at_utc": run_started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": git_value(["config", "--get", "remote.origin.url"]),
        },
        "environment": environment_info(),
        "slurm": slurm_info(),
        "gpu": gpu_info(),
        "resource": {"maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "artifact_dir": str(artifact_dir),
        "base_model": "MultitaskTiM4Rec",
        "base_run": "multitask_tim4rec_tuned_001",
        "model": model_name,
        "generic_moe_reference": generic_ref,
        "dataset": {
            "name": "KuaiRand",
            "protocol": "B",
            "split_used_for_smoke": "train",
            "fingerprint_expected": EXPECTED_FINGERPRINT,
            "identity_hash_expected": EXPECTED_IDENTITY_HASH,
            "validation_only_summary": summary,
            "test_rows_in_validation_only_benchmark": int(summary["rows"]["test"]),
        },
        "test_safety": {
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
            "test_evaluated": False,
            "test_evaluation_count": 0,
        },
        "smoke": {
            "batches": len(batches),
            "batch_size": len(batches[0]),
            "train_batches_available": len(train_data),
            "optimization_steps_only": True,
            "epochs_run": 0,
            "full_validation_run": False,
            "full_training_run": False,
        },
        "architecture": {
            "routing_mode": variant["routing_mode"],
            "experts": list(moe_model.expert_names),
            "expert_semantics": expert_semantics,
            "allowed_expert_masks": {
                TASK_LABELS[task]: list(experts)
                for task, experts in moe_model.allowed_experts.items()
            },
            "expert_mlp": "Linear(hidden, hidden) -> GELU -> Dropout -> Linear(hidden, hidden)",
            "router": (
                "separate learned Linear(hidden, num_experts) router head per task; "
                "structured mode masks forbidden logits before softmax"
            ),
            "task_conditioned_routing": {
                "tasks": [TASK_LABELS[task] for task in ROUTING_TASKS],
                "implementation": "task-specific router heads over shared h",
                "current_behavior_labels_used_as_router_input": False,
                "explicit_task_embeddings": False,
                "task_context": "separate router head per task",
            },
            "residual": "h_task = h + residual_scale * sum_e p(task,e|h) * expert_e(h)",
            "routing_formula": "allowed logits -> softmax over allowed set -> expanded 4-expert weights with exact zero on forbidden paths",
            "behavior_moe": variant["moe_config"],
            "structured_allowed_experts_reference": {
                TASK_LABELS[task]: list(experts)
                for task, experts in STRUCTURED_ALLOWED_EXPERTS.items()
            },
            "load_balance": {
                **behavior_config["architecture"]["load_balance"],
                "enabled": False,
                "coefficient": lb_weight,
            },
        },
        "loss_config": {
            "source": "multitask_tim4rec_optuna_v1 trial 110 / best_params.yaml",
            "lambda_aux": sampled["lambda_aux"],
            "normalized_task_weights": sampled["normalized_task_weights"],
            "effective_pos_weights": sampled["effective_pos_weights"],
            "learning_rate": sampled["learning_rate"],
            "weight_decay": sampled["weight_decay"],
            "head_lr_multiplier": sampled["head_lr_multiplier"],
            "head_learning_rate": sampled["head_learning_rate"],
            "load_balance_weight": lb_weight,
        },
        "checkpoint": {
            "tuned_fixed_reference": baseline_checkpoint,
            "behavior_moe_initialized_from_tuned_fixed": moe_checkpoint,
        },
        "parameter_counts": param_counts,
        "forward_probe_initial": initial_probe,
        "forward_probe_final": final_probe,
        "routing_diagnostics_initial": routing_initial,
        "routing_diagnostics_final": routing_final,
        "expert_utilization": routing_final["global_expert_utilization"],
        "task_routing_distances": routing_final["task_routing_distances"],
        "specialization_summary_initial": specialization_initial,
        "specialization_summary": specialization_final,
        "ranking_routing": specialization_final["ranking_expert_distribution"],
        "gradient_diagnostics": {
            "all_experts_receive_gradients": all_experts_grad,
            "router_receives_gradients": router_grad,
            "all_behavior_heads_receive_gradients": all_heads_grad,
            "group_summary_last_step": gradients,
            "head_summary_last_step": heads,
        },
        "gradient_connectivity": gradient_connectivity,
        "tuned_fixed_reference_smoke": tuned_fixed_smoke,
        "behavior_moe_smoke": moe_smoke,
        "cost_summary": {
            "tuned_fixed_reference": {
                "mean_step_time_sec": tuned_fixed_smoke["mean_step_time_sec"],
                "trimmed_mean_step_time_sec": tuned_fixed_smoke["trimmed_mean_step_time_sec"],
                "max_allocated_bytes": tuned_fixed_smoke["cuda_memory"]["max_allocated_bytes"],
                "max_reserved_bytes": tuned_fixed_smoke["cuda_memory"]["max_reserved_bytes"],
            },
            "behavior_moe": {
                "mean_step_time_sec": moe_smoke["mean_step_time_sec"],
                "trimmed_mean_step_time_sec": moe_smoke["trimmed_mean_step_time_sec"],
                "max_allocated_bytes": moe_smoke["cuda_memory"]["max_allocated_bytes"],
                "max_reserved_bytes": moe_smoke["cuda_memory"]["max_reserved_bytes"],
            },
            "overhead_vs_tuned_fixed_step_time": overhead,
            "trimmed_overhead_vs_tuned_fixed_step_time": trimmed_overhead,
            "generic_reference_comparable_hardware": (
                generic_ref.get("gpu", {}).get("name") == gpu_info().get("name")
                if generic_ref.get("available")
                else None
            ),
            "generic_reference_step_time": generic_ref.get("step_time", {}),
        },
        "risk_checks": {
            "router_collapse": collapse["expert_collapse"],
            "all_task_same_routing": collapse["all_task_same_routing"],
            "shared_expert_domination": collapse["shared_expert_domination"],
            "all_auxiliary_shared_domination": collapse["all_auxiliary_shared_domination"],
            "forbidden_paths_not_exact_zero": not collapse["all_forbidden_paths_exact_zero"],
            "specialization_not_above_uniform": specialization_final["specialization_above_uniform"] <= 0.0,
            "too_large_model_overhead": param_counts["relative_increase_vs_tuned_multitask_pct"] >= 10.0,
            "gradients_not_reaching_some_experts": not all_experts_grad,
            "router_gradient_missing": not router_grad,
            "expected_gradient_connectivity_missing": not gradient_connectivity["expected_connectivity_met"],
            "unstable_routing_logits": any(abs(stats["max"]) > 20.0 or abs(stats["min"]) > 20.0 for stats in routing_final["router_logits"].values()),
            "moe_drowning_ranking_representation": float(variant["moe_config"]["residual_scale"]) > 0.25,
        },
        "decision": {
            "pipeline_correct": bool(final_probe["all_outputs_finite"] and moe_smoke["all_losses_finite"] and moe_smoke["all_gradients_finite"]),
            "router_not_collapsed": not collapse["expert_collapse"],
            "forbidden_paths_exact_zero": collapse["all_forbidden_paths_exact_zero"],
            "specialists_used_not_all_shared": not collapse["all_auxiliary_shared_domination"],
            "specialization_score_above_uniform": specialization_final["specialization_above_uniform"] > 0.0,
            "expected_gradient_connectivity_met": gradient_connectivity["expected_connectivity_met"],
            "task_specific_routing_signal": task_signal,
            "moe_overhead_acceptable": param_counts["relative_increase_vs_tuned_multitask_pct"] < 10.0,
            "ready_for_5_epoch_sanity": ready_for_sanity,
            "recommended_next_sanity_run": next_run,
            "summary": decision_summary,
        },
        "source_files": source_hashes(),
    }
    save_json(variant["output"], result)
    write_notes(variant["notes"], result)


if __name__ == "__main__":
    main()
