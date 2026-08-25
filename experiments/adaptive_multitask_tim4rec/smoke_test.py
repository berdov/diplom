#!/usr/bin/env python
"""Smoke tests for adaptive multitask optimization on real KuaiRand train batches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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
from recbole.utils import init_seed


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from experiments.adaptive_multitask_tim4rec.methods.common import (  # noqa: E402
    AUX_TARGETS,
    TASK_ORDER,
    assign_flat_gradient,
    assign_gradient_tensors,
    conflict_summary,
    cosine_matrix,
    ensure_finite_gradients,
    finite_named_scalars,
    gradient_norms,
    max_cuda_memory,
    parameter_group_summary,
    shared_parameter_entries,
    task_gradient_vectors,
    tensor_to_float,
)
from experiments.adaptive_multitask_tim4rec.methods.gradnorm import GradNormAuxiliaryBalancer  # noqa: E402
from experiments.adaptive_multitask_tim4rec.methods.metabalance import MetaBalanceAuxiliaryBalancer  # noqa: E402
from experiments.adaptive_multitask_tim4rec.methods.pcgrad import PCGradProjector  # noqa: E402
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec  # noqa: E402
from experiments.multitask_tim4rec.train import (  # noqa: E402
    EXPECTED_FINGERPRINT,
    EXPECTED_IDENTITY_HASH,
    all_gradient_check,
    count_parameters,
    load_target_stats,
)
from experiments.multitask_tim4rec_optuna.optuna_search import (  # noqa: E402
    build_config,
    compute_tuned_losses,
    create_loaders,
    load_data_bundle,
    load_yaml,
    optimizer_for_trial,
    pos_weight_tensors,
    project_path,
)
from experiments.multitask_tim4rec_optuna.run_locked_tuned import sampled_from_locked_params  # noqa: E402


RUN_ID = "adaptive_smoke_001"
DEFAULT_OUTPUT = ROOT / "experiments" / "adaptive_multitask_tim4rec" / "runs" / f"{RUN_ID}.json"
DEFAULT_NOTES = ROOT / "experiments" / "adaptive_multitask_tim4rec" / "runs" / f"{RUN_ID}_notes.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments" / "adaptive_multitask_tim4rec" / "config.yaml"))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--notes", default=str(DEFAULT_NOTES))
    parser.add_argument("--artifact-dir", default="/home/daryumin/iberdov/diplom/experiments/adaptive_multitask_tim4rec/adaptive_smoke_001")
    parser.add_argument("--batches", type=int, default=None)
    return parser.parse_args()


def git_value(args: list[str], default: str = "unknown") -> str:
    env_fallbacks = {
        ("rev-parse", "HEAD"): "ADAPTIVE_MTL_GIT_COMMIT",
        ("rev-parse", "--abbrev-ref", "HEAD"): "ADAPTIVE_MTL_GIT_BRANCH",
        ("config", "--get", "remote.origin.url"): "ADAPTIVE_MTL_GIT_REMOTE",
    }
    try:
        value = subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        if value:
            return value
    except Exception:
        pass
    env_key = env_fallbacks.get(tuple(args))
    if env_key:
        return os.environ.get(env_key) or default
    return default


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


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
        "constraint": os.environ.get("SLURM_JOB_CONSTRAINT"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    }


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def elapsed_sec(start: float) -> float:
    sync_cuda()
    return time.monotonic() - start


def strip_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return tensor_to_float(value)
        return {"shape": list(value.shape), "norm": float(torch.linalg.vector_norm(value.detach().float()).cpu().item())}
    if isinstance(value, dict):
        return {key: strip_tensors(item) for key, item in value.items() if key not in {"vectors", "combined_gradient", "combined_gradients", "vectors_before", "vectors_after"}}
    if isinstance(value, list):
        return [strip_tensors(item) for item in value]
    return value


def load_checkpoint(model: MultitaskTiM4Rec, path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tuned checkpoint is required for smoke diagnostics: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint has no state_dict: {path}")
    model.load_state_dict(state_dict, strict=True)
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "best_valid_score": checkpoint.get("best_valid_score"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def new_model_and_optimizer(config: Any, train_dataset: Any, checkpoint_path: Path, sampled: dict[str, Any]) -> tuple[MultitaskTiM4Rec, torch.optim.Optimizer, dict[str, Any]]:
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    device = config["device"]
    model = MultitaskTiM4Rec(config, train_dataset).to(device)
    checkpoint_info = load_checkpoint(model, checkpoint_path, device)
    model.train()
    optimizer = optimizer_for_trial(model, sampled)
    return model, optimizer, checkpoint_info


def collect_batches(train_data: Any, device: torch.device, count: int) -> list[Any]:
    batches = []
    for interaction in train_data:
        batches.append(interaction.to(device))
        if len(batches) >= count:
            break
    if len(batches) != count:
        raise RuntimeError(f"Expected {count} smoke batches, got {len(batches)}")
    return batches


def split_tuned_losses(losses: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    raw_aux = {target: losses[f"{target}_loss"] for target in AUX_TARGETS}
    task_contributions = {"rank": losses["rank"]}
    for target in AUX_TARGETS:
        task_contributions[target] = losses[f"{target}_scaled_contribution"]
    return raw_aux, task_contributions


def fixed_total_from_raw(rank_loss: torch.Tensor, aux_losses: dict[str, torch.Tensor], sampled: dict[str, Any]) -> torch.Tensor:
    total = rank_loss
    for target in AUX_TARGETS:
        total = total + float(sampled["lambda_aux"]) * float(sampled["normalized_task_weights"][target]) * aux_losses[target]
    return total


def first_batch_tuned_diagnostics(
    model: MultitaskTiM4Rec,
    interaction: Any,
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    shared_entries: list[Any],
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    losses = compute_tuned_losses(model, interaction, sampled, pos_weights)
    _raw_aux, task_contributions = split_tuned_losses(losses)
    vectors = task_gradient_vectors(task_contributions, shared_entries, TASK_ORDER)
    matrix = cosine_matrix(vectors, TASK_ORDER)
    norms = gradient_norms(vectors, TASK_ORDER)
    per_task = []
    for task in TASK_ORDER:
        raw_loss = losses["rank"] if task == "rank" else losses[f"{task}_loss"]
        weighted_loss = losses["rank"] if task == "rank" else losses[f"{task}_scaled_contribution"]
        per_task.append(
            {
                "task": task,
                "raw_loss": tensor_to_float(raw_loss),
                "weighted_loss": tensor_to_float(weighted_loss),
                "shared_gradient_norm": norms[task],
                "cosine_with_ranking": 1.0 if task == "rank" else matrix["rank"][task],
            }
        )
    model.zero_grad(set_to_none=True)
    return {
        "task_table": per_task,
        "cosine_matrix": matrix,
        "gradient_norms": norms,
        "conflicts": conflict_summary(matrix, TASK_ORDER),
        "losses": {key: tensor_to_float(value) for key, value in losses.items()},
        "finite_losses": finite_named_scalars(losses),
    }


def run_fixed_steps(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    batches: list[Any],
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
) -> dict[str, Any]:
    reset_cuda_peak()
    step_times = []
    last_losses: dict[str, float] = {}
    all_finite = True
    start_total = time.monotonic()
    for batch in batches:
        sync_cuda()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = compute_tuned_losses(model, batch, sampled, pos_weights)
        losses["total"].backward()
        finite = all_gradient_check(model)
        all_finite = all_finite and bool(finite["all_finite"])
        optimizer.step()
        step_times.append(elapsed_sec(start))
        last_losses = {key: tensor_to_float(value) for key, value in losses.items()}
    return {
        "status": "completed",
        "steps": len(batches),
        "step_times_sec": step_times,
        "mean_step_time_sec": sum(step_times) / len(step_times),
        "total_time_sec": elapsed_sec(start_total),
        "autograd_calls_per_step": {"backward": 1, "autograd_grad": 0},
        "all_gradients_finite": all_finite,
        "last_losses": last_losses,
        "cuda_memory": max_cuda_memory(),
    }


def run_gradnorm_steps(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    batches: list[Any],
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    shared_entries: list[Any],
    method_config: dict[str, Any],
) -> dict[str, Any]:
    balancer = GradNormAuxiliaryBalancer(
        sampled["normalized_task_weights"],
        alpha=float(method_config["alpha"]),
        learning_rate=float(method_config["weight_learning_rate"]),
    ).to(next(model.parameters()).device)
    initial_weights = balancer.weights_dict()
    reset_cuda_peak()
    step_times = []
    diagnostics = []
    all_finite = True
    start_total = time.monotonic()
    for batch in batches:
        sync_cuda()
        start = time.monotonic()
        model.zero_grad(set_to_none=True)
        losses = compute_tuned_losses(model, batch, sampled, pos_weights)
        raw_aux, _task_contributions = split_tuned_losses(losses)
        weight_diag = balancer.step_weights(raw_aux, shared_entries)

        optimizer.zero_grad(set_to_none=True)
        refreshed = compute_tuned_losses(model, batch, sampled, pos_weights)
        refreshed_raw_aux, _ = split_tuned_losses(refreshed)
        total = refreshed["rank"] + float(sampled["lambda_aux"]) * balancer.weighted_auxiliary_sum(refreshed_raw_aux)
        total.backward()
        finite = all_gradient_check(model)
        all_finite = all_finite and bool(finite["all_finite"])
        optimizer.step()
        diagnostics.append(strip_tensors(weight_diag | {"model_total_loss": tensor_to_float(total)}))
        step_times.append(elapsed_sec(start))
    return {
        "status": "completed",
        "steps": len(batches),
        "initial_weights": initial_weights,
        "final_weights": balancer.weights_dict(),
        "weights_changed": any(abs(balancer.weights_dict()[target] - initial_weights[target]) > 1e-12 for target in AUX_TARGETS),
        "step_diagnostics": diagnostics,
        "step_times_sec": step_times,
        "mean_step_time_sec": sum(step_times) / len(step_times),
        "total_time_sec": elapsed_sec(start_total),
        "autograd_calls_per_step": {"backward": 2, "autograd_grad": len(AUX_TARGETS)},
        "all_gradients_finite": all_finite,
        "cuda_memory": max_cuda_memory(),
    }


def run_pcgrad_steps(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    batches: list[Any],
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    shared_entries: list[Any],
    method_config: dict[str, Any],
) -> dict[str, Any]:
    projector = PCGradProjector(mode=str(method_config["mode"]), seed=int(method_config["seed"]))
    reset_cuda_peak()
    step_times = []
    diagnostics = []
    all_finite = True
    start_total = time.monotonic()
    for batch in batches:
        sync_cuda()
        start = time.monotonic()
        model.zero_grad(set_to_none=True)
        losses = compute_tuned_losses(model, batch, sampled, pos_weights)
        raw_aux, task_contributions = split_tuned_losses(losses)
        vectors = task_gradient_vectors(task_contributions, shared_entries, TASK_ORDER)
        projection = projector.project(vectors, TASK_ORDER)

        optimizer.zero_grad(set_to_none=True)
        refreshed = compute_tuned_losses(model, batch, sampled, pos_weights)
        refreshed_raw_aux, _ = split_tuned_losses(refreshed)
        total = fixed_total_from_raw(refreshed["rank"], refreshed_raw_aux, sampled)
        total.backward()
        assign_flat_gradient(shared_entries, projection["combined_gradient"])
        finite = all_gradient_check(model)
        shared_finite = ensure_finite_gradients(shared_entries)
        all_finite = all_finite and bool(finite["all_finite"]) and bool(shared_finite["all_finite"])
        optimizer.step()
        diagnostics.append(strip_tensors(projection | {"model_total_loss": tensor_to_float(total), "shared_gradient_finite": shared_finite}))
        step_times.append(elapsed_sec(start))
    return {
        "status": "completed",
        "steps": len(batches),
        "mode": projector.mode,
        "step_diagnostics": diagnostics,
        "step_times_sec": step_times,
        "mean_step_time_sec": sum(step_times) / len(step_times),
        "total_time_sec": elapsed_sec(start_total),
        "autograd_calls_per_step": {"backward": 1, "autograd_grad": len(TASK_ORDER)},
        "all_gradients_finite": all_finite,
        "cuda_memory": max_cuda_memory(),
    }


def run_metabalance_steps(
    model: MultitaskTiM4Rec,
    optimizer: torch.optim.Optimizer,
    batches: list[Any],
    sampled: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    shared_entries: list[Any],
    method_config: dict[str, Any],
) -> dict[str, Any]:
    balancer = MetaBalanceAuxiliaryBalancer(
        relax_factor=float(method_config["relax_factor"]),
        beta=float(method_config["beta"]),
    )
    reset_cuda_peak()
    step_times = []
    diagnostics = []
    all_finite = True
    start_total = time.monotonic()
    for batch in batches:
        sync_cuda()
        start = time.monotonic()
        model.zero_grad(set_to_none=True)
        losses = compute_tuned_losses(model, batch, sampled, pos_weights)
        raw_aux, task_contributions = split_tuned_losses(losses)
        balanced = balancer.balanced_shared_gradients(task_contributions, shared_entries, TASK_ORDER)

        optimizer.zero_grad(set_to_none=True)
        refreshed = compute_tuned_losses(model, batch, sampled, pos_weights)
        refreshed_raw_aux, _ = split_tuned_losses(refreshed)
        total = fixed_total_from_raw(refreshed["rank"], refreshed_raw_aux, sampled)
        total.backward()
        assign_gradient_tensors(shared_entries, balanced["combined_gradients"])
        finite = all_gradient_check(model)
        shared_finite = ensure_finite_gradients(shared_entries)
        all_finite = all_finite and bool(finite["all_finite"]) and bool(shared_finite["all_finite"])
        optimizer.step()
        diagnostics.append(strip_tensors(balanced | {"model_total_loss": tensor_to_float(total), "shared_gradient_finite": shared_finite}))
        step_times.append(elapsed_sec(start))
    return {
        "status": "completed",
        "steps": len(batches),
        "variant": "MetaBalance-Fix",
        "relax_factor": balancer.relax_factor,
        "beta": balancer.beta,
        "step_diagnostics": diagnostics,
        "step_times_sec": step_times,
        "mean_step_time_sec": sum(step_times) / len(step_times),
        "total_time_sec": elapsed_sec(start_total),
        "autograd_calls_per_step": {"backward": 1, "autograd_grad": len(TASK_ORDER)},
        "all_gradients_finite": all_finite,
        "cuda_memory": max_cuda_memory(),
    }


def recommendation_from(result: dict[str, Any]) -> dict[str, Any]:
    baseline = result["fixed_tuned_gradient_diagnostics"]
    conflicts = baseline["conflicts"]
    rank_conflicts = [
        pair for pair in conflicts["negative_pairs_detail"] if pair["left"] == "rank" or pair["right"] == "rank"
    ]
    weights = result["tuned_fixed_configuration"]["normalized_task_weights"]
    task_table = {row["task"]: row for row in baseline["task_table"]}
    like = task_table["is_like"]
    explanation = (
        "is_like has the largest tuned fixed loss weight and remains rare; smoke diagnostics should be interpreted "
        "as gradient-scale evidence, not as NDCG evidence."
    )
    return {
        "next_sanity_methods": ["ranking_anchored_pcgrad", "gradnorm_auxiliary_only"],
        "most_relevant_for_ranking_primary_setup": "ranking_anchored_pcgrad",
        "real_gradient_conflicts_detected": bool(conflicts["negative_pairs"] > 0),
        "rank_auxiliary_conflicts_detected": bool(rank_conflicts),
        "rank_conflicting_pairs": rank_conflicts,
        "high_is_like_weight_analysis": {
            "normalized_weight": weights["is_like"],
            "positive_rate": result["target_statistics"]["is_like"]["positive_rate"],
            "effective_pos_weight": result["tuned_fixed_configuration"]["effective_pos_weights"]["is_like"],
            "raw_loss": like["raw_loss"],
            "weighted_loss": like["weighted_loss"],
            "shared_gradient_norm": like["shared_gradient_norm"],
            "cosine_with_ranking": like["cosine_with_ranking"],
            "interpretation": explanation,
        },
        "possible_own_mechanisms": [
            "behavior-value-aware gradient weighting: start from tuned weights, then cap or boost auxiliary gradients by rank cosine and behavior rarity/value.",
            "ranking-anchored adaptive routing: keep ranking gradient unchanged and route only auxiliary updates through magnitude/conflict gates per shared block.",
        ],
    }


def write_notes(path: Path, result: dict[str, Any]) -> None:
    fixed = result["fixed_tuned_gradient_diagnostics"]
    gradnorm = result["methods"]["gradnorm"]
    pcgrad = result["methods"]["pcgrad"]
    metabalance = result["methods"]["metabalance"]
    rec = result["recommendation"]
    rows = []
    for item in fixed["task_table"]:
        rows.append(
            "| {task} | {raw_loss:.6f} | {weighted_loss:.6f} | {shared_gradient_norm:.6f} | {cosine} |".format(
                task=item["task"],
                raw_loss=item["raw_loss"],
                weighted_loss=item["weighted_loss"],
                shared_gradient_norm=item["shared_gradient_norm"],
                cosine=(
                    ""
                    if item["cosine_with_ranking"] is None
                    else f"{float(item['cosine_with_ranking']):.6f}"
                ),
            )
        )
    lines = [
        f"# {result['run_id']}",
        "",
        "Smoke adaptive multitask optimization на реальных train batches KuaiRand Protocol B.",
        "",
        "## Test safety",
        "",
        f"- `test_evaluation_count`: `{result['test_safety']['test_evaluation_count']}`.",
        f"- Test dataloader created: `{result['test_safety']['test_dataloader_created']}`.",
        "",
        "## Fixed tuned gradient diagnostics",
        "",
        "| task | raw loss | weighted loss | shared grad norm | cosine with rank |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        f"- Negative pairs: `{fixed['conflicts']['negative_pairs']}` / `{fixed['conflicts']['pairs']}`.",
        "",
        "## Adaptive smoke",
        "",
        f"- GradNorm: `{gradnorm['status']}`, weights `{gradnorm['initial_weights']}` -> `{gradnorm['final_weights']}`, mean step `{gradnorm['mean_step_time_sec']:.4f}s`.",
        f"- PCGrad: `{pcgrad['status']}`, mode `{pcgrad['mode']}`, first-step conflicts `{pcgrad['step_diagnostics'][0]['conflicts_before']['negative_pairs']}` -> `{pcgrad['step_diagnostics'][0]['conflicts_after']['negative_pairs']}`, mean step `{pcgrad['mean_step_time_sec']:.4f}s`.",
        f"- MetaBalance: `{metabalance['status']}`, relax `{metabalance['relax_factor']}`, beta `{metabalance['beta']}`, mean step `{metabalance['mean_step_time_sec']:.4f}s`.",
        "",
        "## Recommendation",
        "",
        f"- Next sanity methods: `{', '.join(rec['next_sanity_methods'])}`.",
        f"- Most relevant for ranking-primary setup: `{rec['most_relevant_for_ranking_primary_setup']}`.",
        f"- Real gradient conflicts detected: `{rec['real_gradient_conflicts_detected']}`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_started = datetime.now(timezone.utc)
    adaptive_config = load_yaml(Path(args.config))
    optuna_config = load_yaml(project_path(adaptive_config["base"]["optuna_config"]))
    best_params = load_yaml(project_path(adaptive_config["base"]["best_params"]))
    target_stats = load_target_stats(project_path(optuna_config["source"]["target_statistics"]))
    sampled = sampled_from_locked_params(best_params, target_stats)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data = load_data_bundle(optuna_config, artifact_dir / "data_probe")
    config = build_config(optuna_config, artifact_dir / "recbole", sampled)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for adaptive MultitaskTiM4Rec smoke.")
    if tuple(config["multitask_targets"]) != AUX_TARGETS:
        raise RuntimeError(f"Task set changed: {config['multitask_targets']}")
    if not bool(config["is_time"]):
        raise RuntimeError("TiM4Rec is_time must remain True.")
    if data.validation_only_summary["forbidden_test_paths_loaded"] != []:
        raise RuntimeError(f"Validation-only prep touched test paths: {data.validation_only_summary}")
    if int(data.validation_only_summary["rows"]["test"]) != 0:
        raise RuntimeError(f"Validation-only data unexpectedly has test rows: {data.validation_only_summary['rows']}")

    train_data, _valid_data = create_loaders(config, data.train_dataset, data.valid_dataset)
    device = config["device"]
    batch_count = int(args.batches or adaptive_config["base"]["smoke_batches"])
    batches = collect_batches(train_data, device, batch_count)
    checkpoint_path = Path(adaptive_config["base"]["tuned_checkpoint"])
    pos_weights = pos_weight_tensors(sampled["effective_pos_weights"], device)

    baseline_model, baseline_optimizer, checkpoint_info = new_model_and_optimizer(config, train_data.dataset, checkpoint_path, sampled)
    all_shared = shared_parameter_entries(baseline_model, str(adaptive_config["base"]["diagnostic_shared_selector"]))
    gradnorm_shared = shared_parameter_entries(baseline_model, str(adaptive_config["gradnorm"]["shared_selector"]))
    diagnostics = first_batch_tuned_diagnostics(baseline_model, batches[0], sampled, pos_weights, all_shared)
    fixed_step = run_fixed_steps(baseline_model, baseline_optimizer, batches, sampled, pos_weights)

    gradnorm_model, gradnorm_optimizer, _ = new_model_and_optimizer(config, train_data.dataset, checkpoint_path, sampled)
    gradnorm_entries = shared_parameter_entries(gradnorm_model, str(adaptive_config["gradnorm"]["shared_selector"]))
    gradnorm = run_gradnorm_steps(gradnorm_model, gradnorm_optimizer, batches, sampled, pos_weights, gradnorm_entries, adaptive_config["gradnorm"])

    pcgrad_model, pcgrad_optimizer, _ = new_model_and_optimizer(config, train_data.dataset, checkpoint_path, sampled)
    pcgrad_entries = shared_parameter_entries(pcgrad_model, str(adaptive_config["pcgrad"]["shared_selector"]))
    pcgrad = run_pcgrad_steps(pcgrad_model, pcgrad_optimizer, batches, sampled, pos_weights, pcgrad_entries, adaptive_config["pcgrad"])

    metabalance_model, metabalance_optimizer, _ = new_model_and_optimizer(config, train_data.dataset, checkpoint_path, sampled)
    metabalance_entries = shared_parameter_entries(metabalance_model, str(adaptive_config["metabalance"]["shared_selector"]))
    metabalance = run_metabalance_steps(
        metabalance_model,
        metabalance_optimizer,
        batches,
        sampled,
        pos_weights,
        metabalance_entries,
        adaptive_config["metabalance"],
    )

    result: dict[str, Any] = {
        "run_id": args.run_id,
        "status": "completed",
        "created_at_utc": run_started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": git_value(["config", "--get", "remote.origin.url"]),
        },
        "environment": environment_info(),
        "slurm": slurm_info(),
        "artifact_dir": str(artifact_dir),
        "dataset": {
            "name": "KuaiRand",
            "protocol": "B",
            "split_used_for_smoke": "train",
            "fingerprint_expected": EXPECTED_FINGERPRINT,
            "identity_hash_expected": EXPECTED_IDENTITY_HASH,
            "validation_only_summary": data.validation_only_summary,
            "loader": {
                "train_batches_available": len(train_data),
                "smoke_batches": batch_count,
                "batch_size": len(batches[0]),
            },
        },
        "test_safety": {
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
            "test_evaluated": False,
            "test_evaluation_count": 0,
        },
        "base_reference_runs": {
            "base": "tim4rec_001",
            "fixed_multitask": "multitask_tim4rec_001",
            "tuned_fixed_multitask": "multitask_tim4rec_tuned_001",
        },
        "target_statistics": target_stats,
        "tuned_fixed_configuration": {
            "source_study": best_params["study_name"],
            "source_trial": int(best_params["trial_number"]),
            "lambda_aux": sampled["lambda_aux"],
            "normalized_task_weights": sampled["normalized_task_weights"],
            "effective_pos_weights": sampled["effective_pos_weights"],
            "effective_loss_multipliers": sampled["effective_loss_multipliers"],
            "head_lr_multiplier": sampled["head_lr_multiplier"],
        },
        "checkpoint": checkpoint_info,
        "model_parameters": count_parameters(baseline_model),
        "shared_parameters": {
            "diagnostic": {
                "selector": adaptive_config["base"]["diagnostic_shared_selector"],
                "definition": "all trainable TiM4Rec backbone parameters; auxiliary linear heads excluded",
                "summary": parameter_group_summary(all_shared),
            },
            "gradnorm": {
                "selector": adaptive_config["gradnorm"]["shared_selector"],
                "definition": "last shared TiSSDLayer parameters for GradNorm norm loss, matching the paper's last-shared-layer recommendation",
                "summary": parameter_group_summary(gradnorm_shared),
            },
        },
        "fixed_tuned_gradient_diagnostics": diagnostics,
        "methods": {
            "fixed_tuned_step": fixed_step,
            "gradnorm": gradnorm,
            "pcgrad": pcgrad,
            "metabalance": metabalance,
        },
        "cost_summary": {
            "fixed_tuned": {
                "mean_step_time_sec": fixed_step["mean_step_time_sec"],
                "max_allocated_bytes": fixed_step["cuda_memory"]["max_allocated_bytes"],
                "autograd_calls_per_step": fixed_step["autograd_calls_per_step"],
            },
            "gradnorm": {
                "mean_step_time_sec": gradnorm["mean_step_time_sec"],
                "max_allocated_bytes": gradnorm["cuda_memory"]["max_allocated_bytes"],
                "autograd_calls_per_step": gradnorm["autograd_calls_per_step"],
            },
            "pcgrad": {
                "mean_step_time_sec": pcgrad["mean_step_time_sec"],
                "max_allocated_bytes": pcgrad["cuda_memory"]["max_allocated_bytes"],
                "autograd_calls_per_step": pcgrad["autograd_calls_per_step"],
            },
            "metabalance": {
                "mean_step_time_sec": metabalance["mean_step_time_sec"],
                "max_allocated_bytes": metabalance["cuda_memory"]["max_allocated_bytes"],
                "autograd_calls_per_step": metabalance["autograd_calls_per_step"],
            },
        },
        "resource": {
            "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    result["recommendation"] = recommendation_from(result)
    save_json(Path(args.output), result)
    write_notes(Path(args.notes), result)


if __name__ == "__main__":
    main()
