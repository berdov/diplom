#!/usr/bin/env python
"""Run validation-only and final-test EPO + MoE experiments."""

from __future__ import annotations

import argparse
import copy
import csv
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
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
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

from experiments.adaptive_multitask_tim4rec.methods.common import (  # noqa: E402
    ensure_finite_gradients,
    parameter_group_summary,
    shared_parameter_entries,
    task_gradient_vectors,
)
from experiments.epo_moe.model import MoEMultitaskTiM4Rec, model_architecture_record, moe_config_from_mapping  # noqa: E402
from experiments.moo_8families.evaluation.objectives import (  # noqa: E402
    gradient_diagnostics,
    scalar_loss_record,
    task_losses,
)
from experiments.moo_8families.evaluation.pareto import (  # noqa: E402
    EVAL_OBJECTIVE_ORDER,
    RANKING_OPERATING_POINT_ID,
    validation_summary_from_records,
)
from experiments.moo_8families.strategies.base import (  # noqa: E402
    AUX_TARGETS,
    TASK_ORDER,
    losses_to_vector,
    preference_tensor,
    sha256_file,
    sha256_json,
    tensor_to_float,
)
from experiments.moo_8families.strategies.epo import ExactParetoPreferenceSolver  # noqa: E402
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS  # noqa: E402
from experiments.multitask_tim4rec.train import (  # noqa: E402
    EXPECTED_FINGERPRINT,
    EXPECTED_IDENTITY_HASH,
    all_gradient_check,
    check_hit_recall_equal,
    count_parameters,
    evaluate_auxiliary,
    evaluate_full_sort_with_checks,
    inspect_eval_loader,
    load_json,
    load_target_stats,
    metric_subset,
)
from experiments.multitask_tim4rec_optuna.optuna_search import (  # noqa: E402
    assert_protocol_config,
    load_data_bundle,
    normalize_metrics,
    optimizer_for_trial,
    pos_weight_tensors,
    project_path,
)
from experiments.multitask_tim4rec_optuna.run_locked_tuned import sampled_from_locked_params  # noqa: E402


EXPERIMENT_DIR = ROOT / "experiments" / "epo_moe"
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "epo_moe.yaml"
STAGES = ("sanity", "validation", "final_test")
METRIC_TOPK = (5, 10, 20, 50)
LOCKED_TEST_SUMMARY = Path(
    "/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/locked_test_recbole/locked_test_dataset.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--stage", choices=STAGES, default="validation")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--routing-json", default=None)
    parser.add_argument("--routing-csv", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--checkpoint-json", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--final-test-checkpoint-json", default=None)
    parser.add_argument("--final-test-run-json", default=None)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=json_default) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def load_json_file(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def git_dir() -> Path:
    path = ROOT / ".git"
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text.startswith("gitdir:"):
            target = Path(text.split(":", 1)[1].strip())
            return target if target.is_absolute() else (ROOT / target).resolve()
    return path


def read_git_ref(ref: str) -> str | None:
    directory = git_dir()
    ref_path = directory / ref
    if ref_path.exists():
        value = ref_path.read_text(encoding="utf-8", errors="ignore").strip()
        return value or None
    packed_refs = directory / "packed-refs"
    if packed_refs.exists():
        for line in packed_refs.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("#") or line.startswith("^"):
                continue
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def git_metadata_fallback(args: list[str], default: str = "unknown") -> str:
    directory = git_dir()
    head_path = directory / "HEAD"
    if args == ["rev-parse", "HEAD"] and head_path.exists():
        head = head_path.read_text(encoding="utf-8", errors="ignore").strip()
        if head.startswith("ref:"):
            return read_git_ref(head.split(":", 1)[1].strip()) or default
        return head or default
    if args == ["rev-parse", "--abbrev-ref", "HEAD"] and head_path.exists():
        head = head_path.read_text(encoding="utf-8", errors="ignore").strip()
        if head.startswith("ref: refs/heads/"):
            return head.removeprefix("ref: refs/heads/")
        return "HEAD"
    if args == ["config", "--get", "remote.origin.url"]:
        config_path = directory / "config"
        if not config_path.exists():
            return default
        in_origin = False
        for raw_line in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if line.startswith("["):
                in_origin = line == '[remote "origin"]'
            elif in_origin and line.startswith("url"):
                _key, _sep, value = line.partition("=")
                return value.strip() or default
    return default


def git_value(args: list[str], default: str = "unknown") -> str:
    env_map = {
        ("rev-parse", "HEAD"): "EPO_MOE_GIT_COMMIT",
        ("rev-parse", "--abbrev-ref", "HEAD"): "EPO_MOE_GIT_BRANCH",
        ("config", "--get", "remote.origin.url"): "EPO_MOE_GIT_REMOTE",
    }
    env_key = env_map.get(tuple(args))
    if env_key and os.environ.get(env_key) and os.environ[env_key] != "unknown":
        return str(os.environ[env_key])
    for git_bin in (os.environ.get("EPO_MOE_GIT_BIN"), "/usr/bin/git", "git"):
        if not git_bin:
            continue
        try:
            value = subprocess.check_output(
                [git_bin, *args],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if value and value != "unknown":
                return value
        except Exception:
            continue
    return git_metadata_fallback(args, default)


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
        "numpy": np.__version__,
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
        "capability": ".".join(map(str, torch.cuda.get_device_capability(current))),
        "device_count": torch.cuda.device_count(),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def source_hashes() -> dict[str, str]:
    relative = [
        "experiments/epo_moe/model.py",
        "experiments/epo_moe/run.py",
        "experiments/epo_moe/summarize.py",
        "experiments/epo_moe/configs/epo_moe.yaml",
        "experiments/multitask_tim4rec/model.py",
        "experiments/multitask_tim4rec_optuna/optuna_search.py",
        "experiments/moo_8families/strategies/base.py",
        "experiments/moo_8families/strategies/epo.py",
        "experiments/moo_8families/evaluation/objectives.py",
        "experiments/moo_8families/evaluation/pareto.py",
        "slurm/epo_moe.sh",
    ]
    return {path: sha256_file(ROOT / path) for path in relative if (ROOT / path).exists()}


def run_config(config: Mapping[str, Any], run_key: str) -> dict[str, Any]:
    try:
        raw = config["runs"][run_key]
    except KeyError as exc:
        available = ", ".join(sorted(config["runs"].keys()))
        raise KeyError(f"Unknown run key {run_key!r}; available: {available}") from exc
    merged = deep_merge(config["moe_defaults"], raw)
    merged["run_key"] = run_key
    merged["run_id"] = str(raw.get("run_id", f"epo_moe_{run_key}_001"))
    merged["label"] = str(raw.get("label", run_key))
    return merged


def resolve_paths(args: argparse.Namespace, config: Mapping[str, Any], run_cfg: Mapping[str, Any]) -> dict[str, Path]:
    run_id = args.run_id or str(run_cfg["run_id"])
    outputs = config["outputs"]
    runs_dir = project_path(outputs["runs_dir"])
    routing_dir = project_path(outputs["routing_dir"])
    artifact_root = Path(outputs["artifact_root"])
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else artifact_root / run_id
    return {
        "artifact_dir": artifact_dir,
        "result_json": Path(args.result_json) if args.result_json else runs_dir / f"{run_id}.json",
        "partial_json": (Path(args.result_json) if args.result_json else runs_dir / f"{run_id}.json").with_suffix(".partial.json"),
        "routing_json": Path(args.routing_json) if args.routing_json else routing_dir / f"{run_id}_routing.json",
        "routing_csv": Path(args.routing_csv) if args.routing_csv else routing_dir / f"{run_id}_routing.csv",
        "notes": Path(args.notes) if args.notes else runs_dir / f"{run_id}_notes.md",
        "checkpoint_json": Path(args.checkpoint_json) if args.checkpoint_json else runs_dir / f"{run_id}_checkpoint.json",
    }


def assert_output_allowed(paths: Mapping[str, Path], artifact_dir: Path, allow_overwrite: bool, resume: bool) -> None:
    if allow_overwrite or resume:
        return
    for key in ("result_json", "routing_json", "routing_csv", "notes", "checkpoint_json"):
        path = paths[key]
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing {key}: {path}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty artifact dir: {artifact_dir}")


def apply_sampled_overrides(sampled: Mapping[str, Any], run_cfg: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(sampled))
    if run_cfg.get("learning_rate_multiplier") is not None:
        result["learning_rate"] = float(result["learning_rate"]) * float(run_cfg["learning_rate_multiplier"])
    if run_cfg.get("head_lr_multiplier_override") is not None:
        result["head_lr_multiplier"] = float(run_cfg["head_lr_multiplier_override"])
    if run_cfg.get("weight_decay") is not None:
        result["weight_decay"] = float(run_cfg["weight_decay"])
    if run_cfg.get("dropout_prob") is not None:
        result["dropout_prob"] = float(run_cfg["dropout_prob"])
    result["head_learning_rate"] = float(result["learning_rate"]) * float(result["head_lr_multiplier"])
    result["effective_loss_multipliers"] = {
        target: float(result["lambda_aux"]) * float(result["normalized_task_weights"][target])
        for target in AUX_TARGETS
    }
    result["effective_positive_multipliers"] = {
        target: float(result["effective_loss_multipliers"][target]) * float(result["effective_pos_weights"][target])
        for target in AUX_TARGETS
    }
    return result


def build_recbole_config(
    optuna_config: Mapping[str, Any],
    run_cfg: Mapping[str, Any],
    artifact_dir: Path,
    sampled: Mapping[str, Any],
    epochs: int,
    model_class: type[MultitaskTiM4Rec],
    *,
    validation_only: bool = True,
    locked_test_summary: Mapping[str, Any] | None = None,
) -> Config:
    overrides = copy.deepcopy(optuna_config["recbole_overrides"])
    if validation_only:
        overrides["benchmark_filename"] = ["train", "valid"]
        overrides["final_test_evaluation_count"] = 0
        overrides["test_evaluation_count"] = 0
    else:
        if locked_test_summary is None:
            raise RuntimeError("locked_test_summary is required for final test config.")
        overrides["data_path"] = locked_test_summary["output_root"]
        overrides["dataset"] = locked_test_summary["dataset"]
        overrides["benchmark_filename"] = ["train", "valid", "test"]
        overrides["eval_args"] = {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            "group_by": "user",
            "mode": "full",
        }
        overrides["final_test_evaluation_count"] = 1
        overrides["test_evaluation_count"] = 1

    overrides.update(
        {
            "checkpoint_dir": str(artifact_dir / "recbole_checkpoints"),
            "epochs": int(epochs),
            "stopping_step": int(epochs) + 1,
            "learning_rate": float(sampled["learning_rate"]),
            "weight_decay": float(sampled["weight_decay"]),
            "dropout_prob": float(sampled["dropout_prob"]),
            "train_batch_size": int(optuna_config["recbole_overrides"].get("train_batch_size", 2048)),
            "eval_batch_size": int(optuna_config["recbole_overrides"].get("eval_batch_size", 4096)),
            "metrics": ["Hit", "Recall", "NDCG"],
            "topk": list(METRIC_TOPK),
            "valid_metric": "NDCG@10",
            "show_progress": False,
            "log_wandb": False,
            **moe_config_from_mapping(run_cfg),
        }
    )
    return Config(
        model=model_class,
        config_file_list=[str(project_path(optuna_config["source"]["base_config"]))],
        config_dict=overrides,
    )


def assert_protocol_guards(config: Mapping[str, Any], data: Any, recbole_config: Any) -> None:
    protocol = config["protocol"]
    expected = {
        "users": int(protocol["users"]),
        "items": int(protocol["items"]),
        "interactions": int(protocol["interactions"]),
        "train": int(protocol["train"]),
        "validation": int(protocol["validation"]),
        "test": int(protocol["test"]),
    }
    if EXPECTED_FINGERPRINT != expected:
        raise RuntimeError(f"Code fingerprint mismatch: {EXPECTED_FINGERPRINT} != {expected}")
    summary = data.validation_only_summary
    observed = summary.get("protocol_fingerprint") or summary.get("dataset_fingerprint")
    if observed != expected:
        raise RuntimeError(f"Protocol B fingerprint mismatch: {observed} != {expected}")
    identity = summary.get("identity_hash")
    if identity != str(protocol["identity_hash"]) or identity != EXPECTED_IDENTITY_HASH:
        raise RuntimeError(f"Identity hash mismatch: {identity}")
    if tuple(recbole_config["multitask_targets"]) != TARGETS:
        raise RuntimeError(f"Task set changed: {recbole_config['multitask_targets']}")
    if tuple(config["protocol"]["task_order"]) != TASK_ORDER:
        raise RuntimeError(f"Configured task order changed: {config['protocol']['task_order']}")
    if not bool(recbole_config["is_time"]):
        raise RuntimeError("TiM4Rec is_time must remain True.")
    if summary.get("forbidden_test_paths_loaded") != []:
        raise RuntimeError(f"Validation-only prep touched test paths: {summary}")
    if bool(summary.get("test_path_passed_to_search")):
        raise RuntimeError(f"Test path passed to validation-only prep: {summary}")
    if int(summary["rows"]["test"]) != 0 or int(summary["test_rows_in_inter_file"]) != 0:
        raise RuntimeError(f"Validation-only RecBole data contains test rows: {summary['rows']}")


def preference_records(preferences: Mapping[str, Any], set_id: str) -> list[dict[str, Any]]:
    return [
        {"id": pref_id, "weights": [float(value) for value in preferences["preferences"][pref_id]["weights"]]}
        for pref_id in preferences["sets"][set_id]
    ]


def load_preferences(path: Path) -> dict[str, Any]:
    payload = load_yaml(path)
    if payload["objective_order"] != list(TASK_ORDER):
        raise RuntimeError(f"Preference task order mismatch: {payload['objective_order']}")
    for pref_id, spec in payload["preferences"].items():
        pref = torch.tensor(spec["weights"], dtype=torch.float32)
        normalized = preference_tensor(pref)
        if abs(float(normalized.sum().item()) - 1.0) > 1e-6:
            raise RuntimeError(f"Preference does not sum to one after normalization: {pref_id}")
    return payload


def evaluation_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    ref = config["evaluation"]["pareto_reference"]
    order = list(ref["objective_order"])
    if order != list(EVAL_OBJECTIVE_ORDER):
        raise RuntimeError(f"Evaluation objective order mismatch: {order} != {list(EVAL_OBJECTIVE_ORDER)}")
    values = [float(value) for value in ref["values"]]
    if values != [1.0, 2.0, 2.0, 2.0, 2.0]:
        raise RuntimeError(f"Unexpected EPO+MoE Pareto reference: {values}")
    if str(ref.get("invalid_reference_policy")) != "raise":
        raise RuntimeError(f"invalid_reference_policy must stay raise: {ref}")
    return {"objective_order": order, "values": values, "invalid_reference_policy": "raise"}


def model_class_for(run_cfg: Mapping[str, Any]) -> type[MultitaskTiM4Rec]:
    return MultitaskTiM4Rec if int(run_cfg["num_experts"]) == 0 else MoEMultitaskTiM4Rec


def epo_gradient_parameter_entries(model: Any) -> list[Any]:
    """Shared EPO parameters: TiM4Rec backbone plus shared MoE experts, excluding task gates."""

    entries = [
        entry
        for entry in shared_parameter_entries(model, "all_backbone")
        if not entry.name.startswith("moe_gates.")
    ]
    if not entries:
        raise RuntimeError("No EPO shared parameters selected.")
    return entries


def instantiate_epo_models(
    config: Mapping[str, Any],
    recbole_config: Config,
    train_dataset: Any,
    sampled: Mapping[str, Any],
    preferences: Sequence[Mapping[str, Any]],
    run_cfg: Mapping[str, Any],
) -> tuple[list[Any], list[Any], list[ExactParetoPreferenceSolver]]:
    models = []
    optimizers = []
    solvers = []
    model_class = model_class_for(run_cfg)
    for idx, pref in enumerate(preferences):
        init_seed(int(config["training"]["seed"]) + idx, recbole_config["reproducibility"])
        model = model_class(recbole_config, train_dataset).to(recbole_config["device"])
        models.append(model)
        optimizers.append(optimizer_for_trial(model, dict(sampled)))
        solvers.append(
            ExactParetoPreferenceSolver(
                pref["weights"],
                eps=float(config["epo"]["eps"]),
                alpha_multiplier=len(TASK_ORDER),
            )
        )
    return models, optimizers, solvers


def create_loaders(config: Config, train_dataset: Any, valid_dataset: Any) -> tuple[Any, FullSortEvalDataLoader]:
    train_loader = get_dataloader(config, "train")(config, train_dataset, None, shuffle=config["shuffle"])
    valid_loader = get_dataloader(config, "valid")(config, valid_dataset, None, shuffle=False)
    if not isinstance(valid_loader, FullSortEvalDataLoader):
        raise RuntimeError(f"Expected FullSortEvalDataLoader, got {type(valid_loader).__name__}")
    return train_loader, valid_loader


def train_epo_epoch(
    *,
    models: list[Any],
    optimizers: list[Any],
    solvers: list[ExactParetoPreferenceSolver],
    train_data: Any,
    sampled: Mapping[str, Any],
    pos_weights: Mapping[str, Any],
    loss_scales: Sequence[float],
    max_batches: int | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sums: dict[str, float] = {}
    solution_sums: list[dict[str, float]] = [{} for _ in models]
    examples = 0
    batches = 0
    first_diag = None
    solver_fallback_count = 0
    solver_fallback_by_type: dict[str, int] = {}
    solution_fallback_counts = [0 for _ in models]

    for interaction in train_data:
        interaction = interaction.to(next(models[0].parameters()).device)
        batch_size = len(interaction)
        for idx, (model, optimizer, solver, sol_sums) in enumerate(zip(models, optimizers, solvers, solution_sums)):
            model.train()
            shared_entries = epo_gradient_parameter_entries(model)
            optimizer.zero_grad(set_to_none=True)
            probe = task_losses(model, interaction, dict(sampled), dict(pos_weights), loss_scales=loss_scales)
            vectors = task_gradient_vectors(
                {task: probe["normalized_task_vector"][task_idx] for task_idx, task in enumerate(TASK_ORDER)},
                shared_entries,
                TASK_ORDER,
            )
            gradient_matrix = torch.stack([vectors[task] for task in TASK_ORDER], dim=0)
            alpha = solver.alpha(probe["normalized_task_vector"].detach(), gradient_matrix)
            fallback = None if solver.last_result is None else solver.last_result.get("fallback")
            if fallback:
                fallback_name = str(fallback)
                solver_fallback_count += 1
                solution_fallback_counts[idx] += 1
                solver_fallback_by_type[fallback_name] = solver_fallback_by_type.get(fallback_name, 0) + 1
            optimizer.zero_grad(set_to_none=True)
            losses = task_losses(model, interaction, dict(sampled), dict(pos_weights), loss_scales=loss_scales)
            scalar = (alpha.detach() * losses["normalized_task_vector"]).sum()
            scalar.backward()
            if not torch.isfinite(scalar):
                raise RuntimeError(f"Non-finite EPO scalar loss: {tensor_to_float(scalar)}")
            finite = all_gradient_check(model)
            shared_finite = ensure_finite_gradients(shared_entries)
            if not bool(finite["all_finite"]) or not bool(shared_finite["all_finite"]):
                raise RuntimeError(f"Non-finite gradients: model={finite}, shared={shared_finite}")
            optimizer.step()
            for key, value in scalar_loss_record(losses).items():
                sol_sums[key] = sol_sums.get(key, 0.0) + value * batch_size
                sums[key] = sums.get(key, 0.0) + value * batch_size
            sol_sums["moo_scalar"] = sol_sums.get("moo_scalar", 0.0) + tensor_to_float(scalar) * batch_size
            sums["moo_scalar"] = sums.get("moo_scalar", 0.0) + tensor_to_float(scalar) * batch_size
            if first_diag is None:
                fresh = task_losses(model, interaction, dict(sampled), dict(pos_weights), loss_scales=loss_scales)
                first_diag = gradient_diagnostics(model, fresh, selector="all_backbone") | {
                    "epo": solver.state_dict(),
                    "epo_shared_parameter_summary": parameter_group_summary(shared_entries),
                }
        examples += batch_size
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    denom = max(examples * len(models), 1)
    aggregate = {key: value / denom for key, value in sums.items()}
    aggregate["batches"] = int(batches)
    aggregate["examples"] = int(examples)
    aggregate["solutions"] = [
        summarize_epoch(solution_sums[idx], examples, batches)
        | {"epo": solvers[idx].state_dict(), "solver_fallback_count": int(solution_fallback_counts[idx])}
        for idx in range(len(models))
    ]
    aggregate["solver_fallback_count"] = int(solver_fallback_count)
    aggregate["solver_fallback_by_type"] = solver_fallback_by_type
    aggregate["method_state"] = {"solvers": [solver.state_dict() for solver in solvers]}
    return aggregate, first_diag


def summarize_epoch(sums: Mapping[str, float], examples: int, batches: int) -> dict[str, Any]:
    if examples <= 0:
        raise RuntimeError("No training examples.")
    result = {key: float(value) / examples for key, value in sums.items()}
    result["batches"] = int(batches)
    result["examples"] = int(examples)
    if "rank" in result:
        rank = result["rank"]
        result["auxiliary_rank_ratio"] = None if rank == 0 else result.get("weighted_aux_sum", 0.0) / rank
    return result


def evaluate_models(
    *,
    models: list[Any],
    trainers: list[Any],
    valid_data: Any,
    train_data: Any,
    topk: Sequence[int],
    pareto_reference_point: Sequence[float],
    preferences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = []
    for idx, model in enumerate(models):
        valid_result, checks = evaluate_full_sort_with_checks(trainers[idx], valid_data, train_data)
        check_hit_recall_equal(valid_result, list(topk))
        if not checks["raw_scores_all_finite"] or not checks["positive_scores_all_finite"]:
            raise RuntimeError(f"Non-finite validation scores: {checks}")
        aux = evaluate_auxiliary(model, valid_data, next(model.parameters()).device)
        records.append(
            {
                "solution_index": idx,
                "preference_id": preferences[idx]["id"],
                "preference": preferences[idx]["weights"],
                "metrics": normalize_metrics(metric_subset(valid_result)),
                "auxiliary_validation": aux,
                "checks": checks,
            }
        )
    return validation_summary_from_records(
        records,
        method="epo",
        reference_point=pareto_reference_point,
        ranking_preference_id=RANKING_OPERATING_POINT_ID,
    )


def save_checkpoint(
    path: Path,
    models: Sequence[Any],
    optimizers: Sequence[Any],
    epoch: int,
    best_score: float | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_valid_score": best_score,
        "models": [{key: value.detach().cpu() for key, value in model.state_dict().items()} for model in models],
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "metadata": dict(metadata),
    }
    torch.save(payload, path, pickle_protocol=4)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_checkpoint(path: Path, models: Sequence[Any], optimizers: Sequence[Any] | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location=next(models[0].parameters()).device, weights_only=False)
    for model, state in zip(models, payload["models"]):
        model.load_state_dict(state, strict=True)
    if optimizers is not None and "optimizers" in payload:
        for optimizer, state in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
    return payload


def parameter_summary(models: Sequence[Any]) -> dict[str, Any]:
    return {
        "model_count": len(models),
        "total_parameters": int(sum(count_parameters(model)["total"] for model in models)),
        "total_trainable": int(sum(count_parameters(model)["trainable"] for model in models)),
        "per_model": [count_parameters(model) for model in models],
        "shared": [parameter_group_summary(shared_parameter_entries(model, "all_backbone")) for model in models],
        "epo_shared": [parameter_group_summary(epo_gradient_parameter_entries(model)) for model in models],
        "architecture": [model_architecture_record(model) for model in models],
    }


def grad_norm_by_name(model: Any, predicate: Any) -> float | None:
    values = []
    for name, param in model.named_parameters():
        if param.grad is None or not predicate(name):
            continue
        values.append(torch.linalg.vector_norm(param.grad.detach().float()))
    if not values:
        return None
    return float(torch.linalg.vector_norm(torch.stack(values)).cpu().item())


def finite_gradients_by_name(model: Any, predicate: Any) -> dict[str, Any]:
    checked = 0
    nonfinite = []
    for name, param in model.named_parameters():
        if param.grad is None or not predicate(name):
            continue
        checked += 1
        if not torch.isfinite(param.grad).all().item():
            nonfinite.append(name)
    return {
        "checked_tensors": checked,
        "nonfinite_tensor_count": len(nonfinite),
        "nonfinite_tensors": nonfinite[:10],
        "all_finite": len(nonfinite) == 0,
    }


def head_prefix_for_task(task: str) -> str | None:
    return {
        "is_click": "click_head",
        "long_view": "long_view_head",
        "is_like": "like_head",
        "is_profile_enter": "profile_enter_head",
    }.get(task)


def is_timt4rec_backbone_parameter(name: str) -> bool:
    head_tokens = ("click_head", "long_view_head", "like_head", "profile_enter_head")
    return not name.startswith("moe_") and not any(token in name for token in head_tokens)


def sanity_gradient_check(
    model: Any,
    solver: ExactParetoPreferenceSolver,
    train_data: Any,
    sampled: Mapping[str, Any],
    pos_weights: Mapping[str, Any],
    loss_scales: Sequence[float],
) -> dict[str, Any]:
    try:
        interaction = next(iter(train_data)).to(next(model.parameters()).device)
    except StopIteration as exc:
        raise RuntimeError("Cannot run sanity gradient check: train loader is empty.") from exc

    model.train()
    losses = task_losses(model, interaction, dict(sampled), dict(pos_weights), loss_scales=loss_scales)
    raw_losses = {task: losses["rank" if task == "rank" else f"{task}_loss"] for task in TASK_ORDER}
    raw_loss_values = {task: tensor_to_float(loss) for task, loss in raw_losses.items()}
    if not all(math.isfinite(value) for value in raw_loss_values.values()):
        raise RuntimeError(f"Non-finite sanity raw losses: {raw_loss_values}")

    routing = {"moe_enabled": hasattr(model, "routing_probabilities_from_representation")}
    if hasattr(model, "routing_probabilities_from_representation"):
        with torch.no_grad():
            representation = model.shared_representation(interaction)
            probs = model.routing_probabilities_from_representation(representation)
            routing["gate_probability_sum_max_abs_error"] = {
                task: float((value.sum(dim=-1) - 1.0).abs().max().detach().cpu().item())
                for task, value in probs.items()
            }
            routing["gate_probabilities_all_finite"] = {
                task: bool(torch.isfinite(value).all().detach().cpu().item())
                for task, value in probs.items()
            }

    per_task = {}
    for task in TASK_ORDER:
        model.zero_grad(set_to_none=True)
        task_losses_payload = task_losses(model, interaction, dict(sampled), dict(pos_weights), loss_scales=loss_scales)
        loss = task_losses_payload["rank" if task == "rank" else f"{task}_loss"]
        loss.backward()
        task_head_prefix = head_prefix_for_task(task)
        record = {
            "loss": tensor_to_float(loss),
            "all_model_gradients_finite": all_gradient_check(model),
            "tim4rec_backbone_grad_norm": grad_norm_by_name(model, is_timt4rec_backbone_parameter),
            "experts_grad_norm": grad_norm_by_name(model, lambda name: name.startswith("moe_experts")),
            "task_gate_grad_norm": grad_norm_by_name(model, lambda name, current=task: name.startswith(f"moe_gates.{current}.")),
            "task_head_grad_norm": None
            if task_head_prefix is None
            else grad_norm_by_name(model, lambda name, prefix=task_head_prefix: name.startswith(prefix)),
            "task_gate_gradients_finite": finite_gradients_by_name(
                model,
                lambda name, current=task: name.startswith(f"moe_gates.{current}."),
            ),
            "task_head_gradients_finite": None
            if task_head_prefix is None
            else finite_gradients_by_name(model, lambda name, prefix=task_head_prefix: name.startswith(prefix)),
        }
        per_task[task] = record

    model.zero_grad(set_to_none=True)
    probe = task_losses(model, interaction, dict(sampled), dict(pos_weights), loss_scales=loss_scales)
    entries = epo_gradient_parameter_entries(model)
    vectors = task_gradient_vectors(
        {task: probe["normalized_task_vector"][idx] for idx, task in enumerate(TASK_ORDER)},
        entries,
        TASK_ORDER,
    )
    alpha = solver.alpha(probe["normalized_task_vector"].detach(), torch.stack([vectors[task] for task in TASK_ORDER], dim=0))
    model.zero_grad(set_to_none=True)
    scalar_losses = task_losses(model, interaction, dict(sampled), dict(pos_weights), loss_scales=loss_scales)
    scalar = (alpha.detach() * scalar_losses["normalized_task_vector"]).sum()
    scalar.backward()
    epo_backward = {
        "scalar": tensor_to_float(scalar),
        "alpha": [tensor_to_float(value) for value in alpha],
        "solver": solver.state_dict(),
        "epo_shared_parameter_summary": parameter_group_summary(entries),
        "all_model_gradients_finite": all_gradient_check(model),
        "tim4rec_backbone_grad_norm": grad_norm_by_name(model, is_timt4rec_backbone_parameter),
        "experts_grad_norm": grad_norm_by_name(model, lambda name: name.startswith("moe_experts")),
        "gates_grad_norm": grad_norm_by_name(model, lambda name: name.startswith("moe_gates")),
        "heads_grad_norm": grad_norm_by_name(
            model,
            lambda name: any(token in name for token in ("click_head", "long_view_head", "like_head", "profile_enter_head")),
        ),
    }
    model.zero_grad(set_to_none=True)

    if routing["moe_enabled"]:
        max_sum_error = max(routing["gate_probability_sum_max_abs_error"].values())
        if max_sum_error > 1e-5:
            raise RuntimeError(f"Gate probabilities do not sum to one: {routing}")
        for task, record in per_task.items():
            if record["tim4rec_backbone_grad_norm"] is None or record["tim4rec_backbone_grad_norm"] <= 0.0:
                raise RuntimeError(f"No TiM4Rec gradient for task {task}: {record}")
            if record["experts_grad_norm"] is None or record["experts_grad_norm"] <= 0.0:
                raise RuntimeError(f"No expert gradient for task {task}: {record}")
            if record["task_gate_grad_norm"] is None or record["task_gate_grad_norm"] <= 0.0:
                raise RuntimeError(f"No gate gradient for task {task}: {record}")
            if task != "rank" and (record["task_head_grad_norm"] is None or record["task_head_grad_norm"] <= 0.0):
                raise RuntimeError(f"No head gradient for task {task}: {record}")

    return {
        "status": "passed",
        "batch_size": len(interaction),
        "task_order": list(TASK_ORDER),
        "raw_losses": raw_loss_values,
        "all_raw_losses_finite": True,
        "routing": routing,
        "per_task_backward": per_task,
        "epo_backward": epo_backward,
    }


def optimizer_learning_rates(optimizers: Sequence[Any]) -> dict[str, Any]:
    per_optimizer = [[float(group["lr"]) for group in optimizer.param_groups] for optimizer in optimizers]
    unique = sorted({lr for values in per_optimizer for lr in values})
    return {"unique": unique, "per_optimizer": per_optimizer}


def routing_statistics_for_model(
    model: Any,
    data_loader: Any,
    *,
    max_batches: int | None,
    collapse_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not hasattr(model, "routing_probabilities_from_representation"):
        return {"moe_enabled": False}
    model.eval()
    device = next(model.parameters()).device
    stores: dict[str, list[np.ndarray]] = {task: [] for task in TASK_ORDER}
    batches = 0
    examples = 0
    with torch.no_grad():
        for batch in data_loader:
            interaction = batch[0] if isinstance(batch, (tuple, list)) else batch
            interaction = interaction.to(device)
            representation = model.shared_representation(interaction)
            probs = model.routing_probabilities_from_representation(representation)
            for task, tensor in probs.items():
                stores[task].append(tensor.detach().float().cpu().numpy())
            batches += 1
            examples += len(interaction)
            if max_batches is not None and batches >= max_batches:
                break
    if examples <= 0:
        raise RuntimeError("No examples for routing diagnostics.")

    dead_threshold = float(collapse_config["dead_expert_threshold"])
    dominant_threshold = float(collapse_config["dominant_expert_threshold"])
    uniform_threshold = float(collapse_config["uniform_l1_threshold"])
    task_invariant_threshold = float(collapse_config["task_invariant_l1_threshold"])
    per_task = {}
    task_means = []
    for task, chunks in stores.items():
        values = np.concatenate(chunks, axis=0)
        mean = values.mean(axis=0)
        median = np.median(values, axis=0)
        std = values.std(axis=0)
        top1 = values.argmax(axis=1)
        top1_frequency = np.bincount(top1, minlength=values.shape[1]) / values.shape[0]
        utilization = mean
        entropy = -np.sum(values * np.log(np.clip(values, 1e-12, 1.0)), axis=1)
        dominant_expert = int(np.argmax(mean))
        dominant_share = float(mean[dominant_expert])
        uniform = np.ones_like(mean) / len(mean)
        uniform_l1 = float(np.abs(mean - uniform).sum())
        per_task[task] = {
            "mean": mean.astype(float).tolist(),
            "median": median.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "expert_utilization": utilization.astype(float).tolist(),
            "top1_frequency": top1_frequency.astype(float).tolist(),
            "gate_entropy_mean": float(entropy.mean()),
            "gate_entropy_std": float(entropy.std()),
            "dominant_expert": dominant_expert,
            "dominant_share": dominant_share,
            "dead_expert_indices": [int(idx) for idx, value in enumerate(utilization) if float(value) < dead_threshold],
            "dominant_collapse": dominant_share > dominant_threshold,
            "uniform_l1_distance": uniform_l1,
            "uniform_collapse": uniform_l1 < uniform_threshold,
        }
        task_means.append(mean)

    pairwise_l1 = {}
    for i, left in enumerate(TASK_ORDER):
        for right in TASK_ORDER[i + 1 :]:
            pairwise_l1[f"{left}|{right}"] = float(np.abs(task_means[i] - task_means[TASK_ORDER.index(right)]).sum())
    mean_pairwise_l1 = float(np.mean(list(pairwise_l1.values()))) if pairwise_l1 else None
    severe = any(row["dominant_collapse"] or row["dead_expert_indices"] for row in per_task.values())
    task_invariant = bool(mean_pairwise_l1 is not None and mean_pairwise_l1 < task_invariant_threshold)
    return {
        "moe_enabled": True,
        "split": "validation",
        "batches": int(batches),
        "examples": int(examples),
        "task_order": list(TASK_ORDER),
        "per_task": per_task,
        "pairwise_mean_l1": pairwise_l1,
        "mean_pairwise_task_l1": mean_pairwise_l1,
        "collapse": {
            "dead_expert_threshold": dead_threshold,
            "dominant_expert_threshold": dominant_threshold,
            "uniform_l1_threshold": uniform_threshold,
            "task_invariant_l1_threshold": task_invariant_threshold,
            "any_dead_expert": any(bool(row["dead_expert_indices"]) for row in per_task.values()),
            "any_dominant_expert_gt_threshold": any(bool(row["dominant_collapse"]) for row in per_task.values()),
            "any_uniform_router": any(bool(row["uniform_collapse"]) for row in per_task.values()),
            "task_invariant": task_invariant,
            "severe_collapse": bool(severe or task_invariant),
        },
    }


def routing_diagnostics(
    models: Sequence[Any],
    valid_data: Any,
    preferences: Sequence[Mapping[str, Any]],
    run_cfg: Mapping[str, Any],
    routing_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_cfg["run_id"],
        "moe_num_experts": int(run_cfg["num_experts"]),
        "models": [
            {
                "solution_index": idx,
                "preference_id": preferences[idx]["id"],
                "preference": preferences[idx]["weights"],
                "routing": routing_statistics_for_model(
                    model,
                    valid_data,
                    max_batches=None if routing_cfg["max_batches"] is None else int(routing_cfg["max_batches"]),
                    collapse_config=routing_cfg["collapse"],
                ),
            }
            for idx, model in enumerate(models)
        ],
    }


def write_routing_csv(path: Path, routing: Mapping[str, Any]) -> None:
    rows = []
    for model_record in routing["models"]:
        route = model_record["routing"]
        if not route.get("moe_enabled"):
            rows.append(
                {
                    "solution_index": model_record["solution_index"],
                    "preference_id": model_record["preference_id"],
                    "task": "",
                    "expert": "",
                    "mean_probability": "",
                    "top1_frequency": "",
                    "gate_entropy_mean": "",
                    "dominant_expert": "",
                    "dominant_share": "",
                }
            )
            continue
        for task, stats in route["per_task"].items():
            for expert_idx, mean_probability in enumerate(stats["mean"]):
                rows.append(
                    {
                        "solution_index": model_record["solution_index"],
                        "preference_id": model_record["preference_id"],
                        "task": task,
                        "expert": expert_idx,
                        "mean_probability": mean_probability,
                        "top1_frequency": stats["top1_frequency"][expert_idx],
                        "gate_entropy_mean": stats["gate_entropy_mean"],
                        "dominant_expert": stats["dominant_expert"],
                        "dominant_share": stats["dominant_share"],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_notes(path: Path, result: Mapping[str, Any]) -> None:
    metrics = result.get("validation", {}).get("ranking_operating_point", {}).get("metrics", {})
    lines = [
        f"# {result['run_id']}",
        "",
        f"- Stage: `{result['stage']}`.",
        f"- MoE experts: `{result['architecture']['num_experts']}`.",
        f"- EPO preference set: `{result['preferences']['used_for_training']}`.",
        f"- Task order: `{result['preferences']['objective_order']}`.",
        f"- TEST evaluations: `{result['test_evaluation_count']}`.",
        "",
    ]
    if metrics:
        lines += [
            "| HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {metrics['HR@10']:.4f} | {metrics['HR@20']:.4f} | {metrics['HR@50']:.4f} | "
                f"{metrics['NDCG@10']:.4f} | {metrics['NDCG@20']:.4f} | {metrics['NDCG@50']:.4f} |"
            ),
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def checkpoint_resume_check(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    run_cfg: Mapping[str, Any],
    recbole_config: Config,
    train_dataset: Any,
    sampled: Mapping[str, Any],
    pref_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    models, optimizers, _solvers = instantiate_epo_models(
        config,
        recbole_config,
        train_dataset,
        sampled,
        pref_records,
        run_cfg,
    )
    payload = load_checkpoint(checkpoint_path, models, optimizers)
    return {
        "status": "passed",
        "checkpoint_path": str(checkpoint_path),
        "epoch": int(payload["epoch"]),
        "model_count": len(models),
        "optimizer_count": len(optimizers),
        "strict_model_load": True,
        "optimizer_load": True,
    }


def validation_stage(args: argparse.Namespace, config: Mapping[str, Any], run_cfg: Mapping[str, Any]) -> dict[str, Any]:
    paths = resolve_paths(args, config, run_cfg)
    assert_output_allowed(paths, paths["artifact_dir"], args.allow_overwrite, args.resume)
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    train_log = paths["artifact_dir"] / "training_log.jsonl"
    start_time = time.monotonic()
    started_at = datetime.now(timezone.utc)

    optuna_config = load_yaml(project_path(config["source"]["optuna_config"]))
    assert_protocol_config(optuna_config)
    best_params = load_yaml(project_path(config["source"]["best_params"]))
    data = load_data_bundle(optuna_config, paths["artifact_dir"] / "data_probe")
    sampled = apply_sampled_overrides(sampled_from_locked_params(best_params, data.target_stats), run_cfg)
    stage_epochs = int(args.epochs or (config["training"]["sanity_epochs"] if args.stage == "sanity" else config["training"]["max_epochs"]))
    max_batches = args.max_batches
    if args.stage == "sanity" and max_batches is None:
        max_batches = int(config["training"]["sanity_max_batches"])
    if args.stage == "validation" and max_batches is not None:
        raise RuntimeError("Full validation runs must use the full training split; max_batches is sanity-only.")

    recbole_config = build_recbole_config(
        optuna_config,
        run_cfg,
        paths["artifact_dir"],
        sampled,
        stage_epochs,
        model_class_for(run_cfg),
        validation_only=True,
    )
    assert_protocol_guards(config, data, recbole_config)
    init_seed(int(config["training"]["seed"]) + recbole_config["local_rank"], recbole_config["reproducibility"])
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required; use cHARISMa type_e or pass --allow-cpu only for debugging.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_data, valid_data = create_loaders(recbole_config, data.train_dataset, data.valid_dataset)
    pos_weights = pos_weight_tensors(sampled["effective_pos_weights"], recbole_config["device"])
    preferences = load_preferences(project_path(config["source"]["preferences"]))
    pref_records = preference_records(preferences, config["epo"]["preference_set"])
    if len(pref_records) != int(config["epo"]["solution_count"]):
        raise RuntimeError(f"Expected {config['epo']['solution_count']} EPO solutions, got {len(pref_records)}")
    models, optimizers, solvers = instantiate_epo_models(config, recbole_config, train_data.dataset, sampled, pref_records, run_cfg)
    normalization = compute_normalization(
        models[0],
        train_data,
        sampled=sampled,
        pos_weights=pos_weights,
        batches=int(config["normalization"]["diagnostic_batches"]),
        selector=str(config["normalization"]["gradient_selector"]),
    )
    loss_scales = [float(value) for value in normalization["loss_scales"]]
    sanity_gradient = sanity_gradient_check(
        models[0],
        solvers[0],
        train_data,
        sampled,
        pos_weights,
        loss_scales,
    )
    trainers = [Trainer(recbole_config, model) for model in models]
    for trainer, optimizer in zip(trainers, optimizers):
        trainer.optimizer = optimizer

    start_epoch = 1
    best_validation = None
    best_epoch = None
    best_score = -float("inf")
    epochs_payload: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    validation_checks = 0
    checks_without_improvement = 0
    early_stopped = False
    stop_reason = "max_epochs_reached"
    best_checkpoint = None
    last_checkpoint = None
    diagnostics = []
    if args.resume:
        if not paths["partial_json"].exists():
            raise FileNotFoundError(f"Resume requested but partial JSON is missing: {paths['partial_json']}")
        partial = load_json_file(paths["partial_json"])
        checkpoint = Path(partial["checkpoints"]["last"]["path"])
        load_checkpoint(checkpoint, models, optimizers)
        start_epoch = int(partial["epoch"]) + 1
        best_validation = partial.get("best_validation")
        best_epoch = partial.get("best_epoch")
        best_score = float(partial.get("best_validation_NDCG@10") or -float("inf"))
        epochs_payload = list(partial.get("training_epochs", []))
        validation_history = list(partial.get("validation_history", []))
        validation_checks = int(partial.get("validation_checks", 0))
        checks_without_improvement = int(partial.get("checks_without_improvement", 0))
        best_checkpoint = partial.get("checkpoints", {}).get("best_validation")

    evaluation_ref = evaluation_reference(config)
    validation_interval = 1 if args.stage == "sanity" else int(config["training"]["validation_interval"])
    train_started = time.monotonic()

    for epoch in range(start_epoch, stage_epochs + 1):
        epoch_start = time.monotonic()
        losses, first_diag = train_epo_epoch(
            models=models,
            optimizers=optimizers,
            solvers=solvers,
            train_data=train_data,
            sampled=sampled,
            pos_weights=pos_weights,
            loss_scales=loss_scales,
            max_batches=max_batches,
        )
        train_time = time.monotonic() - epoch_start
        if first_diag is not None:
            diagnostics.append({"epoch": epoch, **first_diag})

        validation_payload = None
        improved = False
        if epoch % validation_interval == 0:
            valid_start = time.monotonic()
            validation_payload = evaluate_models(
                models=models,
                trainers=trainers,
                valid_data=valid_data,
                train_data=train_data,
                topk=list(recbole_config["topk"]),
                pareto_reference_point=evaluation_ref["values"],
                preferences=pref_records,
            )
            validation_payload["validation_time_sec"] = float(time.monotonic() - valid_start)
            validation_checks += 1
            score = float(validation_payload["ranking_operating_point"]["metrics"]["NDCG@10"])
            min_delta = float(config["training"]["min_delta"])
            if score > best_score + min_delta:
                improved = True
                best_score = score
                best_epoch = epoch
                best_validation = validation_payload
                checks_without_improvement = 0
                best_checkpoint = save_checkpoint(
                    paths["artifact_dir"] / "checkpoints" / "best_validation.pth",
                    models,
                    optimizers,
                    epoch,
                    best_score,
                    {
                        "run_id": run_cfg["run_id"],
                        "run_key": run_cfg["run_key"],
                        "stage": args.stage,
                        "model_selection": validation_payload["ranking_operating_point_selection"],
                    },
                )
            else:
                checks_without_improvement += 1
            validation_history.append(
                {
                    "epoch": epoch,
                    "validation_check": validation_checks,
                    "ranking_operating_point_NDCG@10": score,
                    "oracle_best_validation_NDCG@10": float(
                        validation_payload["oracle_best_validation_point"]["metrics"]["NDCG@10"]
                    ),
                    "improved": improved,
                    "checks_without_improvement": checks_without_improvement,
                }
            )
            if (
                args.stage == "validation"
                and epoch >= int(config["training"]["minimum_training_epochs"])
                and checks_without_improvement >= int(config["training"]["patience_validation_checks"])
            ):
                early_stopped = True
                stop_reason = "early_stopping_patience"

        last_checkpoint = save_checkpoint(
            paths["artifact_dir"] / "checkpoints" / "last.pth",
            models,
            optimizers,
            epoch,
            best_score if math.isfinite(best_score) else None,
            {"run_id": run_cfg["run_id"], "run_key": run_cfg["run_key"], "stage": args.stage},
        )
        epoch_record = {
            "epoch": epoch,
            "losses": losses,
            "validation": validation_payload,
            "train_time_sec": float(train_time),
            "epoch_time_sec": float(time.monotonic() - epoch_start),
            "learning_rate": optimizer_learning_rates(optimizers),
            "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        }
        epochs_payload.append(epoch_record)
        write_jsonl(train_log, epoch_record)
        partial = {
            "run_id": run_cfg["run_id"],
            "run_key": run_cfg["run_key"],
            "status": "partial",
            "stage": args.stage,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_NDCG@10": None if not math.isfinite(best_score) else best_score,
            "best_validation": best_validation,
            "validation_checks": validation_checks,
            "checks_without_improvement": checks_without_improvement,
            "training_epochs": epochs_payload,
            "validation_history": validation_history,
            "checkpoints": {"best_validation": best_checkpoint, "last": last_checkpoint},
            "test_evaluation_count": 0,
        }
        save_json(paths["partial_json"], partial)
        print(
            json.dumps(
                {
                    "run_id": run_cfg["run_id"],
                    "stage": args.stage,
                    "epoch": epoch,
                    "train_loss_scalar": losses.get("moo_scalar"),
                    "ranking_operating_point_ndcg10": None
                    if validation_payload is None
                    else validation_payload["ranking_operating_point"]["metrics"]["NDCG@10"],
                    "best_ndcg10": None if not math.isfinite(best_score) else best_score,
                    "improved": improved if validation_payload is not None else None,
                    "early_stopped": early_stopped,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            flush=True,
        )
        if early_stopped:
            break

    if best_validation is None or best_epoch is None or best_checkpoint is None:
        raise RuntimeError(f"{args.stage} finished without a best validation checkpoint.")

    checkpoint_payload = load_checkpoint(Path(best_checkpoint["path"]), models, optimizers)
    routing = routing_diagnostics(models, valid_data, pref_records, run_cfg, config["routing_diagnostics"])
    save_json(paths["routing_json"], routing)
    write_routing_csv(paths["routing_csv"], routing)
    resume_check = (
        checkpoint_resume_check(
            Path(best_checkpoint["path"]),
            config,
            run_cfg,
            recbole_config,
            train_data.dataset,
            sampled,
            pref_records,
        )
        if args.stage == "sanity"
        else {"status": "not_run", "reason": "validation run"}
    )

    total_runtime = time.monotonic() - start_time
    result = {
        "run_id": run_cfg["run_id"],
        "run_key": run_cfg["run_key"],
        "status": "completed",
        "record_type": "epo_moe_validation" if args.stage == "validation" else "epo_moe_sanity",
        "stage": args.stage,
        "objective": "validation_full_ranking_NDCG@10_EPO_MoE",
        "created_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "starting_sha": config["starting_commit"],
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": git_value(["config", "--get", "remote.origin.url"]),
        },
        "source_files": source_hashes(),
        "source_checksum": sha256_json(source_hashes()),
        "config_checksum": sha256_json(config),
        "environment": environment_info(),
        "slurm": slurm_info(),
        "gpu": gpu_info(),
        "memory": {"process_ru_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
        "architecture": model_architecture_record(models[0]),
        "run_config": dict(run_cfg),
        "exact_epo_baseline_reference": dict(config["exact_epo_baseline"]),
        "tuned_fixed_configuration": {
            "source_study": best_params["study_name"],
            "source_trial": int(best_params["trial_number"]),
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
            "locked_param_diff_checks": sampled.get("locked_param_diff_checks"),
        },
        "preferences": {
            "objective_order": list(TASK_ORDER),
            "used_for_training": config["epo"]["preference_set"],
            "used_for_validation": config["epo"]["preference_set"],
            "ranking_operating_point_id": RANKING_OPERATING_POINT_ID,
            "records": pref_records,
            "source": str(project_path(config["source"]["preferences"])),
        },
        "dataset": {
            "name": "KuaiRand",
            "protocol": "B",
            "fingerprint_expected": EXPECTED_FINGERPRINT,
            "identity_hash_expected": EXPECTED_IDENTITY_HASH,
            "validation_only_summary": data.validation_only_summary,
            "loader": {
                "train_batches": len(train_data),
                "valid_batches": len(valid_data),
                "train_examples": len(data.train_dataset),
                "validation_examples": len(data.valid_dataset),
                "batch_size": int(recbole_config["train_batch_size"]),
            },
        },
        "normalization": normalization,
        "sanity_gradient_check": sanity_gradient,
        "evaluation": {
            "pareto_reference": evaluation_ref,
            "ranking_operating_point_id": RANKING_OPERATING_POINT_ID,
            "model_selection_metric": config["evaluation"]["model_selection_metric"],
        },
        "training": {
            "seed": int(config["training"]["seed"]),
            "epochs": epochs_payload,
            "requested_epochs": stage_epochs,
            "actual_epochs": len(epochs_payload),
            "stop_epoch": epochs_payload[-1]["epoch"],
            "validation_interval": validation_interval,
            "validation_checks": validation_checks,
            "early_stopping": {
                "metric": config["evaluation"]["model_selection_metric"],
                "mode": "maximize",
                "minimum_training_epochs": int(config["training"]["minimum_training_epochs"]),
                "patience_validation_checks": int(config["training"]["patience_validation_checks"]),
                "min_delta": float(config["training"]["min_delta"]),
            },
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
            "max_batches": max_batches,
            "training_log_jsonl": str(train_log),
        },
        "validation_history": validation_history,
        "validation": best_validation,
        "best_epoch": best_epoch,
        "best_validation_NDCG@10": float(best_score),
        "gradient_diagnostics": diagnostics,
        "routing": {
            "json": str(paths["routing_json"]),
            "csv": str(paths["routing_csv"]),
            "summary": routing,
        },
        "model_parameters": parameter_summary(models),
        "checkpoints": {
            "best_validation": best_checkpoint,
            "last": last_checkpoint,
            "best_checkpoint_load_for_routing": {
                "status": "passed",
                "epoch": int(checkpoint_payload["epoch"]),
                "strict_model_load": True,
            },
            "resume_check": resume_check,
        },
        "test_safety": {
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
            "test_evaluated": False,
            "test_evaluation_count": 0,
        },
        "test_evaluation_count": 0,
        "runtime": {
            "total_sec": float(total_runtime),
            "train_total_sec": float(time.monotonic() - train_started),
            "mean_epoch_sec": float(sum(item["epoch_time_sec"] for item in epochs_payload) / len(epochs_payload)),
        },
    }
    save_json(paths["result_json"], result)
    save_json(paths["checkpoint_json"], {"run_id": run_cfg["run_id"], "best_validation_checkpoint": best_checkpoint})
    save_notes(paths["notes"], result)
    if paths["partial_json"].exists():
        paths["partial_json"].unlink()
    return result


def compute_normalization(
    model: Any,
    train_data: Any,
    *,
    sampled: Mapping[str, Any],
    pos_weights: Mapping[str, Any],
    batches: int,
    selector: str,
) -> dict[str, Any]:
    model.train()
    sums = {task: 0.0 for task in TASK_ORDER}
    sums_sq = {task: 0.0 for task in TASK_ORDER}
    seen_batches = 0
    examples = 0
    first_gradient_diagnostic = None
    for interaction in train_data:
        interaction = interaction.to(next(model.parameters()).device)
        losses = task_losses(model, interaction, dict(sampled), dict(pos_weights), loss_scales=None)
        vector = losses_to_vector(losses)
        for index, task in enumerate(TASK_ORDER):
            value = tensor_to_float(vector[index])
            sums[task] += value
            sums_sq[task] += value * value
        if first_gradient_diagnostic is None:
            first_gradient_diagnostic = gradient_diagnostics(model, losses, selector=selector)
        model.zero_grad(set_to_none=True)
        seen_batches += 1
        examples += len(interaction)
        if seen_batches >= batches:
            break
    if seen_batches == 0:
        raise RuntimeError("No train batches for normalization diagnostics.")
    mean_losses = {task: sums[task] / seen_batches for task in TASK_ORDER}
    std_losses = {}
    for task in TASK_ORDER:
        variance = max(sums_sq[task] / seen_batches - mean_losses[task] * mean_losses[task], 0.0)
        std_losses[task] = math.sqrt(variance)
    return {
        "batches": int(seen_batches),
        "examples": int(examples),
        "task_order": list(TASK_ORDER),
        "loss_scale_source": "train_diagnostic_mean",
        "mean_loss": mean_losses,
        "std_loss": std_losses,
        "loss_scales": [max(mean_losses[task], 1e-8) for task in TASK_ORDER],
        "gradient_selector": selector,
        "first_gradient_diagnostic": first_gradient_diagnostic,
        "test_access": "none",
    }


def source_ids(path: Path) -> set[int]:
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def create_full_split_loaders(config: Config, train_dataset: Any, valid_dataset: Any, test_dataset: Any) -> tuple[Any, Any, Any]:
    train_data = get_dataloader(config, "train")(config, train_dataset, None, shuffle=config["shuffle"])
    valid_data = get_dataloader(config, "valid")(config, valid_dataset, None, shuffle=False)
    test_data = get_dataloader(config, "test")(config, test_dataset, None, shuffle=False)
    if not isinstance(valid_data, FullSortEvalDataLoader):
        raise RuntimeError(f"Expected FullSortEvalDataLoader for valid, got {type(valid_data).__name__}")
    if not isinstance(test_data, FullSortEvalDataLoader):
        raise RuntimeError(f"Expected FullSortEvalDataLoader for test, got {type(test_data).__name__}")
    return train_data, valid_data, test_data


def ensure_locked_test_dataset() -> dict[str, Any]:
    if not LOCKED_TEST_SUMMARY.exists():
        prep_python = os.environ.get("EPO_MOE_PREP_PYTHON", sys.executable)
        script = ROOT / "experiments" / "multitask_tim4rec_optuna" / "prepare_locked_test_benchmark.py"
        subprocess.check_call([prep_python, str(script)], cwd=ROOT)
    summary = load_json(LOCKED_TEST_SUMMARY)
    if summary["identity_hash"] != EXPECTED_IDENTITY_HASH:
        raise RuntimeError(f"Locked test identity mismatch: {summary['identity_hash']}")
    return summary


def final_test_stage(args: argparse.Namespace, config: Mapping[str, Any], run_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if args.final_test_checkpoint_json is None or args.final_test_run_json is None:
        raise RuntimeError("--final-test-checkpoint-json and --final-test-run-json are required for final_test.")
    paths = resolve_paths(args, config, run_cfg)
    assert_output_allowed(paths, paths["artifact_dir"], args.allow_overwrite, args.resume)
    guard_path = paths["artifact_dir"] / "test_evaluation_guard.json"
    if guard_path.exists():
        guard = load_json(guard_path)
        raise RuntimeError(f"Test evaluation guard exists; refusing repeat: {guard}")
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    optuna_config = load_yaml(project_path(config["source"]["optuna_config"]))
    best_params = load_yaml(project_path(config["source"]["best_params"]))
    target_stats = load_target_stats(project_path(optuna_config["source"]["target_statistics"]))
    sampled = apply_sampled_overrides(sampled_from_locked_params(best_params, target_stats), run_cfg)
    locked_summary = ensure_locked_test_dataset()
    recbole_config = build_recbole_config(
        optuna_config,
        run_cfg,
        paths["artifact_dir"],
        sampled,
        int(config["training"]["max_epochs"]),
        model_class_for(run_cfg),
        validation_only=False,
        locked_test_summary=locked_summary,
    )
    init_seed(int(config["training"]["seed"]) + recbole_config["local_rank"], recbole_config["reproducibility"])
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required for final TEST evaluation.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dataset = create_dataset(recbole_config)
    built = dataset.build()
    if len(built) != 3:
        raise RuntimeError(f"Expected train/valid/test splits, got {len(built)}")
    train_dataset, valid_dataset, test_dataset = built
    if len(train_dataset) != EXPECTED_FINGERPRINT["train"] - EXPECTED_FINGERPRINT["users"]:
        raise RuntimeError(f"Train examples changed in locked dataset: {len(train_dataset)}")
    if len(valid_dataset) != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Validation examples changed in locked dataset: {len(valid_dataset)}")
    if len(test_dataset) != EXPECTED_FINGERPRINT["test"]:
        raise RuntimeError(f"Test examples changed in locked dataset: {len(test_dataset)}")
    train_data, valid_data, test_data = create_full_split_loaders(recbole_config, train_dataset, valid_dataset, test_dataset)
    item_num = int(test_data._dataset.item_num)
    validation_inspection = inspect_eval_loader(
        valid_data,
        item_num,
        source_ids(Path(locked_summary["validation_source_row_ids_path"])),
    )
    test_inspection = inspect_eval_loader(
        test_data,
        item_num,
        source_ids(Path(locked_summary["test_source_row_ids_path"])),
    )
    for split, inspection in (("validation", validation_inspection), ("test", test_inspection)):
        if not inspection["one_positive_per_row"] or not inspection["positive_targets_within_item_universe"]:
            raise RuntimeError(f"{split} loader inspection failed: {inspection}")

    preferences = load_preferences(project_path(config["source"]["preferences"]))
    pref_records = preference_records(preferences, config["epo"]["preference_set"])
    models, _optimizers, _solvers = instantiate_epo_models(
        config,
        recbole_config,
        train_data.dataset,
        sampled,
        pref_records,
        run_cfg,
    )
    checkpoint_ref = load_json_file(args.final_test_checkpoint_json)
    checkpoint_path = Path(checkpoint_ref["best_validation_checkpoint"]["path"])
    load_checkpoint(checkpoint_path, models, None)
    ranking_indices = [
        idx for idx, pref in enumerate(pref_records) if pref["id"] == RANKING_OPERATING_POINT_ID
    ]
    if len(ranking_indices) != 1:
        raise RuntimeError(f"Expected one frozen ranking operating point, got {ranking_indices}")
    ranking_index = ranking_indices[0]
    trainer = Trainer(recbole_config, models[ranking_index])
    save_json(
        guard_path,
        {
            "status": "started",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy": "single final TEST evaluation after frozen EPO+MoE configuration",
            "checkpoint": str(checkpoint_path),
            "preference_id": RANKING_OPERATING_POINT_ID,
            "solution_index": ranking_index,
            "test_evaluation_count": 0,
        },
    )
    records = []
    eval_started = time.monotonic()
    model = models[ranking_index]
    test_result, checks = evaluate_full_sort_with_checks(trainer, test_data, train_data)
    checks["evaluation"] = "test_full_7111_items"
    check_hit_recall_equal(test_result, list(METRIC_TOPK))
    if not checks["raw_scores_all_finite"] or not checks["positive_scores_all_finite"]:
        raise RuntimeError(f"Non-finite TEST scores: {checks}")
    aux = evaluate_auxiliary(model, test_data, next(model.parameters()).device)
    records.append(
        {
            "solution_index": ranking_index,
            "preference_id": pref_records[ranking_index]["id"],
            "preference": pref_records[ranking_index]["weights"],
            "metrics": normalize_metrics(metric_subset(test_result)),
            "auxiliary_test": aux,
            "checks": checks,
        }
    )
    selection = validation_summary_from_records(
        [
            {
                "solution_index": record["solution_index"],
                "preference_id": record["preference_id"],
                "preference": record["preference"],
                "metrics": record["metrics"],
                "auxiliary_validation": record["auxiliary_test"],
                "checks": record["checks"],
            }
            for record in records
        ],
        method="epo",
        reference_point=evaluation_reference(config)["values"],
        ranking_preference_id=RANKING_OPERATING_POINT_ID,
    )
    final_metrics = selection["ranking_operating_point"]["metrics"]
    guard = load_json(guard_path)
    guard.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_evaluation_count": len(records),
            "ranking_operating_point_metrics": final_metrics,
            "records": records,
            "runtime_sec": float(time.monotonic() - eval_started),
        }
    )
    save_json(guard_path, guard)

    validation_run = load_json_file(args.final_test_run_json)
    result = {
        "run_id": args.run_id or f"{run_cfg['run_id']}_final_test",
        "run_key": run_cfg["run_key"],
        "status": "completed",
        "record_type": "epo_moe_final_test",
        "stage": "final_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "starting_sha": config["starting_commit"],
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": git_value(["config", "--get", "remote.origin.url"]),
        },
        "source_files": source_hashes(),
        "environment": environment_info(),
        "slurm": slurm_info(),
        "gpu": gpu_info(),
        "architecture": model_architecture_record(models[ranking_index]),
        "validation_run": {
            "run_id": validation_run["run_id"],
            "best_epoch": validation_run["best_epoch"],
            "validation": validation_run["validation"],
            "checkpoint": checkpoint_ref["best_validation_checkpoint"],
            "git_commit": validation_run["git"]["commit"],
        },
        "frozen_operating_point": {
            "preference_id": RANKING_OPERATING_POINT_ID,
            "solution_index": ranking_index,
            "selection_rule": f"predefined_preference_id:{RANKING_OPERATING_POINT_ID}",
        },
        "locked_test_dataset": locked_summary,
        "test": selection,
        "final_test_metrics": final_metrics,
        "test_records": records,
        "test_evaluation_count": len(records),
        "test_dataset_loaded": True,
        "test_dataloader_created": True,
        "test_used_for_tuning": False,
        "tune_after_test_allowed": False,
        "guard_path": str(guard_path),
        "model_parameters": parameter_summary(models),
        "runtime": {
            "total_sec": float(time.monotonic() - start_time),
            "test_eval_sec": float(guard["runtime_sec"]),
            "process_ru_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }
    save_json(paths["result_json"], result)
    save_notes(paths["notes"], result)
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    run_cfg = run_config(config, args.run_key)
    if args.run_id:
        run_cfg = {**run_cfg, "run_id": args.run_id}
    if args.stage in {"sanity", "validation"}:
        result = validation_stage(args, config, run_cfg)
    else:
        result = final_test_stage(args, config, run_cfg)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"], "result_json": args.result_json}, indent=2), flush=True)


if __name__ == "__main__":
    main()
