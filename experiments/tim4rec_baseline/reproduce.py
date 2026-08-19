#!/usr/bin/env python
"""Full TiM4Rec reproduction run on KuaiRand Protocol B.

This script performs model selection only on validation NDCG@10. It evaluates
the test split exactly once after loading the best validation checkpoint.
"""

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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Collector, Evaluator
from recbole.trainer import Trainer
from recbole.utils import early_stopping, init_seed

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tim4rec import TiM4Rec  # noqa: E402


RUN_ID = "tim4rec_001"
EXPECTED_FINGERPRINT = {
    "users": 23951,
    "items": 7111,
    "interactions": 1134420,
    "train": 1086518,
    "validation": 23951,
    "test": 23951,
}
PAPER_REFERENCE = {
    "recall@10": 0.1109,
    "recall@20": 0.1774,
    "recall@50": 0.3202,
    "ndcg@10": 0.0611,
    "ndcg@20": 0.0779,
    "ndcg@50": 0.1060,
}
EXPECTED_CONFIG = {
    "is_time": True,
    "learning_rate": 0.001,
    "seed": 2026,
    "train_batch_size": 2048,
    "eval_batch_size": 4096,
    "MAX_ITEM_LIST_LENGTH": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "experiments" / "tim4rec_baseline" / "config_kuairand.yaml"),
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--artifact-dir",
        default="/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/tim4rec_001",
    )
    parser.add_argument(
        "--result-json",
        default=str(ROOT / "experiments" / "tim4rec_baseline" / "runs" / f"{RUN_ID}.json"),
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


def load_manifest(path: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {"path": str(manifest_path), "loaded": False}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"path": str(manifest_path), "loaded": True, "payload": payload}


def manifest_fingerprint(manifest_info: dict[str, Any]) -> dict[str, int] | None:
    if not manifest_info.get("loaded"):
        return None
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


def assert_expected_fingerprint(fingerprint: dict[str, int] | None) -> None:
    if fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError(f"Protocol B fingerprint mismatch: {fingerprint} != {EXPECTED_FINGERPRINT}")


def time_param_items(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and "time" in name.lower()
    ]


def finite_or_none(value: float | None) -> bool | None:
    if value is None:
        return None
    return math.isfinite(value)


def grad_norm(param: torch.nn.Parameter) -> float | None:
    if param.grad is None:
        return None
    return float(param.grad.detach().float().norm().cpu().item())


def param_norm(param: torch.nn.Parameter) -> float:
    return float(param.detach().float().norm().cpu().item())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    model: TiM4Rec,
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
    return {"path": str(path), "size_bytes": path.stat().st_size}


def load_model_checkpoint(model: TiM4Rec, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    other_parameter = checkpoint.get("other_parameter")
    if other_parameter is not None and hasattr(model, "load_other_parameter"):
        model.load_other_parameter(other_parameter)
    return checkpoint


def inspect_eval_loader(eval_data: FullSortEvalDataLoader, item_num_with_pad: int) -> dict[str, Any]:
    positives = 0
    rows = 0
    min_positive = None
    max_positive = None
    zero_positive = 0
    for batched_data in eval_data:
        interaction, _history_index, _positive_u, positive_i = batched_data
        rows += len(interaction)
        positives += len(positive_i)
        if len(positive_i):
            min_i = int(positive_i.min().item())
            max_i = int(positive_i.max().item())
            min_positive = min_i if min_positive is None else min(min_positive, min_i)
            max_positive = max_i if max_positive is None else max(max_positive, max_i)
            zero_positive += int((positive_i == 0).sum().item())
    return {
        "rows": rows,
        "positive_targets": positives,
        "one_positive_per_row": positives == rows,
        "min_positive_item_id": min_positive,
        "max_positive_item_id": max_positive,
        "zero_positive_targets": zero_positive,
        "positive_targets_within_item_universe": (
            min_positive is not None and min_positive > 0 and max_positive < item_num_with_pad
        ),
    }


@torch.no_grad()
def evaluate_full_sort_with_checks(
    trainer: Trainer,
    eval_data: FullSortEvalDataLoader,
    train_data,
    split_name: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not isinstance(eval_data, FullSortEvalDataLoader):
        raise RuntimeError(f"Expected FullSortEvalDataLoader, got {type(eval_data).__name__}")

    model = trainer.model
    model.eval()
    collector = Collector(trainer.config)
    collector.data_collect(train_data)
    evaluator = Evaluator(trainer.config)

    tot_item_num = int(eval_data._dataset.item_num)
    raw_nan_scores = 0
    raw_inf_scores = 0
    raw_score_batches = 0
    positive_score_nonfinite = 0
    rows = 0
    positives = 0

    for batched_data in eval_data:
        interaction, history_index, positive_u, positive_i = batched_data
        interaction = interaction.to(trainer.device)
        scores = model.full_sort_predict(interaction).view(-1, tot_item_num)
        raw_score_batches += 1
        raw_nan_scores += int(torch.isnan(scores).sum().item())
        raw_inf_scores += int(torch.isinf(scores).sum().item())
        pos_scores = scores[positive_u.to(scores.device), positive_i.to(scores.device)]
        positive_score_nonfinite += int((~torch.isfinite(pos_scores)).sum().item())

        scores[:, 0] = -float("inf")
        if history_index is not None:
            scores[history_index] = -float("inf")
        collector.eval_batch_collect(scores, interaction, positive_u, positive_i)

        rows += len(interaction)
        positives += len(positive_i)

    result = dict(Evaluator(trainer.config).evaluate(collector.get_data_struct()))
    checks = {
        "split": split_name,
        "loader_type": type(eval_data).__name__,
        "mode": "full",
        "protocol": "B",
        "evaluation": "full_7111_items",
        "item_num_with_padding": tot_item_num,
        "candidate_universe_size": tot_item_num - 1,
        "rows": rows,
        "positive_targets": positives,
        "raw_score_batches": raw_score_batches,
        "raw_nan_scores": raw_nan_scores,
        "raw_inf_scores": raw_inf_scores,
        "positive_score_nonfinite": positive_score_nonfinite,
        "raw_scores_all_finite": raw_nan_scores == 0 and raw_inf_scores == 0,
        "positive_scores_all_finite": positive_score_nonfinite == 0,
    }
    return result, checks


def check_hit_recall_equal(result: dict[str, float], topk: list[int], split_name: str) -> dict[str, Any]:
    diffs = {}
    ok = True
    for k in topk:
        hit = float(result[f"hit@{k}"])
        recall = float(result[f"recall@{k}"])
        diff = abs(hit - recall)
        diffs[str(k)] = diff
        ok = ok and diff <= 1e-12
    if not ok:
        raise RuntimeError(f"Hit/Recall mismatch in one-positive {split_name}: {diffs}")
    return {"ok": ok, "abs_diffs": diffs}


def all_gradient_check(model: torch.nn.Module) -> dict[str, Any]:
    tensors = 0
    nonfinite_tensors = []
    max_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        tensors += 1
        grad = param.grad.detach()
        if not torch.isfinite(grad).all().item():
            nonfinite_tensors.append(name)
        norm = float(grad.float().norm().cpu().item())
        if math.isfinite(norm):
            max_norm = max(max_norm, norm)
    return {
        "checked_tensors": tensors,
        "nonfinite_tensor_count": len(nonfinite_tensors),
        "nonfinite_tensors_sample": nonfinite_tensors[:10],
        "all_finite": len(nonfinite_tensors) == 0,
        "max_finite_grad_norm": max_norm,
    }


def train_one_epoch(trainer: Trainer, train_data, epoch_idx: int) -> tuple[float, dict[str, Any]]:
    model = trainer.model
    model.train()
    optimizer = trainer.optimizer
    device = trainer.device
    total_loss = 0.0
    batches = 0
    diagnostics: dict[str, Any] | None = None

    for batch_idx, interaction in enumerate(train_data):
        interaction = interaction.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = model.calculate_loss(interaction)
        loss = sum(losses) if isinstance(losses, tuple) else losses
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch={epoch_idx}, batch={batch_idx}: {loss.item()}")

        selected_before = {}
        selected_names = []
        if batch_idx == 0:
            time_params = time_param_items(model)
            selected = time_params[:8]
            selected_names = [name for name, _param in selected]
            selected_before = {name: param.detach().clone() for name, param in selected}
            timestamp_seq = interaction["timestamp_list"]
            fork_devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
                time_diff = model.calculate_time_diff(timestamp_seq)
            diagnostics = {
                "epoch": epoch_idx,
                "time_diff_present": time_diff is not None,
                "time_diff_shape": list(time_diff.shape),
                "time_diff_all_finite": bool(torch.isfinite(time_diff).all().item()),
                "trainable_time_parameter_count": int(sum(param.numel() for _name, param in time_params)),
                "trainable_time_tensor_count": len(time_params),
                "sampled_tensors": selected_names,
            }

        loss.backward()

        if batch_idx == 0 and diagnostics is not None:
            grad_checks = []
            for name, param in time_param_items(model)[:8]:
                norm = grad_norm(param)
                grad_checks.append(
                    {
                        "name": name,
                        "grad_norm": norm,
                        "grad_norm_finite": finite_or_none(norm),
                    }
                )
            diagnostics["sampled_time_gradients"] = grad_checks
            diagnostics["all_sampled_time_gradients_finite"] = all(
                item["grad_norm_finite"] is True for item in grad_checks
            )
            diagnostics["all_gradients"] = all_gradient_check(model)
            if not diagnostics["all_gradients"]["all_finite"]:
                raise RuntimeError(f"Non-finite gradients at epoch={epoch_idx}: {diagnostics['all_gradients']}")

        if trainer.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), **trainer.clip_grad_norm)
        optimizer.step()

        if batch_idx == 0 and diagnostics is not None:
            updates = []
            named_params = dict(model.named_parameters())
            for name in selected_names:
                before = selected_before[name]
                after = named_params[name].detach()
                update_norm = float((after - before).float().norm().cpu().item())
                updates.append(
                    {
                        "name": name,
                        "param_norm_after": param_norm(named_params[name]),
                        "update_norm": update_norm,
                        "updated": update_norm > 0.0,
                        "update_norm_finite": math.isfinite(update_norm),
                    }
                )
            diagnostics["sampled_time_optimizer_updates"] = updates
            diagnostics["all_sampled_time_updates_finite"] = all(
                item["update_norm_finite"] for item in updates
            )
            diagnostics["any_sampled_time_parameter_updated"] = any(item["updated"] for item in updates)

        total_loss += float(loss.detach().cpu().item())
        batches += 1

    if diagnostics is None:
        raise RuntimeError(f"No training batches in epoch={epoch_idx}")
    return total_loss / max(batches, 1), diagnostics


def assert_config(config: Config) -> None:
    for key, expected in EXPECTED_CONFIG.items():
        actual = config[key]
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise RuntimeError(f"Unexpected config {key}: {actual} != {expected}")
        else:
            if actual != expected:
                raise RuntimeError(f"Unexpected config {key}: {actual} != {expected}")
    if str(config["valid_metric"]).lower() != "ndcg@10":
        raise RuntimeError(f"Unexpected valid_metric: {config['valid_metric']}")
    if int(config["stopping_step"]) != 10:
        raise RuntimeError(f"Unexpected stopping_step: {config['stopping_step']}")
    if set(config["metrics"]) != {"Hit", "Recall", "NDCG"}:
        raise RuntimeError(f"Unexpected metrics: {config['metrics']}")
    if list(config["topk"]) != [5, 10, 20, 50]:
        raise RuntimeError(f"Unexpected topk: {config['topk']}")


def paper_differences(test_result: dict[str, float]) -> dict[str, dict[str, float]]:
    differences = {}
    for metric, paper_value in PAPER_REFERENCE.items():
        ours = float(test_result[metric])
        abs_diff = ours - paper_value
        differences[metric] = {
            "ours": ours,
            "paper": paper_value,
            "absolute_difference": abs_diff,
            "relative_difference_percent": abs_diff / paper_value * 100.0,
        }
    return differences


def compact_epoch(epoch_result: dict[str, Any]) -> dict[str, Any]:
    validation = epoch_result["validation"]
    return {
        "epoch": epoch_result["epoch"],
        "train_loss": epoch_result["train_loss"],
        "learning_rate": epoch_result["learning_rate"],
        "train_time_sec": epoch_result["train_time_sec"],
        "validation_time_sec": epoch_result["validation_time_sec"],
        "epoch_time_sec": epoch_result["epoch_time_sec"],
        "validation": validation,
        "valid_score": epoch_result["valid_score"],
        "early_stopping": epoch_result["early_stopping"],
        "gpu_peak_allocated_bytes_so_far": epoch_result["gpu_peak_allocated_bytes_so_far"],
        "gpu_peak_reserved_bytes_so_far": epoch_result["gpu_peak_reserved_bytes_so_far"],
    }


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise RuntimeError(f"This reproduction script is pinned to run_id={RUN_ID}, got {args.run_id}")

    result_path = Path(args.result_json)
    partial_result_path = result_path.with_suffix(".partial.json")
    artifact_dir = Path(args.artifact_dir)
    checkpoint_dir = artifact_dir / "checkpoints"
    training_log_path = artifact_dir / "training_log.jsonl"
    environment_path = artifact_dir / "environment.json"

    if result_path.exists() or partial_result_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing run JSON: {result_path}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty artifact dir: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_info = load_manifest(args.manifest)
    fingerprint = manifest_fingerprint(manifest_info)
    assert_expected_fingerprint(fingerprint)

    config_overrides = {
        "epochs": int(args.epochs),
        "metrics": ["Hit", "Recall", "NDCG"],
        "topk": [5, 10, 20, 50],
        "valid_metric": "NDCG@10",
        "stopping_step": 10,
        "checkpoint_dir": str(checkpoint_dir),
        "show_progress": False,
        "log_wandb": False,
    }
    config = Config(model=TiM4Rec, config_file_list=[args.config], config_dict=config_overrides)
    assert_config(config)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for TiM4Rec reproduction.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_monotonic = time.monotonic()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = TiM4Rec(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    if int(dataset.item_num) - 1 != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Full-ranking item universe mismatch: {int(dataset.item_num) - 1}")
    eval_loader_inspection = inspect_eval_loader(valid_data, int(valid_data._dataset.item_num))
    if not eval_loader_inspection["one_positive_per_row"]:
        raise RuntimeError(f"Validation must have one positive per row: {eval_loader_inspection}")
    if not eval_loader_inspection["positive_targets_within_item_universe"]:
        raise RuntimeError(f"Validation positives outside item universe: {eval_loader_inspection}")
    if int(eval_loader_inspection["rows"]) != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Validation user count mismatch: {eval_loader_inspection['rows']}")

    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "recbole": version("recbole"),
        "mamba_ssm": version("mamba-ssm"),
        "causal_conv1d": version("causal-conv1d"),
        "transformers": version("transformers"),
    }
    environment_path.write_text(json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    best_valid_score = -float("inf")
    best_epoch = None
    cur_step = 0
    epoch_results = []
    time_aware_diagnostics = []
    warnings = []
    best_checkpoint = None
    last_checkpoint = None
    stop_reason = "max_epochs"
    topk = list(config["topk"])
    valid_metric = str(config["valid_metric"]).lower()
    valid_metric_bigger = bool(config["valid_metric_bigger"])

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.monotonic()
        train_start = time.monotonic()
        train_loss, diag = train_one_epoch(trainer, train_data, epoch)
        train_time = time.monotonic() - train_start
        time_aware_diagnostics.append(diag)

        valid_start = time.monotonic()
        valid_result, full_ranking_checks = evaluate_full_sort_with_checks(
            trainer, valid_data, train_data, "validation"
        )
        validation_time = time.monotonic() - valid_start
        hit_recall_check = check_hit_recall_equal(valid_result, topk, "validation")
        if not full_ranking_checks["raw_scores_all_finite"]:
            raise RuntimeError(f"Non-finite raw validation scores: {full_ranking_checks}")

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

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
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
            "hit_recall_equal_check": hit_recall_check,
            "full_ranking_checks": full_ranking_checks,
            "time_aware_diagnostics": diag,
            "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()),
        }
        epoch_results.append(epoch_result)

        with training_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_result, ensure_ascii=False, default=json_default) + "\n")

        partial_result_path.parent.mkdir(parents=True, exist_ok=True)
        partial_result_path.write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "partial",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "epochs_completed": len(epoch_results),
                    "latest_epoch": compact_epoch(epoch_result),
                    "best_epoch_so_far": best_epoch,
                    "best_valid_score_so_far": float(best_valid_score),
                    "early_stopping_cur_step": int(cur_step),
                    "remote_artifact_path": str(artifact_dir),
                },
                ensure_ascii=False,
                indent=2,
                default=json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid": valid_result,
                    "best_epoch": best_epoch,
                    "best_valid_score": best_valid_score,
                    "cur_step": cur_step,
                    "train_time_sec": train_time,
                    "validation_time_sec": validation_time,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stop_flag:
            stop_reason = f"early_stopping_no_improvement_{int(config['stopping_step'])}"
            warnings.append(f"early stopping triggered at epoch {epoch}")
            break

    if best_checkpoint is None or best_epoch is None:
        raise RuntimeError("No best validation checkpoint was saved.")

    best_checkpoint_path = Path(best_checkpoint["path"])
    best_checkpoint["sha256"] = sha256_file(best_checkpoint_path)
    if last_checkpoint is not None:
        last_checkpoint["sha256"] = sha256_file(Path(last_checkpoint["path"]))

    checkpoint_payload = load_model_checkpoint(model, best_checkpoint_path, config["device"])
    test_start = time.monotonic()
    test_result, test_full_ranking_checks = evaluate_full_sort_with_checks(
        trainer, test_data, train_data, "test"
    )
    test_time = time.monotonic() - test_start
    test_hit_recall_check = check_hit_recall_equal(test_result, topk, "test")
    if not test_full_ranking_checks["raw_scores_all_finite"]:
        raise RuntimeError(f"Non-finite raw test scores: {test_full_ranking_checks}")
    if int(test_full_ranking_checks["rows"]) != EXPECTED_FINGERPRINT["test"]:
        raise RuntimeError(f"Test user count mismatch: {test_full_ranking_checks['rows']}")

    paper_diff = paper_differences(test_result)
    runtime_sec = time.monotonic() - start_monotonic
    ru_maxrss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    compact_epochs = [compact_epoch(item) for item in epoch_results]
    best_validation_metrics = epoch_results[int(best_epoch) - 1]["validation"]
    result = {
        "run_id": args.run_id,
        "status": "completed",
        "sanity": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": "8d4a6cea6a035c249a7a13999166ba41e8924abe",
        "project_git_commit": os.environ.get("TIM4REC_GIT_COMMIT", "unknown"),
        "branch": os.environ.get("TIM4REC_GIT_BRANCH", "unknown"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
        },
        "environment": environment,
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
            "manifest": {"path": manifest_info["path"], "loaded": bool(manifest_info.get("loaded"))},
            "fingerprint": fingerprint,
            "expected_fingerprint": EXPECTED_FINGERPRINT,
            "recbole": {
                "user_num_with_padding": int(dataset.user_num),
                "item_num_with_padding": int(dataset.item_num),
                "item_universe_without_padding": int(dataset.item_num) - 1,
                "inter_num_after_sequential_augmentation": int(dataset.inter_num),
                "train_batches": len(train_data),
                "valid_batches": len(valid_data),
                "test_batches": len(test_data),
                "validation_loader": eval_loader_inspection,
            },
            "split": {
                "name": "Protocol B chronological leave-one-out",
                "protocol": "B",
                "evaluation": "full_7111_items",
                "test_evaluated": True,
                "test_evaluations_count": 1,
                "test_evaluation_timing": "after_loading_best_validation_checkpoint",
            },
        },
        "config": {
            "config_file": str(Path(args.config).resolve()),
            "seed": int(config["seed"]),
            "recbole_default_seed": 2020,
            "paper_seed": None,
            "upstream_config_seed": None,
            "seed_note": "Paper/upstream KuaiRand config does not specify seed; using reproducible project seed from config.",
            "is_time": bool(config["is_time"]),
            "learning_rate": float(config["learning_rate"]),
            "lr_source": "upstream_config",
            "paper_learning_rate": 0.01,
            "epochs_requested": int(args.epochs),
            "epochs_actual": len(epoch_results),
            "stopping_step": int(config["stopping_step"]),
            "stop_reason": stop_reason,
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "MAX_ITEM_LIST_LENGTH": int(config["MAX_ITEM_LIST_LENGTH"]),
            "metrics": list(config["metrics"]),
            "topk": topk,
            "valid_metric": str(config["valid_metric"]),
            "eval_args": config["eval_args"],
            "architecture": {
                "hidden_size": int(config["hidden_size"]),
                "num_layers": int(config["num_layers"]),
                "dropout_prob": float(config["dropout_prob"]),
                "time_drop_out": float(config["time_drop_out"]),
                "d_state": int(config["d_state"]),
                "d_conv": int(config["d_conv"]),
                "expand": int(config["expand"]),
                "head_dim": int(config["head_dim"]),
                "chunk_size": int(config["chunk_size"]),
                "norm_eps": float(config["norm_eps"]),
                "is_ffn": bool(config["is_ffn"]),
                "p2p_residual": bool(config["p2p_residual"]),
            },
        },
        "epochs": compact_epochs,
        "time_aware_diagnostics": {
            "first_epoch": time_aware_diagnostics[0],
            "last_epoch": time_aware_diagnostics[-1],
            "all_epochs_time_diff_finite": all(item["time_diff_all_finite"] for item in time_aware_diagnostics),
            "all_epochs_gradients_finite": all(
                item["all_gradients"]["all_finite"] for item in time_aware_diagnostics
            ),
            "all_epochs_sampled_time_updates_finite": all(
                item["all_sampled_time_updates_finite"] for item in time_aware_diagnostics
            ),
            "all_epochs_sampled_time_parameter_updated": all(
                item["any_sampled_time_parameter_updated"] for item in time_aware_diagnostics
            ),
        },
        "best_epoch": best_epoch,
        "best_valid_score": float(best_valid_score),
        "best_valid_metric": valid_metric,
        "best_validation_metrics": best_validation_metrics,
        "best_checkpoint_loaded_for_test": {
            "path": str(best_checkpoint_path),
            "checkpoint_epoch": int(checkpoint_payload["epoch"]),
            "checkpoint_best_valid_score": float(checkpoint_payload["best_valid_score"]),
        },
        "final_test_metrics": test_result,
        "test_time_sec": test_time,
        "test_hit_recall_equal_check": test_hit_recall_check,
        "test_full_ranking_checks": test_full_ranking_checks,
        "published_metrics": {
            "tim4rec_paper": PAPER_REFERENCE,
        },
        "paper_differences": paper_diff,
        "checkpoints": {
            "best_validation": best_checkpoint,
            "last": last_checkpoint,
        },
        "remote_artifact_path": str(artifact_dir),
        "remote_training_log_path": str(training_log_path),
        "remote_environment_path": str(environment_path),
        "runtime": {
            "total_sec": runtime_sec,
            "mean_epoch_sec": sum(item["epoch_time_sec"] for item in epoch_results) / max(len(epoch_results), 1),
        },
        "warnings": warnings,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
