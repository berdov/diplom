#!/usr/bin/env python
"""Short full-data TiM4Rec training sanity run.

Runs a small number of epochs on the full KuaiRand Protocol B RecBole dataset.
It does not evaluate test data and does not aim to reproduce the paper result.
"""

from __future__ import annotations

import argparse
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


RUN_ID = "tim4rec_sanity_001"
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
FULL_RANKING_BASELINES = {
    "random_full_ranking": {"hr@10": 0.001127301574, "ndcg@10": 0.000441685851},
    "mostpopular_full_ranking": {"hr@10": 0.029977871488, "ndcg@10": 0.016763898156},
    "xgboost_full_ranking": {"hr@10": 0.030854661601, "ndcg@10": 0.014971581041},
    "tim4rec_paper": {"hr@10": 0.1109, "ndcg@10": 0.0611},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "experiments" / "tim4rec_baseline" / "config_kuairand.yaml"),
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--artifact-dir",
        default="/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/tim4rec_sanity_001",
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
    result = []
    for name, param in model.named_parameters():
        if param.requires_grad and "time" in name.lower():
            result.append((name, param))
    return result


def param_norm(param: torch.nn.Parameter) -> float:
    return float(param.detach().float().norm().cpu().item())


def grad_norm(param: torch.nn.Parameter) -> float | None:
    if param.grad is None:
        return None
    return float(param.grad.detach().float().norm().cpu().item())


def finite_or_none(value: float | None) -> bool | None:
    if value is None:
        return None
    return math.isfinite(value)


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
    valid_data: FullSortEvalDataLoader,
    train_data,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not isinstance(valid_data, FullSortEvalDataLoader):
        raise RuntimeError(f"Expected FullSortEvalDataLoader, got {type(valid_data).__name__}")

    model = trainer.model
    model.eval()
    collector = Collector(trainer.config)
    collector.data_collect(train_data)
    evaluator = Evaluator(trainer.config)

    tot_item_num = int(valid_data._dataset.item_num)
    raw_nan_scores = 0
    raw_inf_scores = 0
    raw_score_batches = 0
    positive_score_nonfinite = 0
    rows = 0
    positives = 0

    for batched_data in valid_data:
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

    struct = collector.get_data_struct()
    result = evaluator.evaluate(struct)
    checks = {
        "loader_type": type(valid_data).__name__,
        "mode": "full",
        "item_num_with_padding": tot_item_num,
        "candidate_universe_size": tot_item_num - 1,
        "validation_rows": rows,
        "positive_targets": positives,
        "raw_score_batches": raw_score_batches,
        "raw_nan_scores": raw_nan_scores,
        "raw_inf_scores": raw_inf_scores,
        "positive_score_nonfinite": positive_score_nonfinite,
        "raw_scores_all_finite": raw_nan_scores == 0 and raw_inf_scores == 0,
        "positive_scores_all_finite": positive_score_nonfinite == 0,
    }
    return dict(result), checks


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


def train_one_epoch(
    trainer: Trainer,
    train_data,
    epoch_idx: int,
    collect_time_diag: bool,
) -> tuple[float, dict[str, Any] | None]:
    model = trainer.model
    model.train()
    optimizer = trainer.optimizer
    device = trainer.device
    total_loss = 0.0
    batches = 0
    time_diag = None

    for batch_idx, interaction in enumerate(train_data):
        interaction = interaction.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = model.calculate_loss(interaction)
        loss = sum(losses) if isinstance(losses, tuple) else losses
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch={epoch_idx}, batch={batch_idx}: {loss.item()}")

        selected_before = {}
        selected_names = []
        if collect_time_diag and batch_idx == 0:
            time_params = time_param_items(model)
            selected = time_params[:8]
            selected_names = [name for name, _param in selected]
            selected_before = {name: param.detach().clone() for name, param in selected}
            timestamp_seq = interaction["timestamp_list"]
            fork_devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
                time_diff = model.calculate_time_diff(timestamp_seq)
            time_diag = {
                "time_diff_present": time_diff is not None,
                "time_diff_shape": list(time_diff.shape),
                "time_diff_all_finite": bool(torch.isfinite(time_diff).all().item()),
                "trainable_time_parameter_count": int(sum(param.numel() for _name, param in time_params)),
                "trainable_time_tensor_count": len(time_params),
                "sampled_tensors": selected_names,
            }

        loss.backward()

        if collect_time_diag and batch_idx == 0 and time_diag is not None:
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
            time_diag["gradients"] = grad_checks
            time_diag["all_sampled_gradients_finite"] = all(
                item["grad_norm_finite"] is True for item in grad_checks
            )

        if trainer.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), **trainer.clip_grad_norm)
        optimizer.step()

        if collect_time_diag and batch_idx == 0 and time_diag is not None:
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
            time_diag["optimizer_updates"] = updates
            time_diag["all_sampled_updates_finite"] = all(item["update_norm_finite"] for item in updates)
            time_diag["any_sampled_time_parameter_updated"] = any(item["updated"] for item in updates)

        total_loss += float(loss.detach().cpu().item())
        batches += 1

    return total_loss / max(batches, 1), time_diag


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise RuntimeError(f"This sanity script is pinned to run_id={RUN_ID}, got {args.run_id}")

    artifact_dir = Path(args.artifact_dir)
    checkpoint_dir = artifact_dir / "checkpoints"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_info = load_manifest(args.manifest)
    fingerprint = manifest_fingerprint(manifest_info)
    assert_expected_fingerprint(fingerprint)

    config_overrides = {
        "epochs": int(args.epochs),
        "metrics": ["Hit", "Recall", "NDCG"],
        "topk": [5, 10, 20, 50],
        "checkpoint_dir": str(checkpoint_dir),
        "show_progress": False,
        "log_wandb": False,
    }
    config = Config(model=TiM4Rec, config_file_list=[args.config], config_dict=config_overrides)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for TiM4Rec sanity training.")
    if not bool(config["is_time"]):
        raise RuntimeError("Sanity training must run full TiM4Rec with is_time=True.")

    start_monotonic = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dataset = create_dataset(config)
    train_data, valid_data, _test_data = data_preparation(config, dataset)
    model = TiM4Rec(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    eval_loader_inspection = inspect_eval_loader(valid_data, int(valid_data._dataset.item_num))
    if not eval_loader_inspection["one_positive_per_row"]:
        raise RuntimeError(f"Validation must have one positive per row: {eval_loader_inspection}")
    if not eval_loader_inspection["positive_targets_within_item_universe"]:
        raise RuntimeError(f"Validation positives outside item universe: {eval_loader_inspection}")

    best_valid_score = -float("inf")
    best_epoch = None
    cur_step = 0
    epoch_results = []
    time_aware_diag = None
    warnings = []
    best_checkpoint = None
    last_checkpoint = None
    topk = list(config["topk"])
    valid_metric = str(config["valid_metric"]).lower()
    valid_metric_bigger = bool(config["valid_metric_bigger"])
    result_path = Path(args.result_json)
    partial_result_path = result_path.with_suffix(".partial.json")

    if int(valid_data._dataset.item_num) - 1 != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(
            f"Full-ranking item universe mismatch: {int(valid_data._dataset.item_num) - 1}"
        )
    if int(eval_loader_inspection["rows"]) != EXPECTED_FINGERPRINT["validation"]:
        raise RuntimeError(f"Validation user count mismatch: {eval_loader_inspection['rows']}")

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.monotonic()
        train_start = time.monotonic()
        train_loss, diag = train_one_epoch(trainer, train_data, epoch, collect_time_diag=(epoch == 1))
        train_time = time.monotonic() - train_start
        if diag is not None:
            time_aware_diag = diag

        valid_start = time.monotonic()
        valid_result, full_ranking_checks = evaluate_full_sort_with_checks(trainer, valid_data, train_data)
        validation_time = time.monotonic() - valid_start
        hit_recall_check = check_hit_recall_equal(valid_result, topk)

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

        if not full_ranking_checks["raw_scores_all_finite"]:
            raise RuntimeError(f"Non-finite raw validation scores: {full_ranking_checks}")

        peak_alloc = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        peak_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
        epoch_results.append(
            {
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
                "gpu_peak_allocated_bytes_so_far": peak_alloc,
                "gpu_peak_reserved_bytes_so_far": peak_reserved,
            }
        )
        partial_result_path.parent.mkdir(parents=True, exist_ok=True)
        partial_result_path.write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "partial",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "epochs_completed": len(epoch_results),
                    "latest_epoch": epoch_results[-1],
                    "best_epoch_so_far": best_epoch,
                    "best_valid_score_so_far": float(best_valid_score),
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
                    "train_time_sec": train_time,
                    "validation_time_sec": validation_time,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stop_flag:
            warnings.append(f"early stopping triggered at epoch {epoch}")
            break

    runtime_sec = time.monotonic() - start_monotonic
    ru_maxrss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result = {
        "run_id": args.run_id,
        "status": "ok",
        "sanity": True,
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
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "recbole": version("recbole"),
            "mamba_ssm": version("mamba-ssm"),
            "causal_conv1d": version("causal-conv1d"),
            "transformers": version("transformers"),
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
                "validation_loader": eval_loader_inspection,
            },
            "split": {
                "name": "Protocol B chronological leave-one-out",
                "test_evaluated": False,
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
            "epochs_completed": len(epoch_results),
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "metrics": list(config["metrics"]),
            "topk": topk,
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
        "time_aware_diagnostics": time_aware_diag,
        "epochs": epoch_results,
        "best_epoch": best_epoch,
        "best_valid_score": float(best_valid_score),
        "best_valid_metric": valid_metric,
        "checkpoints": {
            "best_validation": best_checkpoint,
            "last": last_checkpoint,
        },
        "remote_artifact_path": str(artifact_dir),
        "runtime": {
            "total_sec": runtime_sec,
            "mean_epoch_sec": sum(item["epoch_time_sec"] for item in epoch_results) / max(len(epoch_results), 1),
        },
        "comparison_references": {
            "full_ranking_baselines_validation": FULL_RANKING_BASELINES,
            "paper_tim4rec": PAPER_REFERENCE,
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
