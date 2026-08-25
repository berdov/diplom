#!/usr/bin/env python
"""Full fixed-loss training for MultitaskTiM4Rec.

This script locks the loss policy selected by
`multitask_tim4rec_sanity_001`: lambda_aux=0.2 with train-only pos_weight.
It chooses the checkpoint only by validation NDCG@10 and evaluates test once
after loading that best validation checkpoint.
"""

from __future__ import annotations

import argparse
import csv
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
from recbole.trainer import Trainer
from recbole.utils import early_stopping, init_seed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tim4rec import TiM4Rec  # noqa: E402
from experiments.multitask_tim4rec.model import MultitaskTiM4Rec, TARGETS  # noqa: E402
from experiments.multitask_tim4rec.train import (  # noqa: E402
    DEFAULT_MULTITASK_DIR,
    EXPECTED_FINGERPRINT,
    EXPECTED_IDENTITY_HASH,
    all_gradient_check,
    check_hit_recall_equal,
    compact_epoch,
    count_parameters,
    evaluate_auxiliary,
    evaluate_full_sort_with_checks,
    first_batch,
    format_float,
    git_value,
    human_size,
    inspect_eval_loader,
    json_default,
    load_json,
    load_target_stats,
    pos_weight_tensors,
    run_smoke,
    save_checkpoint,
    sha256_file,
    train_one_epoch,
    version,
    assert_multitask_manifest,
    ensure_recbole_inter,
    expected_validation_source_ids,
    metric_subset,
)


RUN_ID = "multitask_tim4rec_001"
SANITY_RUN_ID = "multitask_tim4rec_sanity_001"
LOCKED_LAMBDA_AUX = 0.2
DEFAULT_MANIFEST = ROOT / "outputs" / "data" / "protocol_b_multitask_manifest.json"
DEFAULT_ARTIFACT_DIR = Path("/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec") / RUN_ID
DEFAULT_RESULTS_CSV = ROOT / "experiments" / "results.csv"
BASELINE_PATHS = {
    "MostPopular": ROOT / "experiments" / "ltr_xgb_baseline" / "runs" / "mostpop_002.json",
    "XGBoost LambdaMART": ROOT / "experiments" / "ltr_xgb_baseline" / "runs" / "ltr_xgb_002.json",
    "XGBoost LambdaMART tuned": ROOT / "experiments" / "ltr_xgb_optuna" / "runs" / "ltr_xgb_optuna_001.json",
    "SSD4Rec": ROOT / "experiments" / "ssd4rec_baseline" / "runs" / "ssd4rec_001.json",
    "TiM4Rec": ROOT / "experiments" / "tim4rec_baseline" / "runs" / "tim4rec_001.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments" / "multitask_tim4rec" / "config.yaml"))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--result-json", default=str(ROOT / "experiments" / "multitask_tim4rec" / "runs" / f"{RUN_ID}.json"))
    parser.add_argument("--notes", default=str(ROOT / "experiments" / "multitask_tim4rec" / "runs" / f"{RUN_ID}_notes.md"))
    parser.add_argument("--multitask-dir", default=str(DEFAULT_MULTITASK_DIR))
    parser.add_argument("--multitask-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--target-statistics", default=str(ROOT / "experiments" / "multitask_tim4rec" / "target_statistics.csv"))
    parser.add_argument("--sanity-run-json", default=str(ROOT / "experiments" / "multitask_tim4rec" / "runs" / f"{SANITY_RUN_ID}.json"))
    return parser.parse_args()


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


def lowercase_metrics(metrics: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in (5, 10, 20, 50):
        out[f"hit@{k}"] = float(metrics[f"HR@{k}"])
        out[f"recall@{k}"] = float(metrics[f"Recall@{k}"])
        out[f"ndcg@{k}"] = float(metrics[f"NDCG@{k}"])
    return out


def load_baseline_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline artifact: {path}")
    payload = load_json(path)
    if "final_test_metrics" in payload:
        test_metrics = normalize_metrics(payload["final_test_metrics"])
    elif "metrics" in payload and "test" in payload["metrics"]:
        test_metrics = normalize_metrics(payload["metrics"]["test"])
    else:
        raise KeyError(f"Cannot find test metrics in {path}")
    validation_metrics = None
    if "best_validation_metrics" in payload:
        validation_metrics = normalize_metrics(payload["best_validation_metrics"])
    elif "metrics" in payload and "validation" in payload["metrics"]:
        validation_metrics = normalize_metrics(payload["metrics"]["validation"])
    return {
        "path": str(path),
        "run_id": payload.get("run_id", path.stem),
        "status": payload.get("status"),
        "test": test_metrics,
        "validation": validation_metrics,
        "runtime": payload.get("runtime", {}),
        "gpu": payload.get("gpu", {}),
    }


def load_baseline_table() -> dict[str, dict[str, Any]]:
    return {name: load_baseline_metrics(path) for name, path in BASELINE_PATHS.items()}


def compare_against(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, dict[str, float | None]]:
    comparison = {}
    for metric in [f"HR@{k}" for k in (5, 10, 20, 50)] + [f"NDCG@{k}" for k in (5, 10, 20, 50)]:
        diff = candidate[metric] - reference[metric]
        rel = None if reference[metric] == 0 else diff / reference[metric] * 100.0
        comparison[metric] = {"reference": reference[metric], "candidate": candidate[metric], "absolute_diff": diff, "relative_diff_percent": rel}
    return comparison


def qualitative_delta(delta: float) -> str:
    if delta >= 0.001:
        return "improves"
    if abs(delta) < 0.001:
        return "roughly equal"
    if delta <= -0.01:
        return "clearly worse"
    return "slightly worse"


def checkpoint_sha(info: dict[str, Any] | None) -> dict[str, Any] | None:
    if info is None:
        return None
    path = Path(info["path"])
    info["sha256"] = sha256_file(path)
    return info


def format_table_metrics(metrics: dict[str, float]) -> list[str]:
    return [
        format_float(metrics["HR@10"]),
        format_float(metrics["HR@20"]),
        format_float(metrics["HR@50"]),
        format_float(metrics["NDCG@10"]),
        format_float(metrics["NDCG@20"]),
        format_float(metrics["NDCG@50"]),
    ]


def build_notes(result: dict[str, Any]) -> str:
    best = normalize_metrics(result["best_validation"]["validation"])
    test = result["final_test"]["recommendation_metrics"]
    aux_valid = result["best_validation"]["auxiliary_validation"]
    aux_test = result["final_test"]["auxiliary_metrics"]
    params = result["model_parameters"]
    nt = result["negative_transfer_analysis"]
    lines = [
        "# Multitask TiM4Rec 001",
        "",
        "## Цель",
        "",
        "Полный fixed-loss запуск первой собственной архитектуры `MultitaskTiM4Rec` на Protocol B.",
        "",
        "## Отличие от TiM4Rec",
        "",
        "Backbone полностью совпадает с `tim4rec_001`; добавлены только четыре linear behavior heads.",
        "",
        "## Данные",
        "",
        f"- Dataset: `{result['dataset']['multitask_dir']}`.",
        f"- Identity hash: `{result['dataset']['identity_hash']}`.",
        f"- Train/validation/test rows: {result['dataset']['fingerprint']['train']} / {result['dataset']['fingerprint']['validation']} / {result['dataset']['fingerprint']['test']}.",
        "",
        "## Архитектура",
        "",
        "- Shared representation строится только из `item_id_list`, `item_length`, `timestamp_list`.",
        "- Heads: `Linear(64, 1)` для каждого target.",
        "- MoE/adaptive loss/new attention/Flow Matching не использовались.",
        "",
        "## Targets",
        "",
        "- `is_click`, `long_view`, `is_like`, `is_profile_enter`.",
        "",
        "## Loss",
        "",
        f"- `{result['loss_formula']}`.",
        f"- `lambda_aux = {result['lambda_aux']}`.",
        f"- Loss config source: `{result['loss_config_source']}`.",
        f"- `pos_weight` locked from train: `{result['pos_weights']}`.",
        "",
        "## Обучение",
        "",
        f"- Requested epochs: {result['training_config']['epochs_requested']}.",
        f"- Actual epochs: {result['training_config']['epochs_completed']}.",
        f"- Stop reason: `{result['training_config']['stop_reason']}`.",
        "",
        "## Early stopping",
        "",
        f"- Criterion: `{result['best_valid_metric']}` maximize.",
        f"- Best epoch: {result['best_epoch']}.",
        f"- Best checkpoint: `{result['checkpoints']['best_validation']['path']}`.",
        "",
        "## Best validation",
        "",
        "| metric | @5 | @10 | @20 | @50 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| HR | {format_float(best['HR@5'])} | {format_float(best['HR@10'])} | {format_float(best['HR@20'])} | {format_float(best['HR@50'])} |",
        f"| Recall | {format_float(best['Recall@5'])} | {format_float(best['Recall@10'])} | {format_float(best['Recall@20'])} | {format_float(best['Recall@50'])} |",
        f"| NDCG | {format_float(best['NDCG@5'])} | {format_float(best['NDCG@10'])} | {format_float(best['NDCG@20'])} | {format_float(best['NDCG@50'])} |",
        "",
        "## Final test",
        "",
        f"- `test_evaluation_count = {result['test_evaluation_count']}`.",
        "| metric | @5 | @10 | @20 | @50 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| HR | {format_float(test['HR@5'])} | {format_float(test['HR@10'])} | {format_float(test['HR@20'])} | {format_float(test['HR@50'])} |",
        f"| Recall | {format_float(test['Recall@5'])} | {format_float(test['Recall@10'])} | {format_float(test['Recall@20'])} | {format_float(test['Recall@50'])} |",
        f"| NDCG | {format_float(test['NDCG@5'])} | {format_float(test['NDCG@10'])} | {format_float(test['NDCG@20'])} | {format_float(test['NDCG@50'])} |",
        "",
        "## Auxiliary behavior metrics",
        "",
        "| target | valid ROC-AUC | valid PR-AUC | test ROC-AUC | test PR-AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for target in TARGETS:
        lines.append(
            f"| `{target}` | {format_float(aux_valid[target]['roc_auc'])} | {format_float(aux_valid[target]['pr_auc'])} | "
            f"{format_float(aux_test[target]['roc_auc'])} | {format_float(aux_test[target]['pr_auc'])} |"
        )
    lines += [
        "",
        "## Сравнение с TiM4Rec",
        "",
        f"- Test NDCG@10 delta: {format_float(nt['test_ndcg10_delta_vs_tim4rec'])} ({nt['test_quality']}).",
        f"- Validation NDCG@10 delta: {format_float(nt['validation_ndcg10_delta_vs_tim4rec'])} ({nt['validation_quality']}).",
        "",
        "## Сравнение с остальными baseline",
        "",
        "| model | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["comparison_vs_other_baselines"]:
        lines.append("| " + row["model"] + " | " + " | ".join(format_table_metrics(row["test_metrics"])) + " |")
    lines += [
        "",
        "## Negative transfer",
        "",
        f"- Формально significant-флаг не выставляется без repeated seeds.",
        f"- Итоговая оценка по test NDCG@10: `{nt['test_quality']}`.",
        "",
        "## Стоимость модели",
        "",
        f"- Base params: {params['base']['total']}.",
        f"- Multitask params: {params['multitask']['total']}.",
        f"- Delta: {params['delta_total']} ({format_float(params['relative_increase_percent'])}%).",
        f"- Runtime: {format_float(result['runtime']['total_sec'], 2)} sec.",
        f"- Peak VRAM allocated: {result['gpu']['peak_allocated_bytes']} bytes.",
        "",
        "## Ограничения",
        "",
        "- Один seed; выводы о статистической значимости не делаются.",
        "- Loss weights и target set намеренно не тюнились.",
        "",
        "## Вывод",
        "",
        f"- Fixed-loss multitask ranking result: `{result['decision']['ranking_outcome']}`.",
        f"- Next step: `{result['decision']['next_recommended_step']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise RuntimeError(f"This script is pinned to run_id={RUN_ID}, got {args.run_id}")
    if int(args.epochs) > 300:
        raise RuntimeError(f"Full fixed-loss run allows at most 300 epochs, got {args.epochs}")

    result_path = Path(args.result_json)
    notes_path = Path(args.notes)
    artifact_dir = Path(args.artifact_dir)
    checkpoint_dir = artifact_dir / "checkpoints"
    training_log_path = artifact_dir / "training_log.jsonl"
    partial_path = result_path.with_suffix(".partial.json")
    if result_path.exists() or partial_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing run JSON: {result_path}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty artifact dir: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sanity_payload = load_json(Path(args.sanity_run_json))
    sanity_decision = sanity_payload["smoke"]["loss_policy_decision"]
    if float(sanity_payload["lambda_aux"]) != LOCKED_LAMBDA_AUX or not sanity_decision["use_pos_weight"]:
        raise RuntimeError(f"Unexpected sanity loss policy: {sanity_decision}")

    multitask_dir = Path(args.multitask_dir)
    manifest_path = Path(args.multitask_manifest)
    manifest = assert_multitask_manifest(manifest_path)
    recbole_inter = ensure_recbole_inter(multitask_dir)
    if int(recbole_inter["rows"]) != EXPECTED_FINGERPRINT["interactions"]:
        raise RuntimeError(f"RecBole .inter row count mismatch: {recbole_inter}")
    if not recbole_inter["validation_source_row_ids_available"]:
        raise RuntimeError(f"Missing validation source_row_id sidecar: {recbole_inter}")
    target_stats = load_target_stats(Path(args.target_statistics))
    baselines = load_baseline_table()

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
    config = Config(model=MultitaskTiM4Rec, config_file_list=[args.config], config_dict=config_overrides)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for MultitaskTiM4Rec full training.")
    if not bool(config["is_time"]):
        raise RuntimeError("Full run must keep TiM4Rec is_time=True.")
    if tuple(config["multitask_targets"]) != TARGETS:
        raise RuntimeError(f"Unexpected multitask targets: {config['multitask_targets']}")

    torch.cuda.reset_peak_memory_stats()
    start_monotonic = time.monotonic()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    expected_source_ids = expected_validation_source_ids(multitask_dir)
    valid_loader_inspection = inspect_eval_loader(valid_data, int(valid_data._dataset.item_num), expected_source_ids)
    test_loader_inspection = inspect_eval_loader(test_data, int(test_data._dataset.item_num), None)
    for name, inspection in (("validation", valid_loader_inspection), ("test", test_loader_inspection)):
        if not inspection["one_positive_per_row"]:
            raise RuntimeError(f"{name} must have one positive per row: {inspection}")
        if not inspection["positive_targets_within_item_universe"]:
            raise RuntimeError(f"{name} positives outside item universe: {inspection}")
    if int(valid_data._dataset.item_num) - 1 != EXPECTED_FINGERPRINT["items"]:
        raise RuntimeError(f"Full-ranking item universe mismatch: {int(valid_data._dataset.item_num) - 1}")

    device = config["device"]
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    smoke_model = MultitaskTiM4Rec(config, train_data.dataset).to(device)
    smoke_optimizer = torch.optim.Adam(
        smoke_model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    smoke = run_smoke(
        smoke_model,
        smoke_optimizer,
        first_batch(train_data, device),
        target_stats,
        configured_lambda_aux=LOCKED_LAMBDA_AUX,
    )
    if float(smoke["loss_policy_decision"]["lambda_aux"]) != LOCKED_LAMBDA_AUX:
        raise RuntimeError(f"Lambda changed unexpectedly: {smoke['loss_policy_decision']}")
    if not bool(smoke["loss_policy_decision"]["use_pos_weight"]):
        raise RuntimeError(f"pos_weight policy changed unexpectedly: {smoke['loss_policy_decision']}")
    smoke["loss_policy_decision"]["lambda_source"] = f"locked_from_{SANITY_RUN_ID}"
    selected_pos_weights = pos_weight_tensors(target_stats, device)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = MultitaskTiM4Rec(config, train_data.dataset).to(device)
    base_model = TiM4Rec(config, train_data.dataset).to(device)
    base_params = count_parameters(base_model)
    multitask_params = count_parameters(model)
    del base_model
    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    trainer = Trainer(config, model)
    trainer.optimizer = optimizer
    model.lambda_aux = LOCKED_LAMBDA_AUX
    model.pos_weights = selected_pos_weights

    best_valid_score = -float("inf")
    best_epoch = None
    best_snapshot: dict[str, Any] | None = None
    cur_step = 0
    epoch_results = []
    topk = list(config["topk"])
    valid_metric = str(config["valid_metric"]).lower()
    valid_metric_bigger = bool(config["valid_metric_bigger"])
    best_checkpoint = None
    last_checkpoint = None
    stop_reason = "max_epochs"
    test_evaluation_count = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.monotonic()
        train_start = time.monotonic()
        losses = train_one_epoch(model, optimizer, train_data, device, LOCKED_LAMBDA_AUX, selected_pos_weights)
        train_time = time.monotonic() - train_start

        valid_start = time.monotonic()
        valid_result, full_ranking_checks = evaluate_full_sort_with_checks(trainer, valid_data, train_data)
        auxiliary_validation = evaluate_auxiliary(model, valid_data, device)
        validation_time = time.monotonic() - valid_start
        hit_recall_check = check_hit_recall_equal(valid_result, topk)
        if not full_ranking_checks["raw_scores_all_finite"]:
            raise RuntimeError(f"Non-finite validation scores: {full_ranking_checks}")

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
            best_checkpoint = checkpoint_sha(
                save_checkpoint(
                    model,
                    optimizer,
                    config,
                    checkpoint_dir / "best_validation.pth",
                    epoch,
                    best_valid_score,
                    valid_result,
                )
            )
        last_checkpoint = checkpoint_sha(
            save_checkpoint(
                model,
                optimizer,
                config,
                checkpoint_dir / "last.pth",
                epoch,
                best_valid_score,
                valid_result,
            )
        )

        epoch_result = {
            "epoch": epoch,
            "losses": losses,
            "validation": metric_subset(valid_result),
            "auxiliary_validation": auxiliary_validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
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
            "train_time_sec": train_time,
            "validation_time_sec": validation_time,
            "epoch_time_sec": time.monotonic() - epoch_start,
            "gpu_peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved()),
        }
        epoch_results.append(epoch_result)
        if update_flag:
            best_snapshot = epoch_result
        with training_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(compact_epoch(epoch_result), ensure_ascii=False, default=json_default) + "\n")
        partial_path.write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "partial",
                    "epochs_completed": len(epoch_results),
                    "latest_epoch": compact_epoch(epoch_results[-1]),
                    "best_epoch_so_far": best_epoch,
                    "best_valid_score_so_far": float(best_valid_score),
                    "test_evaluation_count": test_evaluation_count,
                },
                indent=2,
                ensure_ascii=False,
                default=json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "losses": losses,
                    "validation_ndcg10": valid_result["ndcg@10"],
                    "validation_hit10": valid_result["hit@10"],
                    "cur_step": int(cur_step),
                    "update_flag": bool(update_flag),
                    "stop_flag": bool(stop_flag),
                },
                ensure_ascii=False,
                default=json_default,
            ),
            flush=True,
        )
        if stop_flag:
            stop_reason = f"early_stopping_no_improvement_{int(config['stopping_step'])}"
            break

    if best_snapshot is None or best_checkpoint is None:
        raise RuntimeError("No best validation checkpoint recorded.")

    checkpoint = torch.load(best_checkpoint["path"], map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    trainer.model = model
    test_start = time.monotonic()
    test_result, test_checks = evaluate_full_sort_with_checks(trainer, test_data, train_data)
    test_checks["evaluation"] = "test_full_7111_items"
    test_hit_recall = check_hit_recall_equal(test_result, topk)
    test_auxiliary = evaluate_auxiliary(model, test_data, device)
    test_time = time.monotonic() - test_start
    test_evaluation_count = 1

    final_test_metrics = normalize_metrics(metric_subset(test_result))
    best_validation_metrics = normalize_metrics(best_snapshot["validation"])
    tim4rec_test = baselines["TiM4Rec"]["test"]
    tim4rec_valid = baselines["TiM4Rec"]["validation"]
    comparison_vs_tim4rec = compare_against(tim4rec_test, final_test_metrics)
    comparison_vs_other_baselines = [
        {"model": name, "run_id": data["run_id"], "test_metrics": data["test"]}
        for name, data in baselines.items()
    ]
    comparison_vs_other_baselines.append({"model": "MultitaskTiM4Rec", "run_id": RUN_ID, "test_metrics": final_test_metrics})

    validation_delta = best_validation_metrics["NDCG@10"] - tim4rec_valid["NDCG@10"]
    test_delta = final_test_metrics["NDCG@10"] - tim4rec_test["NDCG@10"]
    runtime_sec = time.monotonic() - start_monotonic
    ru_maxrss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    pipeline_ok = (
        smoke["all_losses_finite"]
        and smoke["all_gradients_after_combined_backward"]["all_finite"]
        and smoke["all_heads_updated"]
        and all(epoch["full_ranking_checks"]["raw_scores_all_finite"] for epoch in epoch_results)
        and test_checks["raw_scores_all_finite"]
        and test_evaluation_count == 1
    )

    result = {
        "run_id": args.run_id,
        "status": "completed" if pipeline_ok else "completed_with_warnings",
        "sanity": False,
        "no_full_training_performed": False,
        "test_evaluation_count": test_evaluation_count,
        "test_safety": {
            "test_used_during_training": False,
            "checkpoint_selection_metric": "validation NDCG@10 only",
            "test_evaluated_after_best_checkpoint_load": True,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": "TiM4Rec",
        "base_run": "tim4rec_001",
        "model_name": "MultitaskTiM4Rec",
        "project_git_commit": os.environ.get("MULTITASK_TIM4REC_GIT_COMMIT", git_value(["git", "rev-parse", "HEAD"])),
        "branch": os.environ.get("MULTITASK_TIM4REC_GIT_BRANCH", git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"])),
        "source_files": {
            "model.py": sha256_file(ROOT / "experiments" / "multitask_tim4rec" / "model.py"),
            "train_full.py": sha256_file(ROOT / "experiments" / "multitask_tim4rec" / "train_full.py"),
            "config.yaml": sha256_file(ROOT / "experiments" / "multitask_tim4rec" / "config.yaml"),
            "slurm/multitask_tim4rec.sh": sha256_file(ROOT / "slurm" / "multitask_tim4rec.sh"),
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
            "device": str(device),
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(torch.cuda.current_device()))),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "memory": {"process_ru_maxrss_kb": ru_maxrss_kb, "process_ru_maxrss": human_size(ru_maxrss_kb * 1024)},
        "dataset": {
            "multitask_dir": str(multitask_dir),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "identity_hash": EXPECTED_IDENTITY_HASH,
            "fingerprint": EXPECTED_FINGERPRINT,
            "multitask_manifest": {
                "join_diagnostics": manifest["join_diagnostics"],
                "dataset_fingerprint": manifest["dataset_fingerprint"],
            },
            "recbole_inter": recbole_inter,
            "recbole": {
                "user_num_with_padding": int(dataset.user_num),
                "item_num_with_padding": int(dataset.item_num),
                "item_universe_without_padding": int(dataset.item_num) - 1,
                "inter_num_after_sequential_augmentation": int(dataset.inter_num),
                "train_batches": len(train_data),
                "valid_batches": len(valid_data),
                "test_batches": len(test_data),
                "validation_loader": valid_loader_inspection,
                "test_loader": test_loader_inspection,
            },
        },
        "targets": list(TARGETS),
        "class_statistics": target_stats,
        "pos_weights": {target: target_stats[target]["negative_positive_ratio"] for target in TARGETS},
        "loss_config_source": SANITY_RUN_ID,
        "loss_formula": "L_total = L_rank + lambda_aux * (L_click + L_long_view + L_like + L_profile)",
        "lambda_aux": LOCKED_LAMBDA_AUX,
        "training_config": {
            "config_file": str(Path(args.config).resolve()),
            "epochs_requested": int(args.epochs),
            "epochs_completed": len(epoch_results),
            "stop_reason": stop_reason,
            "seed": int(config["seed"]),
            "learning_rate": float(config["learning_rate"]),
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "stopping_step": int(config["stopping_step"]),
            "metrics": list(config["metrics"]),
            "topk": topk,
            "valid_metric": str(config["valid_metric"]),
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
                "is_time": bool(config["is_time"]),
                "p2p_residual": bool(config["p2p_residual"]),
            },
        },
        "architecture": {
            "backbone": "validated TiM4Rec from experiments/tim4rec_baseline/upstream/tim4rec.py",
            "shared_representation": "TiM4Rec.forward(item_id_list, item_length, timestamp_list)",
            "heads": {
                "click_head": "Linear(64, 1)",
                "long_view_head": "Linear(64, 1)",
                "like_head": "Linear(64, 1)",
                "profile_enter_head": "Linear(64, 1)",
            },
            "no_moe": True,
            "no_adaptive_loss": True,
            "no_new_attention": True,
            "no_flow_matching": True,
        },
        "model_parameters": {
            "base": base_params,
            "multitask": multitask_params,
            "delta_total": multitask_params["total"] - base_params["total"],
            "delta_trainable": multitask_params["trainable"] - base_params["trainable"],
            "relative_increase_percent": (multitask_params["total"] - base_params["total"]) / base_params["total"] * 100.0,
        },
        "smoke_confirmation": {
            "all_heads_trainable": all(param.requires_grad for params in model.auxiliary_heads().values() for param in params.parameters()),
            "all_losses_finite": smoke["all_losses_finite"],
            "all_gradients_after_combined_backward": smoke["all_gradients_after_combined_backward"],
            "all_heads_updated": smoke["all_heads_updated"],
            "auxiliary_gradients_reach_backbone": {
                item["loss_key"]: item["backbone_grad_norm"] for item in smoke["gradient_diagnostics"] if item["loss_key"] != "rank"
            },
            "gradient_check_after_last_train_batch": all_gradient_check(model),
        },
        "epochs": epoch_results,
        "loss_history": [
            {
                "epoch": epoch["epoch"],
                "L_total": epoch["losses"]["total"],
                "L_rank": epoch["losses"]["rank"],
                "L_click": epoch["losses"]["is_click_loss"],
                "L_long_view": epoch["losses"]["long_view_loss"],
                "L_like": epoch["losses"]["is_like_loss"],
                "L_profile": epoch["losses"]["is_profile_enter_loss"],
                "validation_ndcg10": epoch["validation"]["ndcg@10"],
                "validation_hr10": epoch["validation"]["hit@10"],
                "auxiliary_validation": epoch["auxiliary_validation"],
                "learning_rate": epoch["learning_rate"],
                "train_time_sec": epoch["train_time_sec"],
                "validation_time_sec": epoch["validation_time_sec"],
                "gpu_peak_allocated_bytes": epoch["gpu_peak_allocated_bytes_so_far"],
                "gpu_peak_reserved_bytes": epoch["gpu_peak_reserved_bytes_so_far"],
            }
            for epoch in epoch_results
        ],
        "best_epoch": best_epoch,
        "best_valid_metric": valid_metric,
        "best_valid_score": float(best_valid_score),
        "best_validation": best_snapshot,
        "best_validation_metrics": best_validation_metrics,
        "checkpoints": {"best_validation": best_checkpoint, "last": last_checkpoint},
        "final_test": {
            "recommendation_metrics": final_test_metrics,
            "recommendation_metrics_lowercase": lowercase_metrics(final_test_metrics),
            "auxiliary_metrics": test_auxiliary,
            "full_ranking_checks": test_checks,
            "hit_recall_equal_check": test_hit_recall,
            "evaluation_time_sec": test_time,
        },
        "baseline_comparison": {
            "baselines": baselines,
            "comparison_vs_tim4rec_001": comparison_vs_tim4rec,
        },
        "comparison_vs_other_baselines": comparison_vs_other_baselines,
        "negative_transfer_analysis": {
            "validation_ndcg10_delta_vs_tim4rec": validation_delta,
            "test_ndcg10_delta_vs_tim4rec": test_delta,
            "validation_quality": qualitative_delta(validation_delta),
            "test_quality": qualitative_delta(test_delta),
            "statistical_significance_claimed": False,
            "reason": "single seed; repeated seeds were not run",
        },
        "runtime": {
            "total_sec": runtime_sec,
            "mean_epoch_sec": sum(epoch["epoch_time_sec"] for epoch in epoch_results) / max(len(epoch_results), 1),
            "tim4rec_001_total_sec": baselines["TiM4Rec"].get("runtime", {}).get("total_sec"),
            "runtime_delta_vs_tim4rec_sec": None
            if baselines["TiM4Rec"].get("runtime", {}).get("total_sec") is None
            else runtime_sec - float(baselines["TiM4Rec"]["runtime"]["total_sec"]),
        },
        "remote_artifact_path": str(artifact_dir),
        "remote_training_log_path": str(training_log_path),
        "decision": {
            "pipeline_correct": pipeline_ok,
            "fixed_loss_multitask_helped_ranking_vs_tim4rec": test_delta > 0,
            "exceeds_tim4rec": test_delta > 0,
            "negative_transfer_detected": test_delta < 0,
            "ranking_outcome": qualitative_delta(test_delta),
            "do_optuna_next": False,
            "motivation_for_adaptive_loss_or_behavior_moe": test_delta <= 0,
            "next_recommended_step": (
                "analyze fixed-loss negative transfer before Optuna/adaptive loss/Behavior MoE"
                if test_delta <= 0
                else "repeat seed or small fixed-loss confirmation before tuning"
            ),
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    notes_path.write_text(build_notes(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=json_default), flush=True)


if __name__ == "__main__":
    main()
