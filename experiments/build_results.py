#!/usr/bin/env python
"""Build canonical experiment registry and markdown result tables."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENTS_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENTS_DIR.parent
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
RESULTS_MD = EXPERIMENTS_DIR / "RESULTS.md"
AUDIT_MD = EXPERIMENTS_DIR / "RESULTS_AUDIT.md"
PAPER_REFERENCES = EXPERIMENTS_DIR / "paper_references.yaml"

BASE_FIELDS = [
    "record_type",
    "source",
    "run_id",
    "model",
    "model_variant",
    "dataset",
    "protocol",
    "split",
    "evaluation",
    "status",
    "parent_run",
    "source_paper",
    "paper_version",
    "seed",
    "train_candidates",
    "item_universe",
    "HR@5",
    "HR@10",
    "HR@20",
    "HR@50",
    "Recall@5",
    "Recall@10",
    "Recall@20",
    "Recall@50",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "NDCG@50",
    "best_epoch",
    "actual_epochs",
    "validation_ndcg10",
    "test_evaluation_count",
    "git_commit",
    "notes_path",
]
EXTRA_FIELDS = [
    "trials_complete",
    "trials_pruned",
    "trials_failed",
    "best_trial",
    "best_validation_NDCG@10",
    "test_used",
    "source_json",
]
FIELDS = BASE_FIELDS + EXTRA_FIELDS
METRIC_FIELDS = [
    "HR@5",
    "HR@10",
    "HR@20",
    "HR@50",
    "Recall@5",
    "Recall@10",
    "Recall@20",
    "Recall@50",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "NDCG@50",
]
MAIN_ORDER = [
    ("mostpop_002", "MostPopular"),
    ("ltr_xgb_002", "XGBoost LambdaMART"),
    ("ltr_xgb_optuna_001", "XGBoost LambdaMART tuned"),
    ("ssd4rec_001", "SSD4Rec reproduction"),
    ("tim4rec_001", "TiM4Rec reproduction"),
    ("multitask_tim4rec_001", "MultitaskTiM4Rec fixed"),
    ("multitask_tim4rec_tuned_001", "MultitaskTiM4Rec tuned"),
]
REPRODUCTION_PAIRS = [
    ("TiM4Rec", "paper_tim4rec", "tim4rec_001", ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]),
    ("SSD4Rec", "paper_ssd4rec_v2", "ssd4rec_001", ["HR@10", "HR@20", "NDCG@10", "NDCG@20"]),
]


RUN_METADATA: dict[str, dict[str, Any]] = {
    "random_001": {
        "record_type": "experiment",
        "model": "Random",
        "model_variant": "sampled_history",
        "split": "test",
        "evaluation": "sampled_100",
        "train_candidates": "sampled_100",
    },
    "mostpop_001": {
        "record_type": "experiment",
        "model": "MostPopular",
        "model_variant": "sampled_history",
        "split": "test",
        "evaluation": "sampled_100",
        "train_candidates": "not_applicable",
    },
    "ltr_xgb_001": {
        "record_type": "experiment",
        "model": "XGBoost LambdaMART",
        "model_variant": "sampled_history",
        "split": "test",
        "evaluation": "sampled_100",
        "train_candidates": "sampled_100",
    },
    "random_002": {
        "record_type": "experiment",
        "model": "Random",
        "model_variant": "full_ranking_history",
        "split": "test",
        "evaluation": "full_7111_items",
        "train_candidates": "sampled_100",
    },
    "mostpop_002": {
        "record_type": "experiment",
        "model": "MostPopular",
        "model_variant": "full_ranking_history",
        "split": "test",
        "evaluation": "full_7111_items",
        "train_candidates": "not_applicable",
    },
    "ltr_xgb_002": {
        "record_type": "experiment",
        "model": "XGBoost LambdaMART",
        "model_variant": "baseline_full_ranking",
        "split": "test",
        "evaluation": "full_7111_items",
        "train_candidates": "sampled_100",
    },
    "ltr_xgb_optuna_001": {
        "record_type": "experiment",
        "model": "XGBoost LambdaMART",
        "model_variant": "tuned_optuna",
        "split": "test",
        "evaluation": "full_7111_items",
        "train_candidates": "sampled_100",
        "parent_run": "optuna_search_001",
    },
    "tim4rec_001": {
        "record_type": "experiment",
        "model": "TiM4Rec",
        "model_variant": "reproduction",
        "split": "test",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
    },
    "ssd4rec_001": {
        "record_type": "experiment",
        "model": "SSD4Rec",
        "model_variant": "reproduction",
        "split": "test",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
    },
    "multitask_tim4rec_001": {
        "record_type": "experiment",
        "model": "MultitaskTiM4Rec",
        "model_variant": "fixed_loss",
        "split": "test",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
        "parent_run": "tim4rec_001",
    },
    "multitask_tim4rec_tuned_001": {
        "record_type": "experiment",
        "model": "MultitaskTiM4Rec",
        "model_variant": "tuned_fixed_weights",
        "split": "test",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
        "parent_run": "multitask_optuna_search_001",
    },
    "tim4rec_sanity_001": {
        "record_type": "sanity",
        "model": "TiM4Rec",
        "model_variant": "sanity_5_epoch",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
    },
    "ssd4rec_sanity_001": {
        "record_type": "sanity",
        "model": "SSD4Rec",
        "model_variant": "sanity_5_epoch",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
    },
    "multitask_tim4rec_sanity_001": {
        "record_type": "sanity",
        "model": "MultitaskTiM4Rec",
        "model_variant": "sanity_5_epoch",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
    },
    "smoke_20260818T132855Z": {
        "record_type": "sanity",
        "model": "TiM4Rec",
        "model_variant": "smoke_forward",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
    },
    "smoke_20260819T110252Z": {
        "record_type": "sanity",
        "model": "SSD4Rec",
        "model_variant": "smoke_forward",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
    },
    "target_audit_001": {
        "record_type": "sanity",
        "model": "Multitask target audit",
        "model_variant": "target_labels_audit",
        "split": "train",
        "evaluation": "diagnostic",
        "train_candidates": "not_applicable",
    },
    "optuna_smoke_001": {
        "record_type": "sanity",
        "model": "XGBoost LambdaMART",
        "model_variant": "optuna_smoke",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "sampled_100",
        "parent_run": "ltr_xgb_002",
    },
    "adaptive_smoke_001": {
        "record_type": "sanity",
        "model": "MultitaskTiM4Rec",
        "model_variant": "adaptive_gradient_smoke",
        "split": "train",
        "evaluation": "diagnostic",
        "train_candidates": "full_sequence",
        "parent_run": "multitask_tim4rec_tuned_001",
    },
    "behavior_moe_smoke_001": {
        "record_type": "sanity",
        "model": "BehaviorMoETiM4Rec",
        "model_variant": "behavior_specialized_soft_moe_smoke",
        "split": "train",
        "evaluation": "diagnostic",
        "train_candidates": "full_sequence",
        "parent_run": "multitask_tim4rec_tuned_001",
    },
    "behavior_moe_sanity_001": {
        "record_type": "sanity",
        "model": "BehaviorMoETiM4Rec",
        "model_variant": "behavior_specialized_soft_moe_sanity_5_epoch",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
        "parent_run": "multitask_tim4rec_tuned_001",
    },
    "structured_behavior_moe_smoke_001": {
        "record_type": "sanity",
        "model": "StructuredBehaviorMoE",
        "model_variant": "structured_behavior_moe_architecture_probe",
        "split": "train",
        "evaluation": "diagnostic",
        "train_candidates": "full_sequence",
        "parent_run": "behavior_moe_sanity_001",
    },
    "ple_tim4rec_smoke_001": {
        "record_type": "sanity",
        "model": "PLETiM4Rec",
        "model_variant": "cgc_1level_ple_style_smoke",
        "split": "train",
        "evaluation": "diagnostic",
        "train_candidates": "full_sequence",
        "parent_run": "behavior_moe_sanity_001",
    },
    "ple_tim4rec_sanity_001": {
        "record_type": "sanity",
        "model": "PLETiM4Rec",
        "model_variant": "cgc_1level_ple_style_sanity_5_epoch",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
        "parent_run": "ple_tim4rec_smoke_001",
    },
    "pcgrad_sanity_001": {
        "record_type": "sanity",
        "model": "MultitaskTiM4Rec",
        "model_variant": "adaptive_pcgrad_ranking_anchored",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
        "parent_run": "multitask_tim4rec_tuned_001",
    },
    "metabalance_sanity_001": {
        "record_type": "sanity",
        "model": "MultitaskTiM4Rec",
        "model_variant": "adaptive_metabalance_fix",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
        "parent_run": "multitask_tim4rec_tuned_001",
    },
    "pcgrad_001": {
        "record_type": "experiment_validation_only",
        "model": "MultitaskTiM4Rec",
        "model_variant": "adaptive_pcgrad_ranking_anchored_full",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
        "parent_run": "multitask_tim4rec_tuned_001",
    },
    "optuna_search_001": {
        "record_type": "search",
        "model": "XGBoost LambdaMART",
        "model_variant": "optuna_search",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "sampled_100",
        "parent_run": "ltr_xgb_002",
    },
    "multitask_optuna_search_001": {
        "record_type": "search",
        "model": "MultitaskTiM4Rec",
        "model_variant": "optuna_search",
        "split": "validation",
        "evaluation": "full_7111_items",
        "train_candidates": "full_sequence",
        "parent_run": "multitask_tim4rec_001",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_id_for(path: Path, payload: dict[str, Any]) -> str:
    return str(payload.get("run_id") or path.stem)


def relative_path(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def find_notes(path: Path, run_id: str) -> str:
    candidates = [
        path.with_name(f"{run_id}_notes.md"),
        path.with_name(f"{run_id}.md"),
        path.with_suffix(".md"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return relative_path(candidate)
    return ""


def status_for(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "completed")
    return "completed" if status in {"ok", "COMPLETED"} else status


def nested_get(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_present(payload: dict[str, Any], key_paths: list[list[str]]) -> Any:
    for key_path in key_paths:
        value = nested_get(payload, key_path)
        if value is not None:
            return value
    return None


def value_by_metric(metrics: dict[str, Any] | None, field: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    metric, k = field.split("@")
    candidates = [field, field.lower(), f"{metric.lower()}@{k}", f"{metric.upper()}@{k}"]
    if metric == "HR":
        candidates.extend([f"hit@{k}", f"Hit@{k}"])
    for key in candidates:
        if key in metrics and metrics[key] not in ("", None):
            return float(metrics[key])
    return None


def add_metric_values(row: dict[str, Any], metrics: dict[str, Any] | None) -> None:
    for field in METRIC_FIELDS:
        value = value_by_metric(metrics, field)
        if value is not None:
            row[field] = value


def final_test_metrics(payload: dict[str, Any]) -> dict[str, Any] | None:
    return first_present(
        payload,
        [
            ["final_test_metrics"],
            ["final_test", "recommendation_metrics"],
            ["final_test", "recommendation_metrics_lowercase"],
            ["metrics", "test"],
        ],
    )


def validation_metrics(payload: dict[str, Any]) -> dict[str, Any] | None:
    found = first_present(
        payload,
        [
            ["best_validation_metrics"],
            ["best_validation", "validation"],
            ["best_validation_epoch", "validation"],
            ["validation_reproduction", "reproduced_validation"],
            ["validation_optuna_metrics"],
            ["metrics", "validation"],
        ],
    )
    if found is not None:
        return found
    best_epoch = payload.get("best_epoch")
    epochs = payload.get("epochs")
    if isinstance(best_epoch, int) and isinstance(epochs, list):
        for epoch in epochs:
            if int(epoch.get("epoch", -1)) == best_epoch and isinstance(epoch.get("validation"), dict):
                return epoch["validation"]
    if isinstance(epochs, list) and epochs and isinstance(epochs[-1].get("validation"), dict):
        return epochs[-1]["validation"]
    return None


def item_universe(payload: dict[str, Any]) -> Any:
    return first_present(
        payload,
        [
            ["dataset", "fingerprint", "items"],
            ["dataset", "recbole", "item_universe_without_padding"],
            ["dataset_fingerprint", "items"],
            ["protocol", "item_universe_size"],
            ["item_universe_size"],
        ],
    ) or 7111


def seed_for(payload: dict[str, Any]) -> Any:
    return first_present(
        payload,
        [
            ["seed"],
            ["model_seed"],
            ["config", "seed"],
            ["model", "params", "seed"],
            ["best_params", "params", "seed"],
        ],
    )


def git_commit_for(payload: dict[str, Any]) -> str:
    return str(
        first_present(
            payload,
            [
                ["git_commit"],
                ["project_git_commit"],
                ["git", "commit"],
            ],
        )
        or ""
    )


def best_epoch_for(payload: dict[str, Any]) -> Any:
    return first_present(
        payload,
        [
            ["best_epoch"],
            ["best_validation_epoch", "epoch"],
            ["best_validation", "epoch"],
            ["validation_reproduction", "best_epoch"],
        ],
    )


def actual_epochs_for(payload: dict[str, Any]) -> Any:
    value = first_present(payload, [["actual_epochs"], ["epochs_completed"], ["validation_reproduction", "actual_epochs"]])
    if value is not None:
        return value
    epochs = payload.get("epochs")
    return len(epochs) if isinstance(epochs, list) else None


def test_count_for(record_type: str, split: str, payload: dict[str, Any], metrics: dict[str, Any] | None) -> Any:
    value = payload.get("test_evaluation_count")
    if value is not None:
        return int(value)
    nested = first_present(
        payload,
        [
            ["test_policy", "test_evaluation_count"],
            ["test_safety", "test_evaluation_count"],
            ["dataset", "split", "test_evaluations_count"],
        ],
    )
    if nested is not None:
        return int(nested)
    if record_type == "search":
        return 0
    if record_type == "sanity":
        return 0
    if record_type == "experiment" and split == "test" and metrics:
        return 1
    return ""


def base_row(path: Path, payload: dict[str, Any], run_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    best_epoch = best_epoch_for(payload)
    actual_epochs = actual_epochs_for(payload)
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "record_type": meta["record_type"],
            "source": "ours",
            "run_id": run_id,
            "model": meta["model"],
            "model_variant": meta["model_variant"],
            "dataset": "KuaiRand",
            "protocol": "B",
            "split": meta["split"],
            "evaluation": meta["evaluation"],
            "status": status_for(payload),
            "parent_run": meta.get("parent_run", ""),
            "seed": seed_for(payload) or "",
            "train_candidates": meta.get("train_candidates", ""),
            "item_universe": item_universe(payload),
            "best_epoch": best_epoch if best_epoch is not None else "",
            "actual_epochs": actual_epochs if actual_epochs is not None else "",
            "validation_ndcg10": "",
            "git_commit": git_commit_for(payload),
            "notes_path": find_notes(path, run_id),
            "source_json": relative_path(path),
        }
    )
    return row


def experiment_or_sanity_row(path: Path, payload: dict[str, Any], run_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    row = base_row(path, payload, run_id, meta)
    metrics = final_test_metrics(payload) if meta["split"] == "test" else validation_metrics(payload)
    add_metric_values(row, metrics)
    valid = validation_metrics(payload)
    ndcg10 = value_by_metric(valid, "NDCG@10")
    if ndcg10 is not None:
        row["validation_ndcg10"] = ndcg10
    row["test_evaluation_count"] = test_count_for(meta["record_type"], meta["split"], payload, metrics)
    return row


def xgb_search_row(path: Path, payload: dict[str, Any], run_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    row = base_row(path, payload, run_id, meta)
    best = payload["best_trial"]
    metrics = best.get("validation_metrics")
    add_metric_values(row, metrics)
    row["status"] = "completed"
    row["validation_ndcg10"] = best.get("validation_ndcg10") or best.get("value") or value_by_metric(metrics, "NDCG@10") or ""
    row["best_validation_NDCG@10"] = row["validation_ndcg10"]
    row["best_trial"] = best.get("trial_number", "")
    counts = payload.get("study", {}).get("state_counts", {})
    row["trials_complete"] = counts.get("COMPLETE", "")
    row["trials_pruned"] = counts.get("PRUNED", 0)
    row["trials_failed"] = counts.get("FAIL", "")
    row["test_evaluation_count"] = int(payload.get("test_safety", {}).get("test_evaluation_count", 0))
    row["test_used"] = "no"
    return row


def multitask_search_row(path: Path, payload: dict[str, Any], run_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    row = base_row(path, payload, run_id, meta)
    best = payload["best_trial"]
    metrics = best.get("validation_metrics")
    add_metric_values(row, metrics)
    row["status"] = "completed"
    row["validation_ndcg10"] = best.get("value") or value_by_metric(metrics, "NDCG@10") or ""
    row["best_validation_NDCG@10"] = row["validation_ndcg10"]
    row["best_trial"] = best.get("trial", "")
    counts = payload.get("study_state_counts", {})
    row["trials_complete"] = counts.get("COMPLETE", "")
    row["trials_pruned"] = counts.get("PRUNED", "")
    row["trials_failed"] = counts.get("FAIL", "")
    row["test_evaluation_count"] = int(payload.get("test_safety", {}).get("test_evaluation_count", 0))
    row["test_used"] = "no"
    return row


def optuna_smoke_row(path: Path, payload: dict[str, Any], run_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    row = base_row(path, payload, run_id, meta)
    smoke = payload.get("smoke_trial", {})
    row["status"] = "completed"
    row["best_trial"] = smoke.get("trial_number", "")
    row["validation_ndcg10"] = smoke.get("validation_ndcg10", "")
    row["best_validation_NDCG@10"] = smoke.get("validation_ndcg10", "")
    row["HR@10"] = smoke.get("validation_hr10", "")
    row["Recall@10"] = smoke.get("validation_hr10", "")
    row["NDCG@10"] = smoke.get("validation_ndcg10", "")
    row["test_evaluation_count"] = int(payload.get("test_evaluation_count", 0))
    row["test_used"] = "no"
    return row


def row_for_run(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    run_id = run_id_for(path, payload)
    meta = RUN_METADATA.get(run_id)
    if not meta:
        return None
    if run_id == "optuna_search_001":
        return xgb_search_row(path, payload, run_id, meta)
    if run_id == "multitask_optuna_search_001":
        return multitask_search_row(path, payload, run_id, meta)
    if run_id == "optuna_smoke_001":
        return optuna_smoke_row(path, payload, run_id, meta)
    return experiment_or_sanity_row(path, payload, run_id, meta)


def paper_rows() -> list[dict[str, Any]]:
    text = PAPER_REFERENCES.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]

        payload = yaml.safe_load(text)
    except ModuleNotFoundError:
        payload = json.loads(text)
    rows: list[dict[str, Any]] = []
    for key, ref in payload.items():
        run_id = "paper_tim4rec" if key == "tim4rec" else "paper_ssd4rec_v2"
        row = {field: "" for field in FIELDS}
        row.update(
            {
                "record_type": "paper_reference",
                "source": "paper",
                "run_id": run_id,
                "model": ref["model"],
                "model_variant": ref["model_variant"],
                "dataset": ref["dataset"],
                "protocol": ref["protocol"],
                "split": ref["split"],
                "evaluation": ref["evaluation"],
                "status": "published",
                "source_paper": ref["source_paper"],
                "paper_version": ref["paper_version"],
                "item_universe": 7111,
                "notes_path": ref.get("audit_path", ""),
                "source_json": "experiments/paper_references.yaml",
            }
        )
        add_metric_values(row, ref.get("metrics"))
        rows.append(row)
    return rows


def existing_results_snapshot() -> dict[str, Any]:
    if not RESULTS_CSV.exists():
        return {"exists": False}
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return {
            "exists": True,
            "fieldnames": reader.fieldnames or [],
            "run_ids": sorted({row.get("run_id", "") for row in rows if row.get("run_id")}),
            "rows": len(rows),
        }


def collect_rows() -> tuple[list[dict[str, Any]], list[str]]:
    existing = existing_results_snapshot()
    notes: list[str] = []
    if existing.get("exists") and existing.get("fieldnames", [])[: len(BASE_FIELDS)] != BASE_FIELDS:
        notes.append(
            "Старый experiments/results.csv имел неканоническую схему без record_type/source/split/evaluation; "
            "он пересобран из JSON artifacts."
        )
    rows: list[dict[str, Any]] = []
    unknown_runs: list[str] = []
    for path in sorted(EXPERIMENTS_DIR.glob("**/runs/*.json")):
        payload = load_json(path)
        row = row_for_run(path, payload)
        if row is None:
            unknown_runs.append(relative_path(path))
            continue
        if row["run_id"] == "multitask_tim4rec_tuned_001" and row["status"] != "completed":
            notes.append("multitask_tim4rec_tuned_001 найден, но status не completed; строка исключена до locked test.")
            continue
        if row["run_id"] == "multitask_tim4rec_tuned_001" and (
            payload.get("resume_after_validation_gate_diagnostic") or payload.get("validation_gate_note")
        ):
            gate = payload.get("validation_gate", {}).get("comparisons", {})
            notes.append(
                "multitask_tim4rec_tuned_001 открыт на test после diagnostic tolerance: "
                f"NDCG@10 diff={gate.get('NDCG@10', {}).get('abs_diff')}, "
                f"HR@10 diff={gate.get('HR@10', {}).get('abs_diff')}; "
                "checkpoint не переобучался после первичного validation gate."
            )
        rows.append(row)
    if unknown_runs:
        notes.append("Найдены JSON artifacts без явного mapping в build_results.py: " + ", ".join(unknown_runs))
    rows.extend(paper_rows())
    rows.sort(key=lambda row: (record_order(row["record_type"]), str(row["run_id"])))
    return rows, notes


def record_order(record_type: str) -> int:
    return {
        "experiment": 0,
        "experiment_validation_only": 1,
        "search": 2,
        "sanity": 3,
        "paper_reference": 4,
    }.get(record_type, 99)


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def fmt_pct(value: Any) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.2f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def row_by_id(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row["run_id"] == run_id:
            return row
    return None


def main_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for run_id, label in MAIN_ORDER:
        row = row_by_id(rows, run_id)
        if not row:
            continue
        if row["record_type"] != "experiment" or row["split"] != "test" or row["evaluation"] != "full_7111_items":
            continue
        table_rows.append(
            [
                label,
                fmt(row["HR@10"]),
                fmt(row["HR@20"]),
                fmt(row["HR@50"]),
                fmt(row["NDCG@10"]),
                fmt(row["NDCG@20"]),
                fmt(row["NDCG@50"]),
            ]
        )
    return markdown_table(["Model", "HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"], table_rows)


def paper_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for run_id, label in (("paper_tim4rec", "TiM4Rec paper"), ("paper_ssd4rec_v2", "SSD4Rec paper v2")):
        row = row_by_id(rows, run_id)
        if not row:
            continue
        table_rows.append(
            [
                label,
                row["paper_version"],
                fmt(row["HR@10"]),
                fmt(row["HR@20"]),
                fmt(row["HR@50"]),
                fmt(row["NDCG@10"]),
                fmt(row["NDCG@20"]),
                fmt(row["NDCG@50"]),
            ]
        )
    return markdown_table(
        ["Source", "Version", "HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"],
        table_rows,
    )


def reproduction_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for model, paper_id, ours_id, metrics in REPRODUCTION_PAIRS:
        paper = row_by_id(rows, paper_id)
        ours = row_by_id(rows, ours_id)
        if not paper or not ours:
            continue
        for metric in metrics:
            paper_value = paper.get(metric, "")
            ours_value = ours.get(metric, "")
            if paper_value == "" or ours_value == "":
                continue
            diff = float(ours_value) - float(paper_value)
            rel = 100.0 * diff / float(paper_value) if float(paper_value) != 0 else None
            table_rows.append([model, metric, fmt(paper_value), fmt(ours_value), f"{diff:.4f}", "" if rel is None else f"{rel:.2f}%"])
    return markdown_table(["Model", "Metric", "Paper", "Ours", "Absolute diff", "Relative diff %"], table_rows)


def sanity_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        if row["record_type"] != "sanity":
            continue
        table_rows.append(
            [
                row["run_id"],
                row["model"],
                row["model_variant"],
                row["split"],
                row["evaluation"],
                row["status"],
                fmt(row["validation_ndcg10"] or row["NDCG@10"]),
                row["test_evaluation_count"],
            ]
        )
    return markdown_table(["Run", "Model", "Variant", "Split", "Evaluation", "Status", "NDCG@10", "Test count"], table_rows)


def validation_only_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        if row["record_type"] != "experiment_validation_only":
            continue
        table_rows.append(
            [
                row["run_id"],
                row["model"],
                row["model_variant"],
                row["status"],
                row["best_epoch"],
                row["actual_epochs"],
                fmt(row["HR@10"]),
                fmt(row["NDCG@10"]),
                row["test_evaluation_count"],
            ]
        )
    return markdown_table(
        ["Run", "Model", "Variant", "Status", "Best epoch", "Actual epochs", "HR@10", "NDCG@10", "Test count"],
        table_rows,
    )


def search_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        if row["record_type"] != "search":
            continue
        study = "XGBoost Optuna search" if row["run_id"] == "optuna_search_001" else "MultitaskTiM4Rec Optuna search"
        table_rows.append(
            [
                study,
                row["trials_complete"],
                row["trials_pruned"],
                row["trials_failed"],
                row["best_trial"],
                fmt(row["best_validation_NDCG@10"] or row["validation_ndcg10"]),
                row["test_used"],
            ]
        )
    return markdown_table(
        ["Study", "Trials complete", "Trials pruned", "Trials failed", "Best trial", "Best validation NDCG@10", "Test used?"],
        table_rows,
    )


def registry_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["record_type"],
                row["run_id"],
                row["source"],
                row["model"],
                row["model_variant"],
                row["split"],
                row["evaluation"],
                row["status"],
                fmt(row["HR@10"]),
                fmt(row["NDCG@10"]),
                row["test_evaluation_count"],
            ]
        )
    return markdown_table(
        ["Type", "Run", "Source", "Model", "Variant", "Split", "Evaluation", "Status", "HR@10", "NDCG@10", "Test count"],
        table_rows,
    )


def write_csv(rows: list[dict[str, Any]]) -> None:
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_markdown(rows: list[dict[str, Any]]) -> None:
    sections = [
        "# Основные результаты на KuaiRand Protocol B",
        f"Сгенерировано из `experiments/results.csv`: {datetime.now(timezone.utc).isoformat()}.",
        "Показаны только сопоставимые full-ranking TEST results.",
        main_table(rows),
        "# Опубликованные результаты",
        paper_table(rows),
        "# Воспроизводимость опубликованных моделей",
        reproduction_table(rows),
        "# Validation-only experiments",
        validation_only_table(rows),
        "# Sanity и диагностические запуски",
        sanity_table(rows),
        "# Hyperparameter search",
        search_table(rows),
        "# Полный реестр запусков",
        registry_table(rows),
    ]
    content = "\n\n".join(section.rstrip() for section in sections) + "\n"
    RESULTS_MD.write_text(content, encoding="utf-8")


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    seen = set()
    for row in rows:
        key = (row["record_type"], row["run_id"], row["split"], row["evaluation"])
        if key in seen:
            issues.append(f"Duplicate row key: {key}")
        seen.add(key)
        if row["record_type"] == "paper_reference" and row["source"] != "paper":
            issues.append(f"Paper row source must be paper: {row['run_id']}")
        if row["record_type"] != "paper_reference" and row["source"] != "ours":
            issues.append(f"Non-paper row source must be ours: {row['run_id']}")
        if row["record_type"] == "search":
            if row["split"] == "test":
                issues.append(f"Search row cannot use test split: {row['run_id']}")
            if str(row.get("test_evaluation_count", "")) not in {"", "0"}:
                issues.append(f"Search row must have test_evaluation_count=0: {row['run_id']}")
        if row["record_type"] == "sanity" and row["run_id"] in {run_id for run_id, _label in MAIN_ORDER}:
            issues.append(f"Sanity row would collide with main table run id: {row['run_id']}")
        if row["record_type"] == "experiment_validation_only":
            if row["split"] == "test":
                issues.append(f"Validation-only experiment cannot use test split: {row['run_id']}")
            if str(row.get("test_evaluation_count", "")) not in {"", "0"}:
                issues.append(f"Validation-only experiment must have test_evaluation_count=0: {row['run_id']}")
        if row["evaluation"] == "sampled_100" and row["run_id"].endswith("_002"):
            issues.append(f"*_002 run should not be sampled_100: {row['run_id']}")
        if row["evaluation"] == "full_7111_items" and row["run_id"].endswith("_001") and row["run_id"] in {"ltr_xgb_001"}:
            issues.append(f"ltr_xgb_001 must remain sampled_100: {row['run_id']}")
        for k in (5, 10, 20, 50):
            hr = row.get(f"HR@{k}", "")
            recall = row.get(f"Recall@{k}", "")
            if hr != "" and recall != "" and abs(float(hr) - float(recall)) > 1e-12:
                issues.append(f"HR and Recall differ for one-target row {row['run_id']} @{k}: {hr} vs {recall}")
        if row["record_type"] == "experiment" and row["split"] == "test" and row["evaluation"] == "full_7111_items":
            if row["run_id"] != "random_002" and row.get("test_evaluation_count", "") != 1:
                issues.append(f"Full test experiment should have test_evaluation_count=1: {row['run_id']}")
    main_rows = [
        row
        for row in rows
        if row["record_type"] == "experiment" and row["split"] == "test" and row["evaluation"] == "full_7111_items"
    ]
    bad_main = [row["run_id"] for row in main_rows if row["record_type"] == "sanity"]
    if bad_main:
        issues.append("Sanity rows leaked into main table: " + ", ".join(bad_main))
    return issues


def write_audit(rows: list[dict[str, Any]], notes: list[str], issues: list[str]) -> None:
    kinds = sorted({row["record_type"] for row in rows}, key=record_order)
    counts = {kind: sum(1 for row in rows if row["record_type"] == kind) for kind in kinds}
    lines = [
        "# Аудит результатов",
        "",
        f"Сгенерировано: `{datetime.now(timezone.utc).isoformat()}`.",
        "",
        "## Счётчики",
        "",
        markdown_table(["record_type", "rows"], [[kind, count] for kind, count in counts.items()]),
        "",
        "## Замечания",
        "",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- Критических несогласованностей при генерации registry не найдено.")
    lines.extend(
        [
            "- Исторические `random_001`, `mostpop_001`, `ltr_xgb_001` сохранены как `sampled_100` и исключены из основной full-ranking таблицы.",
            "- SSD4Rec paper v2 не сообщает @50 metrics; эти поля оставлены пустыми в `paper_reference`.",
            "- Validation/search rows и locked test rows представлены отдельными записями.",
            "",
            "## Проблемы валидации",
            "",
        ]
    )
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- Проблем валидации нет.")
    lines.append("")
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows, notes = collect_rows()
    issues = validate_rows(rows)
    if issues:
        write_audit(rows, notes, issues)
        raise SystemExit("Registry validation failed; see experiments/RESULTS_AUDIT.md")
    write_csv(rows)
    write_markdown(rows)
    write_audit(rows, notes, issues)
    kinds = sorted({row["record_type"] for row in rows}, key=record_order)
    counts = {kind: sum(1 for row in rows if row["record_type"] == kind) for kind in kinds}
    print(json.dumps({"rows": len(rows), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
