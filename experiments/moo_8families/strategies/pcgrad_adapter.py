"""Adapter for the historical PCGrad validation-only run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_RUN_ID = "pcgrad_001"


def load_historical_pcgrad(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") != REQUIRED_RUN_ID:
        raise RuntimeError(f"Expected {REQUIRED_RUN_ID}, got {payload.get('run_id')}")
    if payload.get("status") != "completed":
        raise RuntimeError(f"PCGrad historical run is not completed: {payload.get('status')}")
    if int(payload.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError(f"Historical PCGrad touched test: {payload.get('test_evaluation_count')}")
    safety = payload.get("test_safety") or {}
    if bool(safety.get("test_evaluated")) or bool(safety.get("test_dataset_loaded")):
        raise RuntimeError(f"Historical PCGrad test safety failed: {safety}")
    dataset = payload.get("dataset") or {}
    if dataset.get("protocol") != "B":
        raise RuntimeError(f"Historical PCGrad protocol mismatch: {dataset.get('protocol')}")
    best = payload.get("best_validation_metrics") or payload.get("best_validation", {}).get("metrics")
    if not best or "NDCG@10" not in best:
        raise RuntimeError("Historical PCGrad has no validation NDCG@10.")
    aux = payload.get("best_auxiliary_metrics") or payload.get("best_validation", {}).get("auxiliary_validation")
    required_aux = ("is_click", "long_view", "is_like", "is_profile_enter")
    if not aux or any(target not in aux or "bce_loss" not in aux[target] for target in required_aux):
        raise RuntimeError("Historical PCGrad has no complete auxiliary validation BCE metrics.")
    return {
        "run_id": REQUIRED_RUN_ID,
        "source_json": str(path),
        "status": "historical_comparable_validation_only",
        "method": payload.get("method"),
        "best_epoch": payload.get("best_epoch"),
        "actual_epochs": payload.get("actual_epochs"),
        "best_validation_metrics": best,
        "best_auxiliary_metrics": aux,
        "test_evaluation_count": 0,
        "slurm": payload.get("slurm"),
        "limitation": "Existing validation-only result is reused; no automatic rerun in this benchmark branch.",
    }
