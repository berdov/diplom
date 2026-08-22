#!/usr/bin/env python3
"""Guarded entrypoint for the future final Optuna-selected LambdaMART run.

This script is intentionally not used during the smoke stage. It exists to make
the future final-test step explicit and hard to trigger accidentally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "experiments/ltr_xgb_optuna/config.yaml"))
    parser.add_argument("--allow-test-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    storage_path = Path(config["study_storage"])
    summary = {
        "study_name": config["study_name"],
        "storage_path": str(storage_path),
        "base_run_id": config["base_run_id"],
        "test_policy": config["test_policy"],
        "ready_for_final_test": bool(args.allow_test_evaluation),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return
    if not args.allow_test_evaluation:
        raise SystemExit(
            "Final test is locked for the Optuna preparation stage. "
            "Run full search, freeze best hyperparameters, then rerun with --allow-test-evaluation."
        )

    raise SystemExit("Final best-model training/evaluation is intentionally not implemented in this preparation commit.")


if __name__ == "__main__":
    main()
