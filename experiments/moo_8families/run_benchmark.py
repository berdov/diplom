#!/usr/bin/env python
"""Submit or print MOO eight-family benchmark jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "moo_8families" / "config.yaml"
TRAIN_METHODS = ("stch", "famo", "epo", "gradhv", "phn", "cosmos", "palora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "sanity", "historical"), default="smoke")
    parser.add_argument("--method", choices=(*TRAIN_METHODS, "pcgrad", "all"), default="all")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--slurm", default=str(ROOT / "slurm" / "moo_8families.sh"))
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def result_path(config: dict[str, Any], run_id: str) -> Path:
    path = Path(config["run"]["local_runs_dir"])
    if not path.is_absolute():
        path = ROOT / path
    return path / f"{run_id}.json"


def default_run_id(config: dict[str, Any], method: str, stage: str) -> str:
    if method == "pcgrad":
        return str(config["methods"]["pcgrad"]["historical_run_id"])
    key = "smoke_run_id" if stage == "smoke" else "sanity_run_id"
    return str(config["methods"][method][key])


def check_smoke_gate(config: dict[str, Any]) -> None:
    missing = []
    failed = []
    for method in TRAIN_METHODS:
        run_id = default_run_id(config, method, "smoke")
        path = result_path(config, run_id)
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or int(payload.get("test_evaluation_count", -1)) != 0:
            failed.append({"path": str(path), "status": payload.get("status"), "test_evaluation_count": payload.get("test_evaluation_count")})
    if missing or failed:
        raise RuntimeError(f"Smoke gate failed: missing={missing}, failed={failed}")


def command_for(slurm: Path, method: str, stage: str, run_id: str) -> list[str]:
    export = f"ALL,MOO_METHOD={method},MOO_STAGE={stage},MOO_RUN_ID={run_id}"
    return ["sbatch", f"--export={export}", str(slurm)]


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    slurm = Path(args.slurm)
    if args.stage == "sanity":
        check_smoke_gate(config)
    methods = list(TRAIN_METHODS if args.method == "all" else [args.method])
    if args.stage == "historical":
        methods = ["pcgrad"]
    for method in methods:
        if method == "pcgrad":
            cmd = [
                sys.executable,
                "-m",
                "experiments.moo_8families.train",
                "--method",
                "pcgrad",
                "--stage",
                "historical",
            ]
        else:
            run_id = default_run_id(config, method, args.stage)
            cmd = command_for(slurm, method, args.stage, run_id)
        print(" ".join(cmd), flush=True)
        if args.submit:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

