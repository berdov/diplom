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
BASE_TRAIN_METHODS = ("stch", "famo", "epo", "gradhv", "phn", "cosmos", "palora")
CONVERGENCE_METHODS = ("stch", "famo", "pcgrad", "epo", "gradhv", "phn", "cosmos", "palora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "sanity", "historical", "convergence_screening"), default="smoke")
    parser.add_argument("--method", choices=(*CONVERGENCE_METHODS, "all"), default="all")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--slurm", default=str(ROOT / "slurm" / "moo_8families.sh"))
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        value = subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return value or default
    except Exception:
        return default


def result_path(config: dict[str, Any], run_id: str) -> Path:
    path = Path(config["run"]["local_runs_dir"])
    if not path.is_absolute():
        path = ROOT / path
    return path / f"{run_id}.json"


def default_run_id(config: dict[str, Any], method: str, stage: str) -> str:
    if method == "pcgrad" and stage == "historical":
        return str(config["methods"]["pcgrad"]["historical_run_id"])
    key_by_stage = {
        "smoke": "smoke_run_id",
        "sanity": "sanity_run_id",
        "convergence_screening": "convergence_run_id",
    }
    key = key_by_stage[stage]
    return str(config["methods"][method][key])


def check_smoke_gate(config: dict[str, Any]) -> None:
    missing = []
    failed = []
    for method in BASE_TRAIN_METHODS:
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


def check_validation_gate(config: dict[str, Any]) -> None:
    missing = []
    failed = []
    for method in BASE_TRAIN_METHODS:
        run_id = default_run_id(config, method, "sanity")
        path = result_path(config, run_id)
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or int(payload.get("test_evaluation_count", -1)) != 0:
            failed.append(
                {
                    "path": str(path),
                    "status": payload.get("status"),
                    "test_evaluation_count": payload.get("test_evaluation_count"),
                }
            )
    pcgrad_path = result_path(config, default_run_id(config, "pcgrad", "historical"))
    if not pcgrad_path.exists():
        missing.append(str(pcgrad_path))
    else:
        payload = json.loads(pcgrad_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or int(payload.get("test_evaluation_count", -1)) != 0:
            failed.append(
                {
                    "path": str(pcgrad_path),
                    "status": payload.get("status"),
                    "test_evaluation_count": payload.get("test_evaluation_count"),
                }
            )
    if missing or failed:
        raise RuntimeError(f"Validation gate failed: missing={missing}, failed={failed}")


def command_for(slurm: Path, method: str, stage: str, run_id: str) -> list[str]:
    export_values = {
        "MOO_METHOD": method,
        "MOO_STAGE": stage,
        "MOO_RUN_ID": run_id,
        "REPO_DIR": str(ROOT),
        "MOO_GIT_COMMIT": git_value(["rev-parse", "HEAD"]),
        "MOO_GIT_BRANCH": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "MOO_GIT_REMOTE": git_value(["config", "--get", "remote.origin.url"]),
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
    command = ["sbatch", f"--export={export}"]
    if stage == "convergence_screening":
        logs_dir = ROOT / "experiments" / "moo_8families" / "slurm_logs"
        command.extend(
            [
                "--partition=rocky",
                "--constraint=type_e",
                "--gres=gpu:a100:1",
                "--time=24:00:00",
                f"--job-name=moo8-{method}-conv",
                f"--output={logs_dir}/%x-%j.out",
                f"--error={logs_dir}/%x-%j.err",
            ]
        )
    return [*command, str(slurm)]


def methods_for_stage(stage: str, requested: str) -> list[str]:
    if stage == "historical":
        return ["pcgrad"]
    if requested != "all":
        return [requested]
    if stage == "convergence_screening":
        return list(CONVERGENCE_METHODS)
    return list(BASE_TRAIN_METHODS)


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    slurm = Path(args.slurm)
    if args.stage == "sanity":
        check_smoke_gate(config)
    if args.stage == "convergence_screening":
        check_validation_gate(config)
    methods = methods_for_stage(args.stage, args.method)
    for method in methods:
        if args.stage == "historical":
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
