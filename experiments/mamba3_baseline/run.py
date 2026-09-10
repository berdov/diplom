#!/usr/bin/env python
"""Validation-only runner for the vanilla Mamba-3 sequential baseline.

No TEST metrics are produced by this script. Model selection is VALID NDCG@10.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from model import Mamba3Rec


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments" / "mamba3_baseline" / "config_kuairand.yaml"
DEFAULT_RUNS = ROOT / "experiments" / "mamba3_baseline" / "runs"

EXPECTED_PROTOCOL = {
    "users": 23951,
    "items": 7111,
    "interactions": 1134420,
    "train": 1086518,
    "validation": 23951,
    "test": 23951,
    "recbole_inter_sha256": "e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc",
}
MAMBA3_UPSTREAM_REPO = "https://github.com/state-spaces/mamba"
MAMBA3_UPSTREAM_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument(
        "--skip-data-sha-check",
        action="store_true",
        help="Skip sha256 of the 26.5 MiB RecBole .inter file.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def verify_protocol(config: Config, *, check_sha: bool) -> dict[str, Any]:
    manifest_path = ROOT / "outputs" / "data" / "protocol_b_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    actual = {
        "users": int(manifest["filtered_stats"]["users"]),
        "items": int(manifest["filtered_stats"]["items"]),
        "interactions": int(manifest["filtered_stats"]["interactions"]),
        "train": int(manifest["split_stats"]["train"]["interactions"]),
        "validation": int(manifest["split_stats"]["validation"]["interactions"]),
        "test": int(manifest["split_stats"]["test"]["interactions"]),
    }
    for key in ("users", "items", "interactions", "train", "validation", "test"):
        if actual[key] != EXPECTED_PROTOCOL[key]:
            raise RuntimeError(
                f"Protocol B mismatch for {key}: expected {EXPECTED_PROTOCOL[key]}, got {actual[key]}"
            )

    inter_path = Path(str(config["data_path"])) / str(config["dataset"]) / f"{config['dataset']}.inter"
    actual_sha = None
    if check_sha:
        if not inter_path.exists():
            raise FileNotFoundError(f"Protocol B RecBole file not found: {inter_path}")
        actual_sha = _sha256(inter_path)
        if actual_sha != EXPECTED_PROTOCOL["recbole_inter_sha256"]:
            raise RuntimeError(
                "Protocol B .inter sha256 mismatch: "
                f"expected {EXPECTED_PROTOCOL['recbole_inter_sha256']}, got {actual_sha}"
            )

    return {
        **actual,
        "manifest_path": str(manifest_path),
        "recbole_inter_path": str(inter_path),
        "recbole_inter_sha256": actual_sha,
        "sha_checked": bool(check_sha),
    }


def runtime_info() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return {
        "python": os.sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "recbole": _package_version("recbole"),
        "mamba_ssm": _package_version("mamba-ssm"),
        "triton": _package_version("triton"),
        "tilelang": _package_version("tilelang"),
    }


def _make_result_path(args: argparse.Namespace, run_id: str) -> Path:
    if args.result_json:
        return Path(args.result_json)
    return DEFAULT_RUNS / f"{run_id}.json"


def _base_result(
    *,
    args: argparse.Namespace,
    run_id: str,
    config: Config,
    protocol: dict[str, Any],
    model: Mamba3Rec,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "selection_split": "VALID",
        "primary_task": "next_item",
        "valid_metric": str(config["valid_metric"]),
        "test_evaluation_count": 0,
        "test_result": None,
        "protocol": protocol,
        "model": {
            "name": "Mamba3Rec",
            "role": "vanilla Mamba-3 sequential baseline",
            "hidden_size": int(config["hidden_size"]),
            "num_layers": int(config["num_layers"]),
            "dropout_prob": float(config["dropout_prob"]),
            "use_ffn": bool(config["use_ffn"]),
            "ffn_inner_size": int(config["ffn_inner_size"]),
            "mamba3_d_state": int(config["mamba3_d_state"]),
            "mamba3_expand": int(config["mamba3_expand"]),
            "mamba3_headdim": int(config["mamba3_headdim"]),
            "mamba3_ngroups": int(config["mamba3_ngroups"]),
            "mamba3_rope_fraction": float(config["mamba3_rope_fraction"]),
            "mamba3_chunk_size": int(config["mamba3_chunk_size"]),
            "mamba3_is_mimo": bool(config["mamba3_is_mimo"]),
            "mamba3_mimo_rank": int(config["mamba3_mimo_rank"]),
            "mamba3_is_outproj_norm": bool(config["mamba3_is_outproj_norm"]),
            "mamba3_dtype": str(config["mamba3_dtype"]),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
        },
        "upstream": {
            "repo": MAMBA3_UPSTREAM_REPO,
            "pinned_commit": MAMBA3_UPSTREAM_COMMIT,
        },
        "runtime": runtime_info(),
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "remote": _git_value("config", "--get", "remote.origin.url"),
        },
    }


def smoke(
    *,
    config: Config,
    train_data,
    model: Mamba3Rec,
) -> dict[str, Any]:
    model.train()
    try:
        batch = next(iter(train_data))
    except StopIteration as exc:
        raise RuntimeError("Training dataloader is empty") from exc

    batch = batch.to(config["device"])
    loss = model.calculate_loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke loss: {loss.item()}")

    loss.backward()
    finite_gradients = True
    gradient_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient_tensors += 1
            if not torch.isfinite(parameter.grad).all():
                finite_gradients = False
                break

    model.zero_grad(set_to_none=True)
    if not finite_gradients:
        raise RuntimeError("Smoke backward produced non-finite gradients")

    return {
        "smoke_loss": float(loss.detach().cpu()),
        "gradient_tensors": gradient_tensors,
        "finite_gradients": finite_gradients,
    }


def main() -> None:
    args = parse_args()
    run_id = args.run_id or (
        "mamba3_smoke_001" if args.mode == "smoke" else "mamba3_validation_001"
    )
    result_path = _make_result_path(args, run_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Official Mamba-3 GPU kernels require CUDA; no CUDA device is available.")

    config = Config(model=Mamba3Rec, config_file_list=[args.config])
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    protocol = verify_protocol(config, check_sha=not args.skip_data_sha_check)
    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, _test_data = data_preparation(config, dataset)
    model = Mamba3Rec(config, train_data.dataset).to(config["device"])
    logger.info(model)

    result = _base_result(
        args=args,
        run_id=run_id,
        config=config,
        protocol=protocol,
        model=model,
    )

    if args.mode == "smoke":
        result.update(smoke(config=config, train_data=train_data, model=model))
        result["status"] = "ok"
    else:
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=config["show_progress"],
        )
        result.update(
            {
                "status": "ok",
                "best_valid_score": float(best_valid_score),
                "best_valid_result": best_valid_result,
                "checkpoint_dir": str(config["checkpoint_dir"]),
            }
        )

    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
