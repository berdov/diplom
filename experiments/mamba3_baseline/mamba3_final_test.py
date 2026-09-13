#!/usr/bin/env python
"""Single frozen-checkpoint TEST evaluation for Mamba3Rec."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from model import Mamba3Rec
from run import verify_protocol, _json_default, _git_value, runtime_info


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config_kuairand.yaml"
CHECKPOINT = HERE / "checkpoints" / "Mamba3Rec-Sep-12-2026_17-55-38.pth"
RUNS = HERE / "runs"
RESULT = RUNS / "mamba3_final_test_001.json"
LOCK = RUNS / ".mamba3_final_test_001.lock"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)

    if RESULT.exists():
        raise SystemExit(
            f"ABORT: final TEST already exists: {RESULT}"
        )

    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"ABORT: final TEST lock already exists: {LOCK}"
        )

    os.write(fd, f"pid={os.getpid()}\n".encode())
    os.close(fd)

    try:
        config = Config(
            model=Mamba3Rec,
            config_file_list=[str(CONFIG)],
        )

        init_seed(
            config["seed"] + config["local_rank"],
            config["reproducibility"],
        )
        init_logger(config)

        # Exact same Protocol B verification as validation training.
        protocol = verify_protocol(config, check_sha=True)

        dataset = create_dataset(config)
        train_data, _valid_data, test_data = data_preparation(config, dataset)

        loader_class = type(test_data).__name__
        if "FullSort" not in loader_class:
            raise RuntimeError(
                f"Expected FullSort TEST loader, got {loader_class}"
            )

        expected_topk = {5, 10, 20, 50}
        actual_topk = set(int(k) for k in config["topk"])
        if not expected_topk.issubset(actual_topk):
            raise RuntimeError(
                f"Expected topk {sorted(expected_topk)}, got {sorted(actual_topk)}"
            )

        model = Mamba3Rec(
            config,
            train_data.dataset,
        ).to(config["device"])

        # Trusted checkpoint produced by our own training run.
        # Explicit False is required with PyTorch >= 2.6 for this
        # RecBole checkpoint format.
        saved = torch.load(
            CHECKPOINT,
            map_location=config["device"],
            weights_only=False,
        )

        model.load_state_dict(saved["state_dict"], strict=True)

        other_parameter = saved.get("other_parameter")
        if other_parameter is not None:
            model.load_other_parameter(other_parameter)

        trainer = Trainer(config, model)

        # IMPORTANT:
        # checkpoint is already loaded above.
        # RecBole must NOT load it again with torch.load().
        test_result = trainer.evaluate(
            test_data,
            load_best_model=False,
            show_progress=config["show_progress"],
        )

        payload = {
            "run_id": "mamba3_final_test_001",
            "status": "ok",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": "Mamba3Rec",
            "split": "TEST",
            "selection_split": "VALID",
            "test_evaluation_count": 1,
            "evaluation": {
                "mode": "full-ranking",
                "loader_class": loader_class,
                "topk": sorted(actual_topk),
            },
            "checkpoint": {
                "path": str(CHECKPOINT),
                "sha256": sha256(CHECKPOINT),
                "epoch": saved.get("epoch"),
            },
            "protocol": protocol,
            "test_result": test_result,
            "runtime": runtime_info(),
            "git": {
                "commit": _git_value("rev-parse", "HEAD"),
                "branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
            },
        }

        tmp = RESULT.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, RESULT)

        print("FINAL_TEST_OK")
        print(json.dumps(test_result, indent=2, default=_json_default))
        print(f"result={RESULT}")

    finally:
        # Successful run is protected by RESULT existence.
        # A normal technical failure may therefore be retried.
        try:
            LOCK.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
