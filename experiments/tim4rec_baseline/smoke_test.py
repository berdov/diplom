#!/usr/bin/env python
"""Smoke test for TiM4Rec on KuaiRand Protocol B.

This script intentionally does not train the model. It verifies imports, reads the
real RecBole `.inter` file, creates sequential splits, instantiates TiM4Rec, and
performs one GPU forward pass on a real train batch.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tim4rec import TiM4Rec  # noqa: E402


def _dist_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "experiments" / "tim4rec_baseline" / "config_kuairand.yaml"),
    )
    parser.add_argument(
        "--result-json",
        default=str(
            ROOT
            / "experiments"
            / "tim4rec_baseline"
            / "runs"
            / f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        ),
    )
    parser.add_argument("--forward-batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(model=TiM4Rec, config_file_list=[args.config])
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for TiM4Rec smoke test.")

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = TiM4Rec(config, train_data.dataset).to(config["device"])
    model.eval()

    batch = next(iter(train_data)).to(config["device"])
    batch_size = min(args.forward_batch_size, len(batch))

    item_seq = batch[model.ITEM_SEQ][:batch_size]
    item_seq_len = batch[model.ITEM_SEQ_LEN][:batch_size]
    timestamp_seq = batch["timestamp_list"][:batch_size]

    with torch.no_grad():
        seq_output = model.forward(item_seq, item_seq_len, timestamp_seq)

    cuda_index = torch.cuda.current_device()
    result = {
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "recbole": _dist_version("recbole"),
            "mamba_ssm": _dist_version("mamba-ssm"),
            "causal_conv1d": _dist_version("causal-conv1d"),
        },
        "gpu": {
            "device": str(config["device"]),
            "name": torch.cuda.get_device_name(cuda_index),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(cuda_index))),
        },
        "config": {
            "config_file": str(Path(args.config).resolve()),
            "dataset": config["dataset"],
            "data_path": config["data_path"],
            "is_time": bool(config["is_time"]),
            "hidden_size": int(config["hidden_size"]),
            "num_layers": int(config["num_layers"]),
            "max_item_list_length": int(config["MAX_ITEM_LIST_LENGTH"]),
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "topk": list(config["topk"]),
            "eval_args": config["eval_args"],
        },
        "dataset": {
            "user_num_with_pad": int(dataset.user_num),
            "item_num_with_pad": int(dataset.item_num),
            "inter_num_after_recbole_processing": int(dataset.inter_num),
            "fields": list(dataset.field2type.keys()),
            "train_batches": len(train_data),
            "valid_batches": len(valid_data),
            "test_batches": len(test_data),
        },
        "forward": {
            "batch_size": batch_size,
            "item_seq_shape": list(item_seq.shape),
            "timestamp_seq_shape": list(timestamp_seq.shape),
            "seq_output_shape": list(seq_output.shape),
            "seq_output_dtype": str(seq_output.dtype),
            "seq_output_device": str(seq_output.device),
        },
    }

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
