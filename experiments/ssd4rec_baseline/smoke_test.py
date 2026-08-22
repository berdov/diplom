#!/usr/bin/env python
"""Smoke test for SSD4Rec on KuaiRand Protocol B.

The script intentionally does not run full training. It verifies imports,
loads the real RecBole `.inter` file, builds SSD4Rec custom variable-length
dataloaders, performs one train loss/backward/optimizer step on GPU, and
checks one full-ranking validation batch.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from recbole.config import Config
from recbole.utils import init_seed


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ROOT / "experiments" / "ssd4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import custom_utils as ssd4rec_custom_utils  # noqa: E402
from custom_trainer import SSD4RecTrainer  # noqa: E402
from custom_utils import SSD4RecData_preparation, SSD4RecDataset  # noqa: E402
from ssd4rec import SSD4Rec  # noqa: E402


ssd4rec_custom_utils.getLogger = logging.getLogger


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


def _to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in batch)


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    norms = [
        param.grad.detach().float().norm()
        for param in parameters
        if param.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "experiments" / "ssd4rec_baseline" / "config_kuairand.yaml"),
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "outputs" / "data" / "protocol_b_manifest.json"),
    )
    parser.add_argument(
        "--result-json",
        default=str(
            ROOT
            / "experiments"
            / "ssd4rec_baseline"
            / "runs"
            / f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        config = Config(model=SSD4Rec, config_file_list=[args.config])
    finally:
        sys.argv = original_argv
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for SSD4Rec smoke test.")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["protocol_sources"]["ssd4rec"]["paper_fingerprint"]

    dataset = SSD4RecDataset(config)
    train_data, valid_data, test_data = SSD4RecData_preparation(config, dataset)

    user_num_without_pad = int(dataset.user_num - 1)
    item_num_without_pad = int(dataset.item_num - 1)
    expected_examples = int(expected["interactions"] - expected["users"])
    if user_num_without_pad != int(expected["users"]):
        raise AssertionError(f"Unexpected user count: {user_num_without_pad} != {expected['users']}")
    if item_num_without_pad != int(expected["items"]):
        raise AssertionError(f"Unexpected item count: {item_num_without_pad} != {expected['items']}")
    if int(dataset.inter_num) != expected_examples:
        raise AssertionError(f"Unexpected sequential examples: {dataset.inter_num} != {expected_examples}")

    device = config["device"]
    model = SSD4Rec(config, train_data.dataset).to(device)
    trainer = SSD4RecTrainer(config, model)

    model.train()
    train_batch = _to_device(next(iter(train_data)), device)
    item_id, item_id_list, cum_item_length, item_idx, flip_index = train_batch
    seq_lengths = torch.diff(
        cum_item_length,
        prepend=torch.zeros(1, dtype=cum_item_length.dtype, device=cum_item_length.device),
    )

    probe_param = next(param for param in model.parameters() if param.requires_grad)
    probe_before = probe_param.detach().clone()

    trainer.optimizer.zero_grad(set_to_none=True)
    loss = model.calculate_loss(item_id, item_id_list, cum_item_length, item_idx, flip_index)
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite train loss: {loss.item()}")
    loss.backward()
    grad_norm = _grad_norm(list(model.parameters()))
    trainer.optimizer.step()
    update_max_abs = float((probe_param.detach() - probe_before).abs().max().item())

    if update_max_abs <= 0.0:
        raise AssertionError("Optimizer step did not update the probed parameter.")
    if int(cum_item_length[-1].item()) != int(item_id_list.numel()):
        raise AssertionError("cum_item_length does not match concatenated item list length.")
    if int(item_idx.numel()) != int(item_id_list.numel()):
        raise AssertionError("item_idx length does not match concatenated item list length.")
    if int(flip_index.numel()) != int(item_id_list.numel()):
        raise AssertionError("flip_index length does not match concatenated item list length.")

    model.eval()
    valid_batch = _to_device(next(iter(valid_data)), device)
    valid_item_id_list, valid_cum_item_length, valid_item_idx, valid_flip_index, positive_u, positive_i = valid_batch
    with torch.no_grad():
        scores = model.full_sort_predict(
            valid_item_id_list,
            valid_cum_item_length,
            valid_item_idx,
            valid_flip_index,
        )
        scores = scores.view(-1, model.n_items)
        scores[:, 0] = -torch.inf
        positive_scores = scores[positive_u, positive_i]
        topk = min(max(config["topk"]), scores.shape[1] - 1)
        topk_index = torch.topk(scores, k=topk, dim=1).indices
        hits_at_topk = (topk_index == positive_i.unsqueeze(1)).any(dim=1).sum()

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
            "numpy": np.__version__,
            "pandas": _dist_version("pandas"),
        },
        "gpu": {
            "device": str(device),
            "name": torch.cuda.get_device_name(cuda_index),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(cuda_index))),
        },
        "config": {
            "config_file": str(Path(args.config).resolve()),
            "manifest": str(manifest_path.resolve()),
            "dataset": config["dataset"],
            "data_path": config["data_path"],
            "hidden_size": int(config["hidden_size"]),
            "d_state": int(config["d_state"]),
            "d_conv": int(config["d_conv"]),
            "expand": int(config["expand"]),
            "headdim": int(config["headdim"]),
            "beta": float(config["beta"]),
            "maskratio": float(config["maskratio"]),
            "num_layers": int(config["num_layers"]),
            "var_len": bool(config["var_len"]),
            "max_item_list_length": int(config["MAX_ITEM_LIST_LENGTH"]),
            "train_batch_size": int(config["train_batch_size"]),
            "eval_batch_size": int(config["eval_batch_size"]),
            "topk": list(config["topk"]),
            "eval_args": config["eval_args"],
        },
        "dataset": {
            "manifest_users": int(expected["users"]),
            "manifest_items": int(expected["items"]),
            "manifest_interactions": int(expected["interactions"]),
            "user_num_without_pad": user_num_without_pad,
            "item_num_without_pad": item_num_without_pad,
            "inter_num_after_ssd4rec_augmentation": int(dataset.inter_num),
            "expected_sequential_examples": expected_examples,
            "train_batches": len(train_data),
            "valid_batches": len(valid_data),
            "test_batches": len(test_data),
        },
        "train_batch": {
            "targets": int(item_id.numel()),
            "flat_sequence_tokens": int(item_id_list.numel()),
            "cum_item_length_last": int(cum_item_length[-1].item()),
            "item_idx_tokens": int(item_idx.numel()),
            "flip_index_tokens": int(flip_index.numel()),
            "sequence_length_min": int(seq_lengths.min().item()),
            "sequence_length_max": int(seq_lengths.max().item()),
            "sequence_length_mean": float(seq_lengths.float().mean().item()),
            "mask_zero_tokens": int((item_id_list == 0).sum().item()),
            "mask_zero_ratio": float((item_id_list == 0).float().mean().item()),
        },
        "optimizer_step": {
            "loss": float(loss.detach().item()),
            "grad_norm": grad_norm,
            "probe_update_max_abs": update_max_abs,
            "optimizer": config["learner"],
            "learning_rate": float(config["learning_rate"]),
        },
        "validation_batch": {
            "batch_users": int(positive_u.numel()),
            "scores_shape": list(scores.shape),
            "scores_dtype": str(scores.dtype),
            "finite_positive_scores": int(torch.isfinite(positive_scores).sum().item()),
            "hits_at_max_topk": int(hits_at_topk.item()),
            "max_topk_checked": int(topk),
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
