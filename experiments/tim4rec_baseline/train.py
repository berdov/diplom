#!/usr/bin/env python
"""Full TiM4Rec runner for future training runs.

Do not use this script for the current preparation stage unless full training is
explicitly requested. The smoke test is `smoke_test.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.trainer import Trainer
from recbole.utils import get_environment, get_flops, init_logger, init_seed, set_color


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ROOT / "experiments" / "tim4rec_baseline" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tim4rec import TiM4Rec  # noqa: E402


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
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
            / f"train_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(model=TiM4Rec, config_file_list=[args.config])
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = TiM4Rec(config, train_data.dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    trainer = Trainer(config, model)
    if config["checkpoint_path"] is not None:
        trainer.resume_checkpoint(config["checkpoint_path"])

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        show_progress=config["show_progress"],
    )
    test_result = trainer.evaluate(test_data, show_progress=config["show_progress"])

    logger.info("The running environment of this training is as follows:\n" + get_environment(config).draw())
    logger.info(set_color("best valid", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(Path(args.config).resolve()),
        "is_time": bool(config["is_time"]),
        "learning_rate": float(config["learning_rate"]),
        "best_valid_score": float(best_valid_score),
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "flops": flops,
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
