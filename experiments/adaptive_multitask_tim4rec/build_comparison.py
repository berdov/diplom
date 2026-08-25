#!/usr/bin/env python
"""Build comparison report for adaptive multitask 5-epoch sanity runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "experiments" / "adaptive_multitask_tim4rec" / "runs"
OUTPUT = RUNS_DIR / "adaptive_sanity_comparison_001.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any], key: str) -> float:
    variants = [key, key.lower(), key.replace("HR", "hit").lower(), key.replace("Recall", "recall").lower()]
    for variant in variants:
        if variant in metrics:
            return float(metrics[variant])
    raise KeyError(f"Missing metric {key} in {metrics}")


def normalize(metrics: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key in ("HR@5", "HR@10", "HR@20", "HR@50", "Recall@5", "Recall@10", "Recall@20", "Recall@50", "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@50"):
        result[key] = metric_value(metrics, key)
    return result


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def bytes_gb(value: Any) -> str:
    if value in ("", None):
        return ""
    return f"{float(value) / 1e9:.2f} GB"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def reference_rows() -> list[dict[str, Any]]:
    tim4rec = load_json(ROOT / "experiments/tim4rec_baseline/runs/tim4rec_001.json")
    fixed = load_json(ROOT / "experiments/multitask_tim4rec/runs/multitask_tim4rec_001.json")
    tuned = load_json(ROOT / "experiments/multitask_tim4rec_optuna/runs/multitask_tim4rec_tuned_001.json")
    tuned_valid = tuned["validation_reproduction"]["reproduced_validation"]
    return [
        {
            "method": "TiM4Rec reference validation",
            "run_type": "full reference",
            "best_epoch": tim4rec.get("best_epoch"),
            "metrics": normalize(tim4rec["best_validation_metrics"]),
            "epoch_time": tim4rec.get("runtime", {}).get("mean_epoch_sec"),
            "peak_vram": tim4rec.get("gpu", {}).get("peak_allocated_bytes"),
        },
        {
            "method": "Fixed Multitask reference validation",
            "run_type": "full reference",
            "best_epoch": fixed.get("best_epoch"),
            "metrics": normalize(fixed["best_validation_metrics"]),
            "epoch_time": fixed.get("runtime", {}).get("mean_epoch_sec"),
            "peak_vram": fixed.get("gpu", {}).get("peak_allocated_bytes"),
        },
        {
            "method": "Tuned fixed reference validation",
            "run_type": "full tuned reference",
            "best_epoch": tuned["validation_reproduction"].get("best_epoch"),
            "metrics": normalize(tuned_valid),
            "epoch_time": tuned["validation_reproduction"].get("runtime_sec", 0.0)
            / max(int(tuned["validation_reproduction"].get("actual_epochs") or 1), 1),
            "peak_vram": tuned.get("gpu", {}).get("peak_allocated_bytes"),
        },
    ]


def sanity_row(path: Path, label: str) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "method": label,
        "run_type": "5-epoch sanity",
        "best_epoch": payload["best_epoch"],
        "metrics": normalize(payload["best_validation_metrics"]),
        "epoch_time": payload["cost"]["mean_epoch_time_sec"],
        "peak_vram": payload["gpu"]["peak_allocated_bytes"],
        "payload": payload,
    }


def main_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        ["Method", "Run type", "Best epoch", "HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50", "epoch time", "peak VRAM"],
        [
            [
                row["method"],
                row["run_type"],
                row.get("best_epoch") or "",
                fmt(row["metrics"]["HR@10"]),
                fmt(row["metrics"]["HR@20"]),
                fmt(row["metrics"]["HR@50"]),
                fmt(row["metrics"]["NDCG@10"]),
                fmt(row["metrics"]["NDCG@20"]),
                fmt(row["metrics"]["NDCG@50"]),
                fmt(row.get("epoch_time"), 1),
                bytes_gb(row.get("peak_vram")),
            ]
            for row in rows
        ],
    )


def gradient_rows(payload: dict[str, Any], label: str) -> list[list[Any]]:
    rows = []
    for diag in payload["gradient_diagnostics"]:
        cos = diag["rank_aux_cosines_before"]
        rows.append(
            [
                label,
                diag["epoch"],
                fmt(cos["is_click"], 6),
                fmt(cos["long_view"], 6),
                fmt(cos["is_like"], 6),
                fmt(cos["is_profile_enter"], 6),
                fmt(diag["rank_aux_conflicts_before"]["fraction_conflicting"], 4),
            ]
        )
    return rows


def auxiliary_conflict_rows(payload: dict[str, Any], label: str) -> list[Any]:
    summary = payload["gradient_conflict_summary"]
    return [
        label,
        summary["aux_negative_before"],
        summary["aux_pairs_before"],
        fmt(summary["aux_fraction_before"], 4),
        summary["aux_negative_after"],
        summary["aux_pairs_after"],
        fmt(summary["aux_fraction_after"], 4),
        summary["rank_aux_negative_before"],
        summary["rank_aux_pairs_before"],
        fmt(summary["rank_aux_fraction_before"], 4),
    ]


def is_like_rows(payload: dict[str, Any], label: str) -> list[list[Any]]:
    rows = []
    for point in payload["is_like_summary"]["diagnostic_points"]:
        rows.append(
            [
                label,
                point["epoch"] if "epoch" in point else "",
                fmt(point["raw_loss"], 6),
                fmt(point["weighted_or_effective_contribution"], 6),
                fmt(point["shared_gradient_norm_before"], 6),
                fmt(point["cosine_with_ranking_before"], 6),
            ]
        )
    return rows


def build_report() -> str:
    pcgrad = sanity_row(RUNS_DIR / "pcgrad_sanity_001.json", "PCGrad sanity")
    metabalance = sanity_row(RUNS_DIR / "metabalance_sanity_001.json", "MetaBalance sanity")
    rows = reference_rows() + [pcgrad, metabalance]
    gradient = gradient_rows(pcgrad["payload"], "PCGrad") + gradient_rows(metabalance["payload"], "MetaBalance")
    auxiliary = [
        auxiliary_conflict_rows(pcgrad["payload"], "PCGrad"),
        auxiliary_conflict_rows(metabalance["payload"], "MetaBalance"),
    ]
    like = is_like_rows(pcgrad["payload"], "PCGrad") + is_like_rows(metabalance["payload"], "MetaBalance")
    return "\n\n".join(
        [
            "# Adaptive sanity comparison 001",
            f"Сгенерировано: `{datetime.now(timezone.utc).isoformat()}`.",
            "Все adaptive sanity runs обучались 5 эпох на полном train split и оценивались только на full-ranking validation. TEST не использовался.",
            "Reference rows ниже являются уже существующими full/reference runs, поэтому они не равны по бюджету 5-epoch sanity.",
            "## Main comparison",
            main_table(rows),
            "## Rank-aux diagnostic cosines",
            markdown_table(
                [
                    "Method",
                    "Diagnostic epoch",
                    "Rank-vs-click cosine",
                    "Rank-vs-long cosine",
                    "Rank-vs-like cosine",
                    "Rank-vs-profile cosine",
                    "rank-aux conflict fraction",
                ],
                gradient,
            ),
            "## Auxiliary-auxiliary conflicts",
            markdown_table(
                [
                    "Method",
                    "aux negatives before",
                    "aux pairs before",
                    "aux fraction before",
                    "aux negatives after",
                    "aux pairs after",
                    "aux fraction after",
                    "rank-aux negatives before",
                    "rank-aux pairs before",
                    "rank-aux fraction before",
                ],
                auxiliary,
            ),
            "## is_like diagnostics",
            markdown_table(
                [
                    "Method",
                    "Diagnostic epoch",
                    "like raw loss",
                    "like effective contribution",
                    "like shared grad norm",
                    "like cosine with ranking",
                ],
                like,
            ),
        ]
    ) + "\n"


def main() -> None:
    OUTPUT.write_text(build_report(), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
