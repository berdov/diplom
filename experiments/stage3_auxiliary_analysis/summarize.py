"""Build compact Stage 3 reports from completed validation-only artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABLATION_KEYS = ("primary_only", "click", "long_view", "like", "profile_enter")
DIAGNOSTIC_KEY = "all_current_aux_diagnostic"
REPORT_PATH = PROJECT_ROOT / "reports" / "STAGE3_AUXILIARY_ANALYSIS.md"
SUMMARY_PATH = PROJECT_ROOT / "experiments" / "stage3_auxiliary_analysis" / "stage3_summary.json"
RESULTS_CSV = PROJECT_ROOT / "experiments" / "results.csv"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel_path(path: str | Path) -> str:
    resolved = project_path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def load_json(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = project_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def fmt(value: Any, digits: int = 4, leading_zero: bool = True) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    text = f"{number:.{digits}f}"
    if not leading_zero and text.startswith("0."):
        return text[1:]
    if not leading_zero and text.startswith("-0."):
        return "-" + text[2:]
    return text


def metric(metrics: Mapping[str, Any], key: str) -> float | None:
    if key in metrics:
        return float(metrics[key])
    return None


def load_stage3_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    runs_dir = project_path(config["outputs"]["runs_dir"])
    runs: dict[str, Any] = {}
    for key in (*ABLATION_KEYS, DIAGNOSTIC_KEY):
        run_id = config["runs"][key]["run_id"]
        path = runs_dir / f"{run_id}.json"
        payload = load_json(path)
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"Run is not COMPLETE: {path}")
        if int(payload.get("test_evaluation_count", -1)) != 0:
            raise RuntimeError(f"TEST was used in {path}")
        runs[key] = payload

    audit_id = config["outputs"]["target_audit_run_id"]
    audit_path = runs_dir / f"{audit_id}.json"
    audit = load_json(audit_path)
    if int(audit.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError(f"TEST was used in {audit_path}")
    return {"runs": runs, "target_audit": audit}


def ablation_rows(runs: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary = runs["primary_only"]["best_validation_metrics"]
    primary_ndcg10 = float(primary["NDCG@10"])
    rows: list[dict[str, Any]] = []
    for key in ABLATION_KEYS:
        run = runs[key]
        metrics = run["best_validation_metrics"]
        active_targets = run["objective"]["active_auxiliary_targets"]
        rows.append(
            {
                "run_key": key,
                "run_id": run["run_id"],
                "auxiliary": "none" if not active_targets else active_targets[0],
                "best_epoch": run["best_epoch"],
                "actual_epochs": run["optimization"]["actual_epochs"],
                "HR@10": metric(metrics, "HR@10"),
                "HR@20": metric(metrics, "HR@20"),
                "HR@50": metric(metrics, "HR@50"),
                "NDCG@10": metric(metrics, "NDCG@10"),
                "NDCG@20": metric(metrics, "NDCG@20"),
                "NDCG@50": metric(metrics, "NDCG@50"),
                "delta_NDCG@10_vs_primary": float(metrics["NDCG@10"]) - primary_ndcg10,
                "test_evaluation_count": run["test_evaluation_count"],
            }
        )
    return rows


def gradient_rows(runs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ABLATION_KEYS:
        if key == "primary_only":
            continue
        run = runs[key]
        active_targets = run["objective"]["active_auxiliary_targets"]
        if not active_targets:
            continue
        target = active_targets[0]
        summary = run["gradient_diagnostics"]["summary"]["per_auxiliary_vs_primary"].get(target, {})
        rows.append(
            {
                "auxiliary": target,
                "diagnostic_batches": summary.get("diagnostic_batches"),
                "median_norm_ratio": (summary.get("norm_ratio_to_primary") or {}).get("median"),
                "median_cosine": (summary.get("cosine_to_primary") or {}).get("median"),
                "q25_cosine": (summary.get("cosine_to_primary") or {}).get("q25"),
                "q75_cosine": (summary.get("cosine_to_primary") or {}).get("q75"),
                "conflict_fraction": summary.get("conflict_fraction_with_primary"),
            }
        )
    return rows


def pairwise_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairwise = run["gradient_diagnostics"]["summary"]["auxiliary_pairwise"]
    rows = []
    for pair, payload in sorted(pairwise.items()):
        left, right = pair.split("|", 1)
        rows.append(
            {
                "left": left,
                "right": right,
                "diagnostic_batches": payload.get("diagnostic_batches"),
                "mean_cosine": (payload.get("cosine") or {}).get("mean"),
                "median_cosine": (payload.get("cosine") or {}).get("median"),
                "conflict_fraction": payload.get("conflict_fraction"),
            }
        )
    return rows


def target_audit_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in audit["target_audit"]:
        if item["type"] == "binary":
            distribution = f"pos={int(item['train_positive_count'])}, rate={fmt(item['train_positive_rate'])}"
        else:
            dist = item.get("basic_distribution_train") or {}
            distribution = (
                f"mean={fmt(dist.get('mean'))}, median={fmt(dist.get('median'))}, "
                f"p90={fmt(dist.get('p90'))}"
            )
        rows.append(
            {
                "signal": item["display_name"],
                "field": item["target"],
                "type": item["type"],
                "current": "yes" if item["currently_used"] else "no",
                "ablated": "yes" if item["stage3_ablation_enabled"] else "no",
                "train_observations": item["train_observations"],
                "train_distribution": distribution,
                "missing_rate": item["missing_rate_train"],
                "leakage_note": item["leakage_note"],
            }
        )
    return rows


def primary_aware_assessment(ablation: Sequence[Mapping[str, Any]], gradients: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delta_by_aux = {
        row["auxiliary"]: row["delta_NDCG@10_vs_primary"]
        for row in ablation
        if row["auxiliary"] != "none"
    }
    cosine_by_aux = {row["auxiliary"]: row["median_cosine"] for row in gradients}
    aligned = [
        (float(delta_by_aux[target]), float(cosine_by_aux[target]))
        for target in sorted(delta_by_aux)
        if cosine_by_aux.get(target) is not None
    ]
    correlation = None
    if len(aligned) >= 2:
        deltas = [item[0] for item in aligned]
        cosines = [item[1] for item in aligned]
        mean_delta = sum(deltas) / len(deltas)
        mean_cosine = sum(cosines) / len(cosines)
        numerator = sum((d - mean_delta) * (c - mean_cosine) for d, c in aligned)
        den_delta = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas))
        den_cosine = math.sqrt(sum((c - mean_cosine) ** 2 for c in cosines))
        if den_delta and den_cosine:
            correlation = numerator / (den_delta * den_cosine)

    helpful = [target for target, delta in delta_by_aux.items() if delta > 0.0]
    harmful = [target for target, delta in delta_by_aux.items() if delta < 0.0]
    if correlation is None:
        support = "insufficient_gradient_variation"
    elif correlation > 0.5:
        support = "directionally_supported"
    elif correlation < -0.5:
        support = "not_supported"
    else:
        support = "weak_or_mixed"
    return {
        "median_cosine_delta_pearson": correlation,
        "helpful_auxiliary_targets": helpful,
        "harmful_auxiliary_targets": harmful,
        "primary_aware_hypothesis_support": support,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def build_markdown(summary: Mapping[str, Any]) -> str:
    audit_rows = summary["target_audit_rows"]
    ablations = summary["ablation_rows"]
    gradients = summary["gradient_rows"]
    pairwise = summary["auxiliary_pairwise_rows"]
    assessment = summary["primary_aware_assessment"]

    lines = [
        "# Stage 3 Auxiliary-Task Analysis",
        "",
        "KuaiRand Protocol B, validation-only. TEST was not used.",
        "",
        "## Target Audit",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Signal", "Field", "Type", "Current", "Ablated", "Train distribution", "Missing rate"],
            [
                [
                    row["signal"],
                    f"`{row['field']}`",
                    row["type"],
                    row["current"],
                    row["ablated"],
                    row["train_distribution"],
                    fmt(row["missing_rate"]),
                ]
                for row in audit_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Single-Auxiliary Ablations",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Run",
                "Auxiliary",
                "Best epoch",
                "HR@10",
                "HR@20",
                "HR@50",
                "NDCG@10",
                "NDCG@20",
                "NDCG@50",
                "Delta NDCG@10",
                "Test evals",
            ],
            [
                [
                    f"`{row['run_id']}`",
                    f"`{row['auxiliary']}`",
                    row["best_epoch"],
                    fmt(row["HR@10"], leading_zero=False),
                    fmt(row["HR@20"], leading_zero=False),
                    fmt(row["HR@50"], leading_zero=False),
                    fmt(row["NDCG@10"], leading_zero=False),
                    fmt(row["NDCG@20"], leading_zero=False),
                    fmt(row["NDCG@50"], leading_zero=False),
                    fmt(row["delta_NDCG@10_vs_primary"], leading_zero=False),
                    row["test_evaluation_count"],
                ]
                for row in ablations
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Gradient Diagnostics",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Auxiliary", "Batches", "Median norm ratio", "Median cosine", "Q25 cosine", "Q75 cosine", "Conflict fraction"],
            [
                [
                    f"`{row['auxiliary']}`",
                    row["diagnostic_batches"],
                    fmt(row["median_norm_ratio"]),
                    fmt(row["median_cosine"]),
                    fmt(row["q25_cosine"]),
                    fmt(row["q75_cosine"]),
                    fmt(row["conflict_fraction"]),
                ]
                for row in gradients
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Auxiliary-Auxiliary Gradient Matrix",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Left", "Right", "Batches", "Mean cosine", "Median cosine", "Conflict fraction"],
            [
                [
                    f"`{row['left']}`",
                    f"`{row['right']}`",
                    row["diagnostic_batches"],
                    fmt(row["mean_cosine"]),
                    fmt(row["median_cosine"]),
                    fmt(row["conflict_fraction"]),
                ]
                for row in pairwise
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Primary-only NDCG@10: `{fmt(summary['primary_ndcg10'])}`.",
            f"- Helpful auxiliary targets by validation NDCG@10 delta: `{assessment['helpful_auxiliary_targets']}`.",
            f"- Harmful auxiliary targets by validation NDCG@10 delta: `{assessment['harmful_auxiliary_targets']}`.",
            f"- Median gradient-cosine/delta Pearson correlation: `{fmt(assessment['median_cosine_delta_pearson'])}`.",
            f"- Primary-aware hypothesis support: `{assessment['primary_aware_hypothesis_support']}`.",
            "- Additional KuaiRand targets were audited but not included in this first single-auxiliary pass when they required a new objective family or a model-scope change.",
            "",
            "## Test Hygiene",
            "",
            "- `test_evaluation_count = 0` for every Stage 3 artifact.",
            "- No TEST metrics are present in Stage 3 artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def result_registry_rows(config: Mapping[str, Any], runs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in (*ABLATION_KEYS, DIAGNOSTIC_KEY):
        run = runs[key]
        metrics = run["best_validation_metrics"]
        active_targets = run["objective"]["active_auxiliary_targets"]
        variant = "primary_only" if not active_targets else "aux_" + "_".join(active_targets)
        if key == DIAGNOSTIC_KEY:
            variant = "all_current_aux_diagnostic"
        rows.append(
            {
                "record_type": "stage3_auxiliary_analysis",
                "source": "ours",
                "run_id": run["run_id"],
                "model": "MultitaskTiM4Rec",
                "model_variant": variant,
                "dataset": "KuaiRand",
                "protocol": "B",
                "split": "validation",
                "evaluation": "full_7111_items",
                "status": "completed",
                "parent_run": "multitask_tim4rec_tuned_001",
                "source_paper": "",
                "paper_version": "",
                "seed": str(run["optimization"]["seed"]),
                "train_candidates": "full_sequence",
                "item_universe": "7111",
                "HR@5": value_or_blank(metrics, "HR@5"),
                "HR@10": value_or_blank(metrics, "HR@10"),
                "HR@20": value_or_blank(metrics, "HR@20"),
                "HR@50": value_or_blank(metrics, "HR@50"),
                "Recall@5": value_or_blank(metrics, "Recall@5"),
                "Recall@10": value_or_blank(metrics, "Recall@10"),
                "Recall@20": value_or_blank(metrics, "Recall@20"),
                "Recall@50": value_or_blank(metrics, "Recall@50"),
                "NDCG@5": value_or_blank(metrics, "NDCG@5"),
                "NDCG@10": value_or_blank(metrics, "NDCG@10"),
                "NDCG@20": value_or_blank(metrics, "NDCG@20"),
                "NDCG@50": value_or_blank(metrics, "NDCG@50"),
                "best_epoch": str(run["best_epoch"]),
                "actual_epochs": str(run["optimization"]["actual_epochs"]),
                "validation_ndcg10": value_or_blank(metrics, "NDCG@10"),
                "test_evaluation_count": "0",
                "git_commit": run["git"]["commit"],
                "notes_path": rel_path(REPORT_PATH),
                "trials_complete": "",
                "trials_pruned": "",
                "trials_failed": "",
                "best_trial": "",
                "best_validation_NDCG@10": value_or_blank(metrics, "NDCG@10"),
                "test_used": "no",
                "source_json": rel_path(Path(config["outputs"]["runs_dir"]) / f"{run['run_id']}.json"),
            }
        )
    return rows


def value_or_blank(metrics: Mapping[str, Any], key: str) -> str:
    value = metrics.get(key)
    return "" if value is None else str(float(value))


def update_results_csv(new_rows: Sequence[Mapping[str, Any]]) -> None:
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if header is None:
            raise RuntimeError(f"Missing header in {RESULTS_CSV}")
        existing = list(reader)
    new_run_ids = {row["run_id"] for row in new_rows}
    kept = [row for row in existing if row.get("run_id") not in new_run_ids]
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in kept:
            writer.writerow(row)
        for row in new_rows:
            writer.writerow({field: row.get(field, "") for field in header})


def build_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = load_stage3_artifacts(config)
    runs = artifacts["runs"]
    target_audit = artifacts["target_audit"]
    ablations = ablation_rows(runs)
    gradients = gradient_rows(runs)
    pairwise = pairwise_rows(runs[DIAGNOSTIC_KEY])
    audit_rows = target_audit_rows(target_audit)
    assessment = primary_aware_assessment(ablations, gradients)
    return {
        "run_group": config["stage3"]["run_group"],
        "status": "COMPLETE",
        "primary_ndcg10": runs["primary_only"]["best_validation_metrics"]["NDCG@10"],
        "target_audit_rows": audit_rows,
        "ablation_rows": ablations,
        "gradient_rows": gradients,
        "auxiliary_pairwise_rows": pairwise,
        "primary_aware_assessment": assessment,
        "test_evaluation_count": 0,
        "test_used": False,
        "source_artifacts": {
            key: rel_path(Path(config["outputs"]["runs_dir"]) / f"{run['run_id']}.json")
            for key, run in runs.items()
        },
        "target_audit_source": rel_path(Path(config["outputs"]["runs_dir"]) / f"{target_audit['run_id']}.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/stage3_auxiliary_analysis/config.yaml")
    parser.add_argument("--update-results-csv", action="store_true")
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    artifacts = load_stage3_artifacts(config)
    summary = build_summary(config)
    save_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(build_markdown(summary), encoding="utf-8")
    if args.update_results_csv:
        update_results_csv(result_registry_rows(config, artifacts["runs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
