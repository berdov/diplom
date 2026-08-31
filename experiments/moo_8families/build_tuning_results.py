#!/usr/bin/env python
"""Build compact reports from completed MOO tuning study summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPACES = ROOT / "configs" / "moo_tuning_spaces.yaml"
REPORTS_DIR = ROOT / "reports"
BEST_CONFIG_DIR = ROOT / "configs" / "best_tuned"
METHOD_ORDER = ("epo", "gradhv", "cosmos", "pcgrad")
CONTROL_NDCG10 = 0.0589


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default=str(DEFAULT_SPACES))
    parser.add_argument("--summary-root", default=None)
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--best-config-dir", default=str(BEST_CONFIG_DIR))
    parser.add_argument("--write-best-configs", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def state_count(summary: Mapping[str, Any] | None, state: str) -> int:
    if not summary:
        return 0
    return int((summary.get("state_counts") or {}).get(state, 0))


def best_trial(summary: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not summary:
        return None
    best = summary.get("best_trial")
    return best if isinstance(best, Mapping) else None


def method_status(summary: Mapping[str, Any] | None, target: int) -> str:
    if not summary:
        return "MISSING_SUMMARY"
    complete = state_count(summary, "COMPLETE")
    if complete >= target:
        return "COMPLETE"
    if complete > 0:
        return "PARTIAL"
    return "NO_COMPLETED_TRIALS"


def key_changed_params(params: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    changed = []
    for key, value in params.items():
        if key not in baseline:
            changed.append(f"{key}={fmt(value, 6)}")
            continue
        base = float(baseline[key])
        current = float(value)
        if abs(current - base) > max(abs(base) * 1e-6, 1e-12):
            changed.append(f"{key}={fmt(current, 6)}")
    return "; ".join(changed) if changed else "baseline"


def main_rows(spaces: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    rows = []
    for method in METHOD_ORDER:
        method_spec = spaces["methods"][method]
        summary = summaries[method]
        best = best_trial(summary)
        params = best.get("params", {}) if best else {}
        baseline = {
            name: spaces["common_parameters"].get(name, method_spec.get("method_parameters", {}).get(name, {})).get("current")
            for name in method_spec["parameters"]
        }
        current = float(method_spec["current_ndcg10"])
        tuned = None if best is None else float(best["NDCG@10"])
        abs_gain = None if tuned is None else tuned - current
        rel_gain = None if tuned is None else abs_gain / current
        rows.append(
            {
                "Method": method,
                "Study": method_spec["study_name"],
                "Status": method_status(summary, int(method_spec["target_complete_trials"])),
                "Target complete": int(method_spec["target_complete_trials"]),
                "Complete": state_count(summary, "COMPLETE"),
                "Failed": state_count(summary, "FAIL"),
                "Pruned": state_count(summary, "PRUNED"),
                "Running": state_count(summary, "RUNNING"),
                "Current NDCG@10": current,
                "Best tuned NDCG@10": tuned,
                "Absolute gain": abs_gain,
                "Relative gain": rel_gain,
                "Best trial": None if best is None else best.get("trial"),
                "Best epoch": None if best is None else best.get("best_epoch"),
                "HR@10": None if best is None else best.get("HR@10"),
                "HV": None if best is None else best.get("HV"),
                "Non-dominated": None if best is None else best.get("non_dominated"),
                "Spread": None if best is None else best.get("spread"),
                "Runtime": None if best is None else best.get("runtime_sec"),
                "vs control NDCG@10": None if tuned is None else tuned - CONTROL_NDCG10,
                "Key changed params": "" if best is None else key_changed_params(params, baseline),
                "Test evaluation count": None if summary is None else summary.get("test_evaluation_count", 0),
                "Summary JSON": "" if summary is None else summary.get("summary_path", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[Mapping[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(fmt(value) if isinstance(value, float) else str(value or ""))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def top_trial_rows(summary: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not summary:
        return []
    rows = []
    for trial in (summary.get("top_trials") or [])[:5]:
        rows.append(
            {
                "trial": trial.get("trial"),
                "NDCG@10": trial.get("NDCG@10"),
                "HR@10": trial.get("HR@10"),
                "best epoch": trial.get("best_epoch"),
                "params": json.dumps(trial.get("params", {}), ensure_ascii=False, sort_keys=True),
                "HV": trial.get("HV"),
                "spread": trial.get("spread"),
                "aux BCE summary": json.dumps(trial.get("aux_bce", {}), ensure_ascii=False, sort_keys=True),
                "runtime": trial.get("runtime_sec"),
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]], summaries: Mapping[str, Mapping[str, Any] | None]) -> None:
    lines = [
        "# Controlled MOO tuning results",
        "",
        "Этот файл строится из compact Optuna summaries, а не из TEST. Если summaries отсутствуют или неполные, таблица фиксирует фактический статус вместо подстановки результатов.",
        "",
        f"Validation control: `multitask_tim4rec_tuned_001`, NDCG@10 ~= `{CONTROL_NDCG10}`.",
        "",
        "## Main comparison",
        "",
    ]
    lines += md_table(
        rows,
        [
            "Method",
            "Status",
            "Complete",
            "Current NDCG@10",
            "Best tuned NDCG@10",
            "Absolute gain",
            "HR@10",
            "HV",
            "Non-dominated",
            "Spread",
            "Best trial",
            "Best epoch",
            "Key changed params",
        ],
    )
    lines += ["", "## Top 5 trials per method", ""]
    for method in METHOD_ORDER:
        lines += [f"### {method}", ""]
        top_rows = top_trial_rows(summaries[method])
        if top_rows:
            lines += md_table(
                top_rows,
                ["trial", "NDCG@10", "HR@10", "best epoch", "params", "HV", "spread", "aux BCE summary", "runtime"],
            )
        else:
            lines.append("Нет completed trials в summary.")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_configs(spaces: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any] | None], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in METHOD_ORDER:
        summary = summaries[method]
        best = best_trial(summary)
        if not best:
            continue
        payload = {
            "method": method,
            "study_group": spaces["study_group"],
            "study": spaces["methods"][method]["study_name"],
            "trial": int(best["trial"]),
            "validation_score": {"NDCG@10": float(best["NDCG@10"]), "HR@10": float(best["HR@10"])},
            "dataset_fingerprint": spaces["protocol"]["identity_hash"],
            "git_commit": (best.get("git") or summary.get("git") or {}).get("commit"),
            "seed": spaces["training_protocol"]["tuning_seed"],
            "params": dict(best.get("params", {})),
            "test_evaluation_count": int(best.get("test_evaluation_count", 0)),
            "source_summary": summary.get("summary_path"),
        }
        (out_dir / f"{method}_best.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    spaces = load_yaml(Path(args.spaces))
    summary_root = project_path(args.summary_root or spaces["storage"]["summary_root"])
    summaries: dict[str, Mapping[str, Any] | None] = {}
    for method in METHOD_ORDER:
        method_spec = spaces["methods"][method]
        path = summary_root / f"{method_spec['study_name']}_summary.json"
        summary = load_json(path) if path.exists() else None
        if summary is not None:
            summary["summary_path"] = str(path)
        summaries[method] = summary

    rows = main_rows(spaces, summaries)
    reports_dir = project_path(args.reports_dir)
    write_csv(reports_dir / "tuning_results.csv", rows)
    write_markdown(reports_dir / "TUNING_RESULTS.md", rows, summaries)
    if args.write_best_configs:
        write_best_configs(spaces, summaries, project_path(args.best_config_dir))
    print(json.dumps({"rows": len(rows), "reports_dir": str(reports_dir)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
