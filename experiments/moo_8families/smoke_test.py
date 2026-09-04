#!/usr/bin/env python
"""Synthetic unit smoke tests for the MOO eight-family benchmark code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.moo_8families.pareto_models.palora import PaLoRALinear  # noqa: E402
from experiments.adaptive_multitask_tim4rec.methods.pcgrad import PCGradProjector  # noqa: E402
from experiments.moo_8families.evaluation.objectives import gradient_diagnostics  # noqa: E402
from experiments.moo_8families.evaluation.pareto import validation_summary_from_records  # noqa: E402
from experiments.moo_8families.strategies.base import TASK_ORDER, preference_tensor  # noqa: E402
from experiments.moo_8families.strategies.epo import ExactParetoPreferenceSolver  # noqa: E402
from experiments.moo_8families.strategies.famo import FAMO  # noqa: E402
from experiments.moo_8families.strategies.gradhv import DominatedHypervolume  # noqa: E402
from experiments.moo_8families.strategies.pcgrad_adapter import load_historical_pcgrad  # noqa: E402
from experiments.moo_8families.strategies.preferences import ContinuousPreferenceSampler  # noqa: E402
from experiments.moo_8families.strategies.stch import SmoothTchebycheffScalarizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/moo_8families/runs/synthetic_smoke.json")
    parser.add_argument("--pcgrad-json", default="experiments/adaptive_multitask_tim4rec/runs/pcgrad_001.json")
    return parser.parse_args()


def finite_grad_check(loss: torch.Tensor, params: list[nn.Parameter]) -> dict[str, Any]:
    for param in params:
        param.grad = None
    loss.backward()
    missing = [idx for idx, param in enumerate(params) if param.grad is None]
    nonfinite = [idx for idx, param in enumerate(params) if param.grad is not None and not torch.isfinite(param.grad).all()]
    return {"missing": missing, "nonfinite": nonfinite, "ok": not missing and not nonfinite}


class TinyDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, len(TASK_ORDER), bias=False)

    def auxiliary_heads(self) -> dict[str, nn.Module]:
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x)


def tiny_diagnostic_losses(model: TinyDiagnosticModel) -> dict[str, torch.Tensor]:
    out = model(torch.eye(3))
    losses = {
        "rank": out[:, 0].pow(2).mean() + 1.0,
        "is_click_loss": (out[:, 1] - 0.2).pow(2).mean() + 1.0,
        "long_view_loss": (out[:, 2] + 0.1).pow(2).mean() + 1.0,
        "is_like_loss": (out[:, 3] - 0.4).pow(2).mean() + 1.0,
        "is_profile_enter_loss": (out[:, 4] + 0.3).pow(2).mean() + 1.0,
    }
    losses["normalized_task_vector"] = torch.stack(
        [losses["rank"], *[losses[f"{task}_loss"] for task in TASK_ORDER[1:]]]
    )
    return losses


def test_fresh_diagnostics_after_backward_regression() -> dict[str, Any]:
    model = TinyDiagnosticModel()
    stale_losses = tiny_diagnostic_losses(model)
    stale_losses["normalized_task_vector"].sum().backward()
    stale_error = None
    try:
        gradient_diagnostics(model, stale_losses, selector="all_backbone")
    except RuntimeError as exc:
        stale_error = str(exc)
    model.zero_grad(set_to_none=True)
    fresh_diag = gradient_diagnostics(model, tiny_diagnostic_losses(model), selector="all_backbone")
    return {
        "stale_graph_rejected": (
            stale_error is not None and "backward through the graph a second time" in stale_error
        ),
        "fresh_graph_after_backward_ok": bool(fresh_diag["all_finite_vectors"]),
        "fresh_gradient_norms": fresh_diag["gradient_norms"],
    }


def test_stch() -> dict[str, Any]:
    losses = torch.tensor([1.2, 0.8, 1.6, 0.4, 2.0], requires_grad=True)
    pref = [0.6, 0.1, 0.1, 0.1, 0.1]
    scalarizer = SmoothTchebycheffScalarizer(mu=1.0, preference=pref, nadir_vector=[1.0] * 5, warmup_steps=0)
    stch = scalarizer.scalarize(losses)
    linear = (preference_tensor(pref, device=losses.device, dtype=losses.dtype) * losses).sum()
    check = finite_grad_check(stch, [losses])
    reference_losses = losses.detach().clone().requires_grad_(True)
    pref_tensor = preference_tensor(pref, device=reference_losses.device, dtype=reference_losses.dtype)
    reference_terms = pref_tensor * torch.log(reference_losses)
    reference = torch.logsumexp(reference_terms / 1.0, dim=0) * len(TASK_ORDER)
    reference.backward()
    reference_grad = reference_losses.grad.detach().clone()
    stch_grad = losses.grad.detach().clone()
    return {
        "stch": float(stch.detach().item()),
        "linear": float(linear.detach().item()),
        "stch_not_linear": abs(float(stch.detach().item()) - float(linear.detach().item())) > 1e-6,
        "grad_check": check,
        "gradient_matches_logsumexp_reference": bool(torch.allclose(stch_grad, reference_grad, atol=1e-6, rtol=1e-6)),
        "max_gradient_abs_diff": float((stch_grad - reference_grad).abs().max().item()),
    }


def test_famo() -> dict[str, Any]:
    param = nn.Parameter(torch.tensor([1.0, -0.5]))
    famo = FAMO(device=param.device)
    losses = torch.stack([(param[0] - i * 0.1).pow(2) + (param[1] + i * 0.2).pow(2) + 1.0 for i in range(5)])
    z = torch.softmax(famo.w, dim=-1)
    denom = (losses - famo.min_losses).clamp_min(famo.eps)
    official_formula = (denom.log() * z / (z / denom).sum().detach().clamp_min(famo.eps)).sum()
    weighted = famo.get_weighted_loss(losses)
    check = finite_grad_check(weighted, [param])
    with torch.no_grad():
        param -= 0.01 * param.grad
    current = torch.stack([(param[0] - i * 0.1).pow(2) + (param[1] + i * 0.2).pow(2) + 1.0 for i in range(5)])
    update = famo.update(current)
    return {
        "weighted_loss": float(weighted.detach().item()),
        "matches_official_formula": bool(torch.allclose(weighted.detach(), official_formula.detach(), atol=1e-8, rtol=1e-8)),
        "grad_check": check,
        "effective_weights": update.effective_weights,
        "weights_sum": sum(update.effective_weights),
    }


def test_epo() -> dict[str, Any]:
    losses = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4])
    gradients = torch.eye(5)
    solver = ExactParetoPreferenceSolver([0.6, 0.1, 0.1, 0.1, 0.1], alpha_multiplier=5)
    alpha = solver.alpha(losses, gradients)
    simplex = [value / 5.0 for value in alpha.detach().cpu().tolist()]
    two_task_right = ExactParetoPreferenceSolver([0.5, 0.5], task_order=("rank", "is_click"), alpha_multiplier=2)
    alpha_right = two_task_right.alpha(torch.tensor([1.0, 2.0]), torch.eye(2))
    two_task_left = ExactParetoPreferenceSolver([0.5, 0.5], task_order=("rank", "is_click"), alpha_multiplier=2)
    alpha_left = two_task_left.alpha(torch.tensor([2.0, 1.0]), torch.eye(2))
    return {
        "alpha": alpha.detach().cpu().tolist(),
        "simplex_sum": sum(simplex),
        "all_non_negative": all(value >= -1e-8 for value in simplex),
        "two_task_known_behavior": {
            "losses_1_2_pref_balanced_alpha": alpha_right.detach().cpu().tolist(),
            "losses_2_1_pref_balanced_alpha": alpha_left.detach().cpu().tolist(),
            "expected_prefers_larger_loss_coordinate_after_task_multiplier": True,
            "right_case_ok": bool(alpha_right[1] > alpha_right[0]),
            "left_case_ok": bool(alpha_left[0] > alpha_left[1]),
        },
        "state": solver.state_dict(),
    }


def test_gradhv() -> dict[str, Any]:
    points = torch.tensor(
        [
            [0.9, 1.0, 1.1, 1.2, 1.3],
            [1.1, 0.9, 1.0, 1.2, 1.4],
            [1.2, 1.1, 0.8, 1.0, 1.1],
        ],
        requires_grad=True,
    )
    hv = DominatedHypervolume([1.5, 1.5, 1.5, 1.5, 1.5])
    loss = hv.loss(points)
    check = finite_grad_check(loss, [points])
    autograd_grad = points.grad.detach().clone()
    eps = 1e-4
    plus = points.detach().clone()
    minus = points.detach().clone()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    finite_diff = float((hv.loss(plus) - hv.loss(minus)).item() / (2 * eps))
    dominated = torch.tensor([[1.0, 1.0], [1.2, 1.2]], requires_grad=True)
    hv2 = DominatedHypervolume([2.0, 2.0], task_order=("rank", "is_click"))
    dominated_loss = hv2.loss(dominated)
    dominated_loss.backward()
    dominated_grad_zero = bool(dominated.grad[1].abs().max().item() < 1e-6)
    return {
        "loss": float(loss.detach().item()),
        "grad_check": check,
        "finite_difference_coordinate_0_0": finite_diff,
        "autograd_coordinate_0_0": float(autograd_grad[0, 0].item()),
        "finite_difference_close": abs(finite_diff - float(autograd_grad[0, 0].item())) < 1e-3,
        "dominated_solution_gradient_zero": dominated_grad_zero,
        "state": hv.state_dict(),
    }


def test_continuous_preference_sampler() -> dict[str, Any]:
    summaries = {}
    for name, alpha in {"phn": 0.2, "cosmos": 1.2, "palora": 1.0}.items():
        sampler = ContinuousPreferenceSampler(alpha=alpha, seed=20260828, coverage_threshold=0.01)
        samples = [sampler.sample_numpy() for _ in range(1000)]
        array = torch.tensor(samples)
        diag = sampler.diagnostics(reproduction_samples=1000)
        summary = diag["summary"]
        summaries[name] = {
            "alpha": alpha,
            "mean": summary["mean"],
            "min": summary["min"],
            "max": summary["max"],
            "max_simplex_sum_error": summary["max_simplex_sum_error"],
            "coverage_fraction": summary["coverage_fraction"],
            "coordinate_nonzero_count": summary["coordinate_nonzero_count"],
            "deterministic_reproduction_max_abs_error": summary["deterministic_reproduction_max_abs_error"],
            "all_coordinates_meaningfully_covered": all(value >= 0.05 for value in summary["coverage_fraction"]),
            "all_simplex_sums_ok": bool(torch.allclose(array.sum(dim=1), torch.ones(1000), atol=1e-6, rtol=1e-6)),
        }
    return summaries


def synthetic_validation_record(preference_id: str | None, ndcg10: float) -> dict[str, Any]:
    return {
        "preference_id": preference_id,
        "metrics": {
            "HR@5": 0.05,
            "HR@10": 0.1,
            "HR@20": 0.15,
            "HR@50": 0.3,
            "NDCG@5": 0.04,
            "NDCG@10": float(ndcg10),
            "NDCG@20": 0.07,
            "NDCG@50": 0.1,
        },
        "auxiliary_validation": {
            "is_click": {"bce_loss": 0.6},
            "long_view": {"bce_loss": 0.6},
            "is_like": {"bce_loss": 0.2},
            "is_profile_enter": {"bce_loss": 0.2},
        },
    }


def test_common_eval_reference_and_selection() -> dict[str, Any]:
    reference = [1.0, 2.0, 2.0, 2.0, 2.0]
    records = [
        synthetic_validation_record("rank_heavy", 0.05),
        synthetic_validation_record("click_heavy", 0.08),
        synthetic_validation_record("like_heavy", 0.07),
    ]
    epo_summary = validation_summary_from_records(records, method="epo", reference_point=reference)
    phn_summary = validation_summary_from_records(records, method="phn", reference_point=reference)
    gradhv_summary = validation_summary_from_records(
        [
            synthetic_validation_record(None, 0.05) | {"solution_index": 0},
            synthetic_validation_record(None, 0.08) | {"solution_index": 1},
        ],
        method="gradhv",
        reference_point=reference,
    )
    return {
        "common_reference_identical": (
            epo_summary["pareto_validation"]["reference_point"]
            == phn_summary["pareto_validation"]["reference_point"]
            == gradhv_summary["pareto_validation"]["reference_point"]
        ),
        "conditional_ranking_operating_point_is_rank_heavy": (
            phn_summary["ranking_operating_point"]["preference_id"] == "rank_heavy"
        ),
        "conditional_oracle_best_is_separate": (
            phn_summary["oracle_best_validation_point"]["preference_id"] == "click_heavy"
            and phn_summary["oracle_best_differs_from_ranking_operating_point"]
        ),
        "conditional_primary_not_oracle": not phn_summary["selection_is_validation_oracle"],
        "gradhv_selection_is_oracle": gradhv_summary["selection_is_validation_oracle"],
        "gradhv_selection_rule": gradhv_summary["ranking_operating_point_selection"],
    }


def test_palora() -> dict[str, Any]:
    base = nn.Linear(4, 3)
    layer = PaLoRALinear(base, task_count=len(TASK_ORDER), rank=1, alpha=1.0)
    with torch.no_grad():
        layer.lora_B.fill_(0.25)
    x = torch.randn(2, 4)
    layer.set_preference([0.6, 0.1, 0.1, 0.1, 0.1])
    y_rank = layer(x)
    layer.set_preference([0.2, 0.1, 0.1, 0.5, 0.1])
    y_like = layer(x)
    loss = y_like.pow(2).mean()
    check = finite_grad_check(loss, [layer.lora_A, layer.lora_B, layer.weight])
    return {
        "output_changes_with_preference": bool((y_rank - y_like).abs().max().detach().item() > 1e-8),
        "grad_check": check,
        "extra_parameters": layer.extra_parameters(),
    }


def test_phn_adapter_unit() -> dict[str, Any]:
    representation = nn.Parameter(torch.randn(3, 4))
    hypernetwork = nn.Sequential(nn.Linear(len(TASK_ORDER), 8), nn.ReLU(), nn.Linear(8, 8))

    def forward(weights: list[float]) -> torch.Tensor:
        pref = preference_tensor(weights)
        gamma, beta = hypernetwork(pref).view(2, 4)
        return representation * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

    y_rank = forward([0.6, 0.1, 0.1, 0.1, 0.1])
    y_like = forward([0.2, 0.1, 0.1, 0.5, 0.1])
    loss = y_like.pow(2).mean()
    params = [representation, *list(hypernetwork.parameters())]
    check = finite_grad_check(loss, params)
    return {
        "output_changes_with_preference": bool((y_rank - y_like).abs().max().detach().item() > 1e-8),
        "grad_check": check,
        "generated_adapter_parameters": 8,
    }


def test_cosmos_conditioning_unit() -> dict[str, Any]:
    representation = nn.Parameter(torch.randn(3, 4))
    encoder = nn.Sequential(nn.Linear(len(TASK_ORDER), 8), nn.ReLU())
    fusion = nn.Linear(12, 4)

    def forward(weights: list[float]) -> torch.Tensor:
        pref = preference_tensor(weights)
        pref_embedding = encoder(pref).expand(representation.shape[0], -1)
        return representation + 0.1 * torch.tanh(fusion(torch.cat([representation, pref_embedding], dim=-1)))

    y_rank = forward([0.6, 0.1, 0.1, 0.1, 0.1])
    y_like = forward([0.2, 0.1, 0.1, 0.5, 0.1])
    loss_vector = torch.tensor([1.0, 0.8, 1.2, 0.7, 1.4], requires_grad=True)
    pref = preference_tensor([0.2, 0.1, 0.1, 0.5, 0.1])
    cosmos_loss = (pref * loss_vector).sum() - 2.0 * torch.nn.functional.cosine_similarity(
        loss_vector.unsqueeze(0),
        pref.unsqueeze(0),
        dim=-1,
    ).mean()
    loss = y_like.pow(2).mean() + cosmos_loss
    params = [representation, loss_vector, *list(encoder.parameters()), *list(fusion.parameters())]
    check = finite_grad_check(loss, params)
    return {
        "output_changes_with_preference": bool((y_rank - y_like).abs().max().detach().item() > 1e-8),
        "grad_check": check,
        "uses_direct_conditioning_not_hypernetwork": True,
    }


def test_pcgrad_current_projection() -> dict[str, Any]:
    vectors = {
        "rank": torch.tensor([1.0, 0.0]),
        "is_click": torch.tensor([-1.0, 0.0]),
        "long_view": torch.tensor([0.5, 0.5]),
        "is_like": torch.tensor([0.0, 1.0]),
        "is_profile_enter": torch.tensor([-0.25, 0.25]),
    }
    projector = PCGradProjector(mode="ranking_anchored", seed=2026)
    projection = projector.project(vectors, TASK_ORDER)
    adjusted = projection["vectors"]
    event_count_by_target = {
        task: sum(1 for event in projection["projection_events"] if event["source"] == task)
        for task in TASK_ORDER
        if task != "rank"
    }
    rank = adjusted["rank"]
    rank_aux_dots = {
        task: float(torch.dot(adjusted[task].float(), rank.float()).item())
        for task in TASK_ORDER
        if task != "rank"
    }
    return {
        "mode": projection["mode"],
        "ranking_anchor_unchanged": bool(torch.allclose(adjusted["rank"], vectors["rank"])),
        "projection_event_count": projection["projection_event_count"],
        "projection_event_count_by_target": event_count_by_target,
        "conflicting_aux_projected": event_count_by_target["is_click"] == 1
        and event_count_by_target["is_profile_enter"] == 1,
        "nonconflicting_aux_not_projected": event_count_by_target["long_view"] == 0
        and event_count_by_target["is_like"] == 0,
        "projected_aux_nonnegative_dot_with_rank": all(value >= -1e-8 for value in rank_aux_dots.values()),
        "combined_gradient_finite": bool(torch.isfinite(projection["combined_gradient"]).all()),
        "rank_aux_dots_after": rank_aux_dots,
    }


def test_pcgrad_historical(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return load_historical_pcgrad(path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(2026)
    result = {
        "status": "completed",
        "task_order": list(TASK_ORDER),
        "tests": {
            "stch": test_stch(),
            "famo": test_famo(),
            "epo": test_epo(),
            "gradhv": test_gradhv(),
            "fresh_diagnostics_after_backward_regression": test_fresh_diagnostics_after_backward_regression(),
            "continuous_preference_sampler": test_continuous_preference_sampler(),
            "common_eval_reference_and_selection": test_common_eval_reference_and_selection(),
            "phn_adapter": test_phn_adapter_unit(),
            "cosmos_conditioning": test_cosmos_conditioning_unit(),
            "palora": test_palora(),
            "pcgrad_current_projection": test_pcgrad_current_projection(),
            "pcgrad_historical": test_pcgrad_historical(ROOT / args.pcgrad_json),
        },
        "test_policy": {
            "test_dataset_loaded": False,
            "test_dataloader_created": False,
            "test_evaluated": False,
            "test_evaluation_count": 0,
        },
    }
    failures = []
    if not result["tests"]["stch"]["stch_not_linear"]:
        failures.append("STCH collapsed to linear weighted sum")
    if not result["tests"]["stch"]["gradient_matches_logsumexp_reference"]:
        failures.append("STCH detached-max gradient differs from logsumexp reference")
    if not result["tests"]["famo"]["grad_check"]["ok"]:
        failures.append("FAMO gradient check failed")
    if not result["tests"]["famo"]["matches_official_formula"]:
        failures.append("FAMO weighted loss differs from official formula")
    if abs(result["tests"]["famo"]["weights_sum"] - 1.0) > 1e-6:
        failures.append("FAMO weights are not simplex")
    if not result["tests"]["epo"]["all_non_negative"] or abs(result["tests"]["epo"]["simplex_sum"] - 1.0) > 1e-6:
        failures.append("EPO alpha is not simplex")
    if not result["tests"]["epo"]["two_task_known_behavior"]["right_case_ok"]:
        failures.append("EPO two-task [1,2] regression failed")
    if not result["tests"]["epo"]["two_task_known_behavior"]["left_case_ok"]:
        failures.append("EPO two-task [2,1] regression failed")
    if not result["tests"]["gradhv"]["grad_check"]["ok"]:
        failures.append("GradHV gradient check failed")
    if not result["tests"]["gradhv"]["finite_difference_close"]:
        failures.append("GradHV finite-difference check failed")
    if not result["tests"]["gradhv"]["dominated_solution_gradient_zero"]:
        failures.append("GradHV dominated solution handling failed")
    if not result["tests"]["fresh_diagnostics_after_backward_regression"]["stale_graph_rejected"]:
        failures.append("Diagnostic stale-graph regression did not reproduce the freed-graph failure")
    if not result["tests"]["fresh_diagnostics_after_backward_regression"]["fresh_graph_after_backward_ok"]:
        failures.append("Fresh diagnostic graph after backward is not finite")
    for name, summary in result["tests"]["continuous_preference_sampler"].items():
        if not summary["all_coordinates_meaningfully_covered"]:
            failures.append(f"{name} Dirichlet sampler does not cover all coordinates")
        if not summary["all_simplex_sums_ok"]:
            failures.append(f"{name} Dirichlet sampler simplex sums failed")
        if summary["deterministic_reproduction_max_abs_error"] != 0.0:
            failures.append(f"{name} Dirichlet sampler is not deterministic")
    selection = result["tests"]["common_eval_reference_and_selection"]
    if not selection["common_reference_identical"]:
        failures.append("Common evaluation HV reference differs across method summaries")
    if not selection["conditional_ranking_operating_point_is_rank_heavy"]:
        failures.append("Conditional ranking operating point is not rank_heavy")
    if not selection["conditional_oracle_best_is_separate"]:
        failures.append("Conditional oracle best is not kept separately")
    if not selection["conditional_primary_not_oracle"]:
        failures.append("Conditional ranking operating point is marked as validation oracle")
    if not selection["gradhv_selection_is_oracle"]:
        failures.append("GradHV preference-free selection is not marked as validation oracle")
    if not result["tests"]["phn_adapter"]["output_changes_with_preference"]:
        failures.append("PHN adapter output does not change with preference")
    if not result["tests"]["phn_adapter"]["grad_check"]["ok"]:
        failures.append("PHN adapter gradient check failed")
    if not result["tests"]["cosmos_conditioning"]["output_changes_with_preference"]:
        failures.append("COSMOS conditioning output does not change with preference")
    if not result["tests"]["cosmos_conditioning"]["grad_check"]["ok"]:
        failures.append("COSMOS conditioning gradient check failed")
    if not result["tests"]["palora"]["output_changes_with_preference"]:
        failures.append("PaLoRA output does not change with preference")
    if not result["tests"]["palora"]["grad_check"]["ok"]:
        failures.append("PaLoRA gradient check failed")
    current_pcgrad = result["tests"]["pcgrad_current_projection"]
    if not current_pcgrad["ranking_anchor_unchanged"]:
        failures.append("PCGrad current projection changed the ranking anchor")
    if not current_pcgrad["conflicting_aux_projected"]:
        failures.append("PCGrad current projection did not project conflicting auxiliary gradients")
    if not current_pcgrad["nonconflicting_aux_not_projected"]:
        failures.append("PCGrad current projection changed non-conflicting auxiliary gradients")
    if not current_pcgrad["projected_aux_nonnegative_dot_with_rank"]:
        failures.append("PCGrad current projection left a negative rank-auxiliary dot product")
    if not current_pcgrad["combined_gradient_finite"]:
        failures.append("PCGrad current combined gradient is not finite")
    result["failures"] = failures
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "failures": failures}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
