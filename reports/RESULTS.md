# Project Results

## Published Comparison

The compact literature table for the TiM4Rec Table 3 KuaiRand benchmark is in [PAPER_RESULTS.md](PAPER_RESULTS.md).

## Our Canonical TEST Reproductions

| Run | Model | Variant | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Test evals |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_002 | Random | full_ranking_history | 0.0013 | 0.0030 | 0.0076 | 0.0006 | 0.0010 | 0.0019 | 1 |
| mostpop_002 | MostPopular | full_ranking_history | 0.0295 | 0.0601 | 0.1030 | 0.0167 | 0.0243 | 0.0327 | 1 |
| ltr_xgb_002 | XGBoost LambdaMART | baseline_full_ranking | 0.0314 | 0.0557 | 0.0999 | 0.0150 | 0.0209 | 0.0297 | 1 |
| ltr_xgb_optuna_001 | XGBoost LambdaMART | tuned_optuna | 0.0333 | 0.0574 | 0.1044 | 0.0177 | 0.0237 | 0.0330 | 1 |
| ssd4rec_001 | SSD4Rec | reproduction | 0.1032 | 0.1683 | 0.3014 | 0.0576 | 0.0739 | 0.1002 | 1 |
| tim4rec_001 | TiM4Rec | reproduction | 0.1053 | 0.1696 | 0.3031 | 0.0598 | 0.0759 | 0.1022 | 1 |
| multitask_tim4rec_001 | MultitaskTiM4Rec | fixed_loss | 0.1041 | 0.1663 | 0.3025 | 0.0581 | 0.0738 | 0.1006 | 1 |
| multitask_tim4rec_tuned_001 | MultitaskTiM4Rec | tuned_fixed_weights | 0.1071 | 0.1746 | 0.3138 | 0.0598 | 0.0767 | 0.1042 | 1 |

## Tuned Multitask Result

`multitask_tim4rec_tuned_001` is the current best committed TEST result from our model family: NDCG@10 `0.0598`, HR@20 `0.1746`, NDCG@50 `0.1042`. The validation search row `multitask_optuna_search_001` used no TEST evaluation and reached validation NDCG@10 `0.0599`.

## Current 8-Family Validation Screening

These rows are validation-only. They are not TEST results.

| Run | Family | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Actual epochs | Test evals |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epo_convergence_001 | EPO | 0.1078 | 0.1767 | 0.3171 | 0.0584 | 0.0756 | 0.1033 | 15 | 30 | 0 |
| gradhv_convergence_001 | HV-Gradient / GradHV-style | 0.0874 | 0.1382 | 0.2440 | 0.0486 | 0.0613 | 0.0820 | 50 | 65 | 0 |
| cosmos_convergence_001 | COSMOS-style | 0.0810 | 0.1257 | 0.2252 | 0.0453 | 0.0565 | 0.0761 | 25 | 40 | 0 |
| pcgrad_convergence_001 | PCGrad | 0.0790 | 0.1259 | 0.2253 | 0.0444 | 0.0562 | 0.0757 | 25 | 40 | 0 |
| stch_convergence_001 | STCH | 0.0749 | 0.1163 | 0.2082 | 0.0424 | 0.0528 | 0.0709 | 80 | 95 | 0 |
| phn_convergence_001 | PHN-adapter | 0.0746 | 0.1155 | 0.2027 | 0.0423 | 0.0526 | 0.0698 | 60 | 75 | 0 |
| palora_convergence_001 | PaLoRA | 0.0750 | 0.1159 | 0.2080 | 0.0422 | 0.0525 | 0.0706 | 35 | 50 | 0 |
| famo_convergence_001 | FAMO | 0.0719 | 0.1102 | 0.1935 | 0.0412 | 0.0508 | 0.0672 | 15 | 30 | 0 |

Historical `pcgrad_001` is kept only as a validation reference for the current PCGrad implementation. The current MOO PCGrad row uses a different runner/objective geometry, so it is not treated as the same operating point.

## Budgeted Top-4 MOO Tuning Snapshot

The top-4 tuning stage is recorded as a time/compute-budgeted validation-only snapshot, not as an equal-trial-count benchmark. Internal tuning `Stage A` corresponds to scientific Stage 2 in [MOO_EXPERIMENT_HISTORY.md](MOO_EXPERIMENT_HISTORY.md). Persistent Optuna storage and raw artifacts stay outside Git; compact results are in [../experiments/moo_8families/runs/moo_stage_history_summary.json](../experiments/moo_8families/runs/moo_stage_history_summary.json). TEST evaluations for this tuning stage: `0`.

| Method | Planned complete | Actual complete | Failed | Stale | Best trial | Best epoch | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EPO | 10 | 5 | 1 | 1 | 0 | 20 | 0.1080 | 0.1778 | 0.3198 | 0.0588 | 0.0763 | 0.1043 | 36-hour walltime exhausted |
| GradHV | 12 | 12 | 0 | 0 | 1 | 90 | 0.0877 | 0.1370 | 0.2460 | 0.0488 | 0.0612 | 0.0827 | completed budget |
| COSMOS | 12 | 9 | 1 | 0 | 0 | 40 | 0.0819 | 0.1274 | 0.2289 | 0.0455 | 0.0569 | 0.0769 | preference guard stopped |
| PCGrad | 12 | 12 | 0 | 0 | 9 | 75 | 0.0828 | 0.1298 | 0.2317 | 0.0464 | 0.0581 | 0.0783 | completed budget |

## Stage 3 Auxiliary-Task Analysis

The validation-only auxiliary-task and gradient-interaction audit is in [STAGE3_AUXILIARY_ANALYSIS.md](STAGE3_AUXILIARY_ANALYSIS.md). In the single-seed ablation, primary-only reached NDCG@10 `0.0586`; the best single auxiliary was `is_click` at `0.0593` (`+0.0007`). TEST evaluations for Stage 3: `0`.
