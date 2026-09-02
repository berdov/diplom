# MOO Experiment History

KuaiRand Protocol B. Stage 1 and Stage 2 are validation-only; TEST was not used for method selection, hyperparameter tuning, or configuration selection.

Internal `Stage A` in the tuning scripts corresponds to scientific `Stage 2` here: the first fixed compute/time budget for top-4 hyperparameter tuning.

## Experimental Stages

| Stage | Scientific role | Compact source |
| --- | --- | --- |
| Stage 0 - Multitask Control | Tuned/fixed-weight `MultitaskTiM4Rec` control for context. | `multitask_tim4rec_tuned_001` in [RESULTS.md](RESULTS.md) |
| Stage 1 - 8-Family MOO Screening | Broad screening of eight MOO families before tuning. | `experiments/moo_8families/runs/*_convergence_001.json` |
| Stage 2 - Top-4 Hyperparameter Tuning | EPO, GradHV, COSMOS and PCGrad under a limited compute/time budget. | cHARISMa Optuna DB/logs and [moo_stage_history_summary.json](../experiments/moo_8families/runs/moo_stage_history_summary.json) |

## Stage 1 - 8-Family MOO Screening

Validation-only full-ranking screening. These are not TEST results.

| Method | Family/category | Run | Status | Best epoch | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Test evals |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STCH | loss_balancing | `stch_convergence_001` | completed | 80 | .0749 | .1163 | .2082 | .0424 | .0528 | .0709 | 0 |
| FAMO | gradient_weighting | `famo_convergence_001` | completed | 15 | .0719 | .1102 | .1935 | .0412 | .0508 | .0672 | 0 |
| PCGrad | gradient_manipulation | `pcgrad_convergence_001` | completed | 25 | .0790 | .1259 | .2253 | .0444 | .0562 | .0757 | 0 |
| EPO | finite_preference_set | `epo_convergence_001` | completed | 15 | .1078 | .1767 | .3171 | .0584 | .0756 | .1033 | 0 |
| GradHV-style | finite_no_preference_set | `gradhv_convergence_001` | completed | 50 | .0874 | .1382 | .2440 | .0486 | .0613 | .0820 | 0 |
| PHN-adapter | infinite_hypernetwork | `phn_convergence_001` | completed | 60 | .0746 | .1155 | .2027 | .0423 | .0526 | .0698 | 0 |
| COSMOS-style | infinite_preference_conditioned | `cosmos_convergence_001` | completed | 25 | .0810 | .1257 | .2252 | .0453 | .0565 | .0761 | 0 |
| PaLoRA | infinite_model_combination | `palora_convergence_001` | completed | 35 | .0750 | .1159 | .2080 | .0422 | .0525 | .0706 | 0 |

## Stage 2 - Top-4 Hyperparameter Tuning

Stage 2 is an exploratory top-4 hyperparameter tuning snapshot under the available compute/time budget. It is considered complete as a budgeted experiment, not as an equal-trial-count benchmark. The factual result for each method is the best COMPLETE trial obtained before the allocated job stopped.

| Method | Planned complete trials | Actual complete trials | Failed trials | Stale trials | Best trial | Best epoch | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Stop reason | Slurm status | Study/run identifier |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| EPO | 10 | 5 | 1 (`0001`) | 1 (`0006`) | 0 | 20 | .1080 | .1778 | .3198 | .0588 | .0763 | .1043 | 36-hour Slurm walltime exhausted; stale partial trial is not counted. | `4295022 TIMEOUT` | `epo_tuning_001` |
| GradHV | 12 | 12 | 0 | 0 | 1 | 90 | .0877 | .1370 | .2460 | .0488 | .0612 | .0827 | Planned internal Stage A budget completed. | `4295023 COMPLETED 0:0` | `gradhv_tuning_001` |
| COSMOS | 12 | 9 | 1 (`0009`) | 0 | 0 | 40 | .0819 | .1274 | .2289 | .0455 | .0569 | .0769 | Scientific `preference_sensitivity` guard failed; collapsed trial is not accepted as a valid result. | `4295024 FAILED 1:0` | `cosmos_tuning_001` |
| PCGrad | 12 | 12 | 0 | 0 | 9 | 75 | .0828 | .1298 | .2317 | .0464 | .0581 | .0783 | Planned internal Stage A budget completed. | `4295025 COMPLETED 0:0` | `pcgrad_tuning_001` |

## Screening To Tuning Comparison

| Method | Screening NDCG@10 | Tuned NDCG@10 | Absolute delta | Relative delta |
| --- | ---: | ---: | ---: | ---: |
| EPO | .0584 | .0588 | +.0004 | +0.68% |
| GradHV | .0486 | .0488 | +.0002 | +0.41% |
| COSMOS | .0453 | .0455 | +.0002 | +0.44% |
| PCGrad | .0444 | .0464 | +.0020 | +4.50% |

## Stage 2 Interpretation

EPO has the best observed NDCG@10 among the tuned MOO methods, but it exhausted the 36-hour Slurm walltime before the originally planned 10 COMPLETE trials. Its stale partial trial `0006` reached validation NDCG@10 `.0585` at epoch 25, but it is not counted as COMPLETE and is not used as best.

GradHV completed the full planned internal Stage A budget. PCGrad also completed the full budget and showed the largest relative improvement over its screening configuration.

COSMOS stopped because trial `0009` failed the scientific `preference_sensitivity` guard: preference conditioning collapsed, so the failed configuration is excluded from valid best-trial selection.

This snapshot supports statements about the best observed validation-only configurations under the available budget. It does not prove absolute superiority of one MOO algorithm over another, because the methods had different completed-trial counts and different stopping modes. It also does not support TEST, Stage B, multiseed, or novel-method claims.

## Test Hygiene

Stage 1 committed run artifacts and 41 Stage 2 tuning `result.json`/`result.partial.json` artifacts were checked. All had `test_evaluation_count = 0`.
