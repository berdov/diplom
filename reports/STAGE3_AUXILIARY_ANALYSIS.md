# Stage 3 Auxiliary-Task Analysis

KuaiRand Protocol B, validation-only. TEST was not used.

## Run Scope

Stage 3 audits the available behavior labels and tests marginal auxiliary-task effects for the established `MultitaskTiM4Rec` setup. The primary objective remains next-item ranking; auxiliary metrics are reported only as diagnostics and are not used to choose the recommender.

Artifacts were produced from infrastructure commit `2177bc4e0d082ad6ccb8532f04b9e015fa80e9a0`. The validation-only RecBole dataset uses `benchmark_filename = ["train", "valid"]`; every artifact records `test_evaluation_count = 0`.

## Target Audit

| Signal | Field | Type | Train observations | Train positives / rate | Validation rate | Missing rate | Current | Ablated |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| click | `is_click` | binary | 1086518 | 502248 / .4623 | .4826 | .0000 | yes | yes |
| long_view | `long_view` | binary | 1086518 | 364383 / .3354 | .3574 | .0000 | yes | yes |
| like | `is_like` | binary | 1086518 | 20232 / .0186 | .0196 | .0000 | yes | yes |
| profile_enter | `is_profile_enter` | binary | 1086518 | 27730 / .0255 | .0237 | .0000 | yes | yes |
| follow | `is_follow` | binary | 1086518 | 1046 / .0010 | .0012 | .0000 | no | no |
| comment | `is_comment` | binary | 1086518 | 2818 / .0026 | .0021 | .0000 | no | no |
| forward | `is_forward` | binary | 1086518 | 1065 / .0010 | .0010 | .0000 | no | no |
| hate | `is_hate` | binary | 1086518 | 458 / .0004 | .0005 | .0000 | no | no |
| play_time | `play_time_ms` | continuous | 1086518 | n/a | n/a | .0000 | no | no |
| play_ratio | `play_ratio` | continuous | 1086518 | n/a | n/a | .0209 | no | no |

Continuous train distributions:

| Signal | Mean | Median | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| play_time | 23109.0289 | 4938.0000 | 69410.3000 | 106473.0000 | 211966.8300 | 964774.0000 |
| play_ratio | .3696 | .1021 | 1.0377 | 1.1561 | 2.0957 | 58.0329 |

## Construction And Eligibility

| Signal | Construction in repository | Interaction-time availability and leakage decision |
| --- | --- | --- |
| `is_click` | Raw post-exposure field retained by `experiments/multitask_tim4rec/audit_targets.py`; exported to validation-only RecBole files by `experiments/multitask_tim4rec_optuna/prepare_validation_only.py`. | Same-row target label only; `MultitaskTiM4Rec.input_fields_used` excludes it, so it is not an input feature. No future target leakage found for Stage 3 use. |
| `long_view` | Raw/derived post-exposure watch label retained by `experiments/multitask_tim4rec/audit_targets.py`; exported as a float target in validation-only RecBole files. | Same-row target label only; not an input feature. It is strongly tied to watch time, so it is eligible as an auxiliary target but not independent evidence of ranking utility. |
| `is_like` | Raw explicit positive action retained by `experiments/multitask_tim4rec/audit_targets.py`; modeled by the existing `like_head` in `experiments/multitask_tim4rec/model.py`. | Same-row target label only; not an input feature. Severe class imbalance is present but no future target leakage was found. |
| `is_profile_enter` | Raw profile-entry action retained by `experiments/multitask_tim4rec/audit_targets.py`; modeled by the existing `profile_enter_head`. | Same-row target label only; not an input feature. Class imbalance is present; Stage 3 keeps it because the head already exists. |
| `is_follow` | Raw explicit action present in Protocol B multitask audit data. | Eligible as a label in principle, but omitted from first ablation because prevalence is only .0010 and adding it requires a model-scope change. |
| `is_comment` | Raw explicit action present in Protocol B multitask audit data. | Eligible as a label in principle, but omitted from first ablation because prevalence is only .0026 and adding it requires a model-scope change. |
| `is_forward` | Raw explicit action present in Protocol B multitask audit data. | Eligible as a label in principle, but omitted from first ablation because prevalence is only .0010 and adding it requires a model-scope change. |
| `is_hate` | Raw negative feedback present in Protocol B multitask audit data. | Not mixed with positive engagement labels; omitted because it is extremely rare and semantically negative, so it needs a separate objective design. |
| `play_time_ms` | Raw watch-time field retained in Protocol B multitask audit data. | Post-exposure continuous signal; omitted because fair use requires a regression, survival, clipping, or transformation design outside this first single-auxiliary pass. |
| `play_ratio` | Derived in audit code as `play_time_ms / duration_ms` when `duration_ms > 0`. | Post-exposure continuous signal with missing values when duration is non-positive; omitted for the same objective-design reason as watch time. |

The first ablation pass therefore uses exactly the four existing heads: `is_click`, `long_view`, `is_like`, and `is_profile_enter`.

## Relationship And Redundancy Analysis

Binary associations use train-split co-occurrence, conditional probabilities, Jaccard overlap, and phi coefficient. These statistics describe label relationships only; target-target association is not evidence that an auxiliary target improves next-item ranking.

| Pair | Phi | Jaccard | P(right=1 \| left=1) | P(left=1 \| right=1) | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `is_click` / `long_view` | .7601 | .7202 | .7224 | .9958 | Strongly related; long_view is almost nested inside click/valid-play, but not identical. |
| `is_click` / `is_like` | .1078 | .0341 | .0343 | .8524 | Likes are rare and usually occur with click, but click rarely implies like. |
| `is_click` / `is_profile_enter` | .1569 | .0520 | .0522 | .9455 | Profile entry is rare and usually occurs with click. |
| `long_view` / `is_like` | .0997 | .0369 | .0376 | .6770 | Likes often involve long views, but most long views are not likes. |
| `long_view` / `is_profile_enter` | .1473 | .0572 | .0582 | .7649 | Profile entry overlaps with long_view but remains a distinct behavior. |
| `is_like` / `is_profile_enter` | .0597 | .0412 | .0939 | .0685 | Explicit positive actions are only weakly overlapping. |

Continuous/watch-time relationships are descriptive. `play_time_ms` and `play_ratio` are strongly associated with `long_view` and `is_click`, which is expected because `long_view` is a watch-time-derived behavior. For example, Pearson(`play_ratio`, `long_view`) is .6480 and Pearson(`play_ratio`, `is_click`) is .5544 on non-missing train rows.

## Single-Auxiliary Ablations

| Run | Auxiliary | Best epoch | Actual epochs | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Delta NDCG@10 | Test evals |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stage3_primary_only_001` | `none` | 11 | 16 | .1080 | .1755 | .3124 | .0586 | .0755 | .1025 | .0000 | 0 |
| `stage3_aux_click_001` | `is_click` | 17 | 22 | .1086 | .1779 | .3183 | .0593 | .0767 | .1043 | +.0007 | 0 |
| `stage3_aux_long_view_001` | `long_view` | 9 | 14 | .1083 | .1736 | .3117 | .0586 | .0750 | .1022 | +.0000 | 0 |
| `stage3_aux_like_001` | `is_like` | 11 | 16 | .1080 | .1777 | .3150 | .0587 | .0762 | .1033 | +.0001 | 0 |
| `stage3_aux_profile_enter_001` | `is_profile_enter` | 11 | 16 | .1088 | .1737 | .3141 | .0590 | .0752 | .1030 | +.0004 | 0 |

The all-current diagnostic control with all four current auxiliary heads reached validation NDCG@10 `.0597` at epoch 17. It is used here for the auxiliary-auxiliary gradient matrix, not as a new TEST result.

## Gradient Diagnostics

Gradients were measured on the same shared TiM4Rec backbone parameter set, excluding task-specific heads. The runner restores RNG state before the training step after diagnostics, so the diagnostic pass does not intentionally change optimization behavior.

| Auxiliary | Batches | Median norm ratio | Mean cosine | Median cosine | Q25 cosine | Q75 cosine | Conflict fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `is_click` | 5 | .1413 | .0452 | .0428 | .0275 | .0565 | .2000 |
| `long_view` | 3 | .1564 | .0223 | .0219 | .0143 | .0301 | .0000 |
| `is_like` | 4 | .4343 | .0265 | .0308 | -.0075 | .0649 | .5000 |
| `is_profile_enter` | 4 | .3254 | -.0082 | .0007 | -.0125 | .0049 | .2500 |

## Combined Auxiliary Interpretation

| Auxiliary | Train prevalence | Delta NDCG@10 | Mean cosine with primary | Conflict rate | Median norm ratio | Auxiliary validation metric | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `is_click` | .4623 | +.0007 | .0452 | .2000 | .1413 | BCE .6382, accuracy .6376 | Best single auxiliary in this pass: small positive ranking delta with mildly aligned gradients. |
| `long_view` | .3354 | +.0000 | .0223 | .0000 | .1564 | BCE .6070, accuracy .6679 | No primary conflict, but near-zero ranking gain; consistent with redundancy with click/watch-time behavior. |
| `is_like` | .0186 | +.0001 | .0265 | .5000 | .4343 | BCE .1545, accuracy .9708 | Rare label with larger relative gradients and frequent conflict; ranking gain is tiny and should be treated as unstable. |
| `is_profile_enter` | .0255 | +.0004 | -.0082 | .2500 | .3254 | BCE .1814, accuracy .9760 | Small ranking gain despite near-zero/negative mean cosine; positive predictions are very sparse, so evidence is mixed. |

No current auxiliary target is actively harmful by validation NDCG@10 in this single-seed pass. Secondary ranking metrics are mixed for `long_view` and `is_profile_enter`, so the result should not be read as broad dominance.

## Auxiliary-Auxiliary Gradient Matrix

The matrix below is from `stage3_all_current_aux_diagnostic_001`.

| Left | Right | Batches | Mean cosine | Median cosine | Conflict fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| `is_click` | `is_like` | 5 | .0795 | .0723 | .2000 |
| `is_click` | `is_profile_enter` | 5 | .0139 | .0212 | .4000 |
| `is_click` | `long_view` | 5 | .5976 | .7392 | .2000 |
| `is_like` | `is_profile_enter` | 5 | .0270 | .0517 | .2000 |
| `long_view` | `is_like` | 5 | .0288 | -.0058 | .6000 |
| `long_view` | `is_profile_enter` | 5 | -.0074 | -.0006 | .6000 |

The strongest auxiliary-auxiliary alignment is `is_click` with `long_view`; this matches the data-level redundancy. `long_view` conflicts more often with the rare explicit-action heads, so a symmetric multi-objective method can spend capacity resolving auxiliary-auxiliary disagreement that does not clearly improve primary ranking.

## Relation To Stage 1/2 MOO Results

Stage 3 is consistent with, but does not causally prove, the following interpretation of the Stage 1/2 observations:

- EPO performed strongly because a finite preference search can include operating points that keep the primary ranking objective protected while still using useful auxiliary signal. Its Stage 2 NDCG@10 `.0588` is close to the Stage 3 primary-only `.0586`, but below the all-current diagnostic `.0597`.
- PCGrad improved after tuning, which is consistent with the observed mild-to-moderate gradient conflicts. The Stage 3 conflicts are not catastrophic for every auxiliary, so projection alone is not automatically enough; weights and operating point still matter.
- GradHV and COSMOS remained weaker in Stage 1/2. The Stage 3 evidence is consistent with the idea that rare heads and auxiliary-auxiliary conflicts can distract methods that treat objectives too symmetrically, but this is not a proof of mechanism.
- COSMOS preference-collapse failures from Stage 2 remain compatible with this picture: conditioning on preferences is only useful if the model actually produces distinct primary-relevant trade-offs.

## Primary-Aware Hypothesis

Hypothesis: for this recommendation setting, the next-item ranking objective is primary and behavior tasks are auxiliary, so a future method should protect the primary gradient and exploit auxiliary gradients only when they do not damage the primary objective.

Evidence for the hypothesis:

- The best single auxiliary (`is_click`) has the largest positive NDCG@10 delta and mildly positive primary cosine.
- `is_like` and `is_profile_enter` have larger relative gradient norms and more conflict, while their ranking gains are small.
- The all-current diagnostic improves NDCG@10, so auxiliary information is useful, but the single-auxiliary pattern suggests the usefulness is uneven.

Evidence against or limitations:

- No current auxiliary target has negative NDCG@10 delta in this single-seed Stage 3 pass.
- The correlation between median primary cosine and NDCG@10 delta is only `.2626`, so gradient alignment alone does not explain ranking impact.
- The sample of diagnostic batches is compact by design and should not be treated as a full training-trajectory proof.

Exact failure mode a primary-aware method would target: large or conflicting auxiliary updates from sparse behavior heads changing shared ranking representations when they do not improve validation NDCG@10.

Existing methods partially cover the idea. PCGrad can remove conflicting components; EPO searches Pareto trade-offs; GradHV uses a hypervolume objective; COSMOS and PHN condition on preferences. A diploma contribution would need to be more specific than "use gradients": it would need a reproducible primary-priority rule, a clear distinction from PCGrad/EPO scalarization, validation-only selection protocol, and later locked TEST evaluation after method selection.

## Additional Target Inclusion

`is_follow`, `is_comment`, `is_forward`, and `is_hate` are available but too sparse for this first single-head pass without additional objective design. `is_hate` should remain a separate negative-feedback target, not a positive engagement label. `play_time_ms` and `play_ratio` are promising consumption signals, but using them fairly requires a continuous-objective design such as `log1p(play_time_ms)` or clipped `play_ratio`; that is a separate experiment.

Recommended next scientific experiment: validation-only primary-aware gradient gating on the existing four heads, compared first against `primary_only`, `is_click`, and the all-current fixed-weight diagnostic under the same seed and budget. Multiseed and TEST should still wait until the method and selection protocol are fixed.

## Slurm Provenance

| Role | Job ID | Partition | Status | Notes |
| --- | ---: | --- | --- | --- |
| Target audit | 4299312 | `cpu-e-quick` | COMPLETED | CPU-only audit, `TEST=0`. |
| Sanity run | 4299311 | `test` | COMPLETED | One epoch, two train batches, finite diagnostics, `TEST=0`. |
| Primary-only ablation | 4299376 | `gpu-ef-quick` | COMPLETED | Produced `stage3_primary_only_001.json`. |
| Click ablation | 4299449 | `test` | TIMEOUT | Final JSON was written as `COMPLETE`; partial file removed; `TEST=0`. |
| Long-view ablation | 4299450 | `test` | COMPLETED | Produced `stage3_aux_long_view_001.json`. |
| Like ablation | 4299451 | `test` | COMPLETED | Produced `stage3_aux_like_001.json`. |
| Profile-enter ablation | 4299452 | `test` | COMPLETED | Produced `stage3_aux_profile_enter_001.json`. |
| All-current diagnostic | 4299453 | `test` | COMPLETED | Produced auxiliary-auxiliary gradient matrix. |

## Test Hygiene

- `test_evaluation_count = 0` for every Stage 3 artifact.
- Stage 3 artifacts report `test_dataset_loaded = false` and `test_metrics_present = false`.
- No TEST metrics are used for target selection, ablation interpretation, or primary-aware hypothesis assessment.
