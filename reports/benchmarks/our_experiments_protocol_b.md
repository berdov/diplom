# Наши эксперименты: KuaiRand Protocol B

Эта таблица содержит только наши project runs из [experiments/results.csv](../../experiments/results.csv) и compact run artifacts. Literature values сюда не включены.

Protocol B fingerprint: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Canonical TEST runs

| Experiment ID | Method | Status | Stage | Split | HR@10 | NDCG@10 | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_002 | Random | CANONICAL | locked_test | test | 0.0013 | 0.0006 | 1 | — |
| mostpop_002 | MostPopular | CANONICAL | locked_test | test | 0.0295 | 0.0167 | 1 | — |
| ltr_xgb_002 | XGBoost LambdaMART | CANONICAL | locked_test | test | 0.0314 | 0.0150 | 1 | experiments/ltr_xgb_baseline/runs/ltr_xgb_002_notes.md |
| ssd4rec_001 | SSD4Rec | CANONICAL | locked_test | test | 0.1032 | 0.0576 | 1 | experiments/ssd4rec_baseline/runs/ssd4rec_001_notes.md |
| tim4rec_001 | TiM4Rec | CANONICAL | locked_test | test | 0.1053 | 0.0598 | 1 | experiments/tim4rec_baseline/runs/tim4rec_001_notes.md |
| multitask_tim4rec_001 | MultitaskTiM4Rec | CANONICAL | locked_test | test | 0.1041 | 0.0581 | 1 | experiments/multitask_tim4rec/runs/multitask_tim4rec_001_notes.md |
| multitask_tim4rec_tuned_001 | MultitaskTiM4Rec | CANONICAL | locked_test | test | 0.1071 | 0.0598 | 1 | experiments/multitask_tim4rec_optuna/runs/multitask_tim4rec_tuned_001_notes.md |

## Текущий 8-family convergence screening

Это validation-only family screening; `test_evaluation_count=0` во всех восьми current rows. Итоговая фигура: [current_moo_validation_ndcg10.png](current_moo_validation_ndcg10.png).

| Family | Representative | Fidelity | Best epoch | NDCG@10 | HR@10 | HV | Non-dominated | Spread | Runtime | Peak VRAM GB | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss_balancing | STCH | exact_or_close_reproduction | 80 | 0.0424 | 0.0749 | 0.2431 | 1 | — | 00:47:52 | 2.254 | BORDERLINE |
| gradient_weighting | FAMO | exact_or_close_reproduction | 15 | 0.0412 | 0.0719 | 0.2642 | 1 | — | 00:20:33 | 2.254 | DROP |
| gradient_manipulation | PCGrad | exact_or_close_reproduction | 25 | 0.0444 | 0.0790 | 0.2676 | 1 | — | 01:11:36 | 2.254 | BORDERLINE |
| finite_preference_set | EPO | exact_or_close_reproduction | 15 | 0.0584 | 0.1078 | 0.3068 | 4 | 0.3014 | 05:03:30 | 2.265 | PROMISING |
| finite_no_preference_set | HV-Gradient / GradHV-style | family-level adaptation | 50 | 0.0486 | 0.0874 | 0.2997 | 2 | 0.2731 | 01:27:44 | 4.886 | PROMISING |
| infinite_hypernetwork | PHN-adapter | family-level adaptation | 60 | 0.0423 | 0.0746 | 0.2659 | 2 | 0.0030 | 00:40:03 | 2.255 | DROP |
| infinite_preference_conditioned | COSMOS-style | method-level adaptation | 25 | 0.0453 | 0.0810 | 0.2585 | 5 | 0.0190 | 00:24:10 | 2.256 | PROMISING |
| infinite_model_combination | PaLoRA | method-level adaptation | 35 | 0.0422 | 0.0750 | 0.2598 | 1 | — | 00:30:08 | 2.260 | BORDERLINE |

## Historical / exploratory / sanity rows

### Historical

| Experiment ID | Method | Status | Stage | Split | HR@10 | NDCG@10 | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pcgrad_001 | MultitaskTiM4Rec | HISTORICAL | historical | validation | 0.1082 | 0.0586 | 0 | experiments/adaptive_multitask_tim4rec/runs/pcgrad_001_notes.md; historical PCGrad reference, not current family decision |

### Exploratory

| Experiment ID | Method | Status | Stage | Split | HR@10 | NDCG@10 | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ltr_xgb_optuna_001 | XGBoost LambdaMART | EXPLORATORY | locked_test | test | 0.0333 | 0.0177 | 1 | experiments/ltr_xgb_optuna/runs/ltr_xgb_optuna_001_notes.md |
| random_001 | Random | EXPLORATORY | locked_test | test | 0.1002 | 0.0453 | 1 | sampled-100; not comparable with full-ranking paper results |
| mostpop_001 | MostPopular | EXPLORATORY | locked_test | test | 0.4956 | 0.2858 | 1 | sampled-100; not comparable with full-ranking paper results |
| ltr_xgb_001 | XGBoost LambdaMART | EXPLORATORY | locked_test | test | 0.4948 | 0.2853 | 1 | experiments/ltr_xgb_baseline/runs/ltr_xgb_001_notes.md; sampled-100; not comparable with full-ranking paper results |
| optuna_search_001 | XGBoost LambdaMART | EXPLORATORY | validation_search | validation | 0.0343 | 0.0184 | 0 | experiments/ltr_xgb_optuna/runs/optuna_search_001_notes.md; validation search; no test evaluation in this row |
| multitask_optuna_search_001 | MultitaskTiM4Rec | EXPLORATORY | validation_search | validation | 0.1093 | 0.0599 | 0 | experiments/multitask_tim4rec_optuna/runs/multitask_optuna_search_001_notes.md; validation search; no test evaluation in this row |

### Sanity

| Experiment ID | Method | Status | Stage | Split | HR@10 | NDCG@10 | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| behavior_moe_sanity_001 | BehaviorMoETiM4Rec | SANITY | sanity | validation | 0.1027 | 0.0562 | 0 | experiments/behavior_moe_tim4rec/runs/behavior_moe_sanity_001_notes.md |
| metabalance_sanity_001 | MultitaskTiM4Rec | SANITY | sanity | validation | 0.0951 | 0.0518 | 0 | experiments/adaptive_multitask_tim4rec/runs/metabalance_sanity_001_notes.md |
| multitask_tim4rec_sanity_001 | MultitaskTiM4Rec | SANITY | sanity | validation | 0.1011 | 0.0557 | 0 | experiments/multitask_tim4rec/runs/multitask_tim4rec_sanity_001_notes.md |
| pcgrad_sanity_001 | MultitaskTiM4Rec | SANITY | sanity | validation | 0.1036 | 0.0568 | 0 | experiments/adaptive_multitask_tim4rec/runs/pcgrad_sanity_001_notes.md |
| ssd4rec_sanity_001 | SSD4Rec | SANITY | sanity | validation | 0.1008 | 0.0559 | 0 | experiments/ssd4rec_baseline/runs/ssd4rec_sanity_001_notes.md |
| tim4rec_sanity_001 | TiM4Rec | SANITY | sanity | validation | 0.1000 | 0.0556 | 0 | experiments/tim4rec_baseline/runs/tim4rec_sanity_001_notes.md |
| stch_sanity_001 | STCH | SANITY | sanity | validation | 0.0580 | 0.0321 | 0 | experiments/moo_8families/runs/stch_sanity_001_notes.md |
| famo_sanity_001 | FAMO | SANITY | sanity | validation | 0.0660 | 0.0377 | 0 | experiments/moo_8families/runs/famo_sanity_001_notes.md |
| epo_sanity_001 | EPO | SANITY | sanity | validation | 0.1050 | 0.0577 | 0 | experiments/moo_8families/runs/epo_sanity_001_notes.md |
| gradhv_sanity_001 | HV-Gradient / GradHV-style | SANITY | sanity | validation | 0.0795 | 0.0452 | 0 | experiments/moo_8families/runs/gradhv_sanity_001_notes.md |
| phn_sanity_001 | PHN-adapter | SANITY | sanity | validation | 0.0630 | 0.0358 | 0 | experiments/moo_8families/runs/phn_sanity_001_notes.md |
| cosmos_sanity_001 | COSMOS-style | SANITY | sanity | validation | 0.0743 | 0.0423 | 0 | experiments/moo_8families/runs/cosmos_sanity_001_notes.md |
| palora_sanity_001 | PaLoRA | SANITY | sanity | validation | 0.0618 | 0.0352 | 0 | experiments/moo_8families/runs/palora_sanity_001_notes.md |

### Implemented but not evaluated

| Experiment ID | Method | Status | Stage | Split | HR@10 | NDCG@10 | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_smoke_001 | MultitaskTiM4Rec | IMPLEMENTED_NOT_EVALUATED | sanity | train | — | — | 0 | experiments/adaptive_multitask_tim4rec/runs/adaptive_smoke_001_notes.md |
| gradnorm_in_adaptive_smoke_001 | GradNorm | IMPLEMENTED_NOT_EVALUATED | implementation_smoke | train | — | — | 0 | GradNorm code and smoke diagnostics exist inside adaptive_smoke_001; no validation/test metric row is available. |

PLE не добавлен как row: в текущем коде/артефактах не найдено отдельной PLE implementation marker или run artifact. Поэтому status `IMPLEMENTED_NOT_EVALUATED` для PLE не применяется, чтобы не создавать фиктивный объект.

## Notes on demotion

- `random_001`, `mostpop_001`, `ltr_xgb_001` являются sampled-100 rows; они сохранены как exploratory и не сопоставляются с published full-ranking benchmarks.
- `pcgrad_001` сохранён как historical validation-only reference; current comparable row для family decision is `pcgrad_convergence_001`.
- `adaptive_smoke_001` и GradNorm smoke diagnostics подтверждают implementation mechanics, но не дают validation/test recommendation metrics.
- MOO convergence rows имеют `CANONICAL` только внутри validation-only screening, не как TEST ranking benchmark.
- Smoke/diagnostic rows без самостоятельного benchmark result не включены в main CSV: `behavior_moe_smoke_001`, `structured_behavior_moe_smoke_001`, `target_audit_001`, `optuna_smoke_001`, `smoke_20260818T132855Z`, `smoke_20260819T110252Z`. Failed benchmark rows в текущем `experiments/results.csv` не найдены.

Machine-readable table: [our_experiments_protocol_b.csv](our_experiments_protocol_b.csv).
