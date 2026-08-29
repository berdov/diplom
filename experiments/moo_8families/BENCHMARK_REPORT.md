# MOO 8 Families Benchmark Report

Отчёт сгенерирован из `experiments/moo_8families/runs/*.json`.

## Table A: Ranking Operating Point

| family | method | point | selection | oracle? | run | stage | HR@10 | NDCG@10 | oracle NDCG@10 | best epoch | test eval |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| loss_balancing | STCH |  | single_solution | False | `stch_sanity_001` | sanity | 0.0580 | 0.0321 | 0.0321 | 5 | 0 |
| gradient_weighting | FAMO |  | single_solution | False | `famo_sanity_001` | sanity | 0.0660 | 0.0377 | 0.0377 | 5 | 0 |
| gradient_manipulation | PCGrad | historical_pcgrad | single_solution | False | `pcgrad_001` | historical | 0.1082 | 0.0586 | 0.0586 | 9 | 0 |
| finite_preference_set | EPO | rank_heavy | predefined_preference_id:rank_heavy | False | `epo_sanity_001` | sanity | 0.1050 | 0.0577 | 0.0577 | 4 | 0 |
| finite_no_preference_set | HV-Gradient / GradHV-style |  | best_validation_NDCG@10_among_preference_free_finite_solutions | True | `gradhv_sanity_001` | sanity | 0.0795 | 0.0452 | 0.0452 | 5 | 0 |
| infinite_hypernetwork | PHN-adapter | rank_heavy | predefined_preference_id:rank_heavy | False | `phn_sanity_001` | sanity | 0.0630 | 0.0358 | 0.0359 | 5 | 0 |
| infinite_preference_conditioned | COSMOS-style | rank_heavy | predefined_preference_id:rank_heavy | False | `cosmos_sanity_001` | sanity | 0.0743 | 0.0423 | 0.0423 | 5 | 0 |
| infinite_model_combination | PaLoRA | rank_heavy | predefined_preference_id:rank_heavy | False | `palora_sanity_001` | sanity | 0.0618 | 0.0352 | 0.0354 | 5 | 0 |

## Oracle Best

| run | ranking point | oracle point | ranking NDCG@10 | oracle NDCG@10 | differs |
|---|---|---|---:|---:|---:|
| `stch_sanity_001` |  |  | 0.0321 | 0.0321 | False |
| `famo_sanity_001` |  |  | 0.0377 | 0.0377 | False |
| `pcgrad_001` | historical_pcgrad | historical_pcgrad | 0.0586 | 0.0586 | False |
| `epo_sanity_001` | rank_heavy | rank_heavy | 0.0577 | 0.0577 | False |
| `gradhv_sanity_001` |  |  | 0.0452 | 0.0452 | False |
| `phn_sanity_001` | rank_heavy | click_heavy | 0.0358 | 0.0359 | True |
| `cosmos_sanity_001` | rank_heavy | rank_heavy | 0.0423 | 0.0423 | False |
| `palora_sanity_001` | rank_heavy | balanced | 0.0352 | 0.0354 | True |

## Compute Cost

| run | wall sec | peak VRAM GB | model count | backward passes/batch |
|---|---:|---:|---:|---:|
| `stch_sanity_001` | 230.2 | 2.302 | 1 | 1 |
| `famo_sanity_001` | 362.1 | 2.254 | 1 | 1 |
| `pcgrad_001` |  |  |  |  |
| `epo_sanity_001` | 2405.0 | 2.312 | 6 | 2 |
| `gradhv_sanity_001` | 449.7 | 4.934 | 3 | 1 |
| `phn_sanity_001` | 333.2 | 2.255 | 1 | 1 |
| `cosmos_sanity_001` | 333.2 | 2.256 | 1 | 1 |
| `palora_sanity_001` | 363.6 | 2.260 | 1 | 1 |

## Raw Data

- Summary CSV: `experiments/moo_8families/runs/summary.csv`.
- NDCG plot: `experiments/moo_8families/figures/validation_ndcg10.png`.
- Cost plot: `experiments/moo_8families/figures/wall_time_sec.png`.
