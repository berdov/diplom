# Основные результаты на KuaiRand Protocol B

Сгенерировано из `experiments/results.csv`: 2026-08-25T09:34:12.241538+00:00.

Показаны только сопоставимые full-ranking TEST results.

| Model | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | --- | --- | --- | --- | --- | --- |
| MostPopular | 0.0295 | 0.0601 | 0.1030 | 0.0167 | 0.0243 | 0.0327 |
| XGBoost LambdaMART | 0.0314 | 0.0557 | 0.0999 | 0.0150 | 0.0209 | 0.0297 |
| XGBoost LambdaMART tuned | 0.0333 | 0.0574 | 0.1044 | 0.0177 | 0.0237 | 0.0330 |
| SSD4Rec reproduction | 0.1032 | 0.1683 | 0.3014 | 0.0576 | 0.0739 | 0.1002 |
| TiM4Rec reproduction | 0.1053 | 0.1696 | 0.3031 | 0.0598 | 0.0759 | 0.1022 |
| MultitaskTiM4Rec fixed | 0.1041 | 0.1663 | 0.3025 | 0.0581 | 0.0738 | 0.1006 |
| MultitaskTiM4Rec tuned | 0.1071 | 0.1746 | 0.3138 | 0.0598 | 0.0767 | 0.1042 |

# Опубликованные результаты

| Source | Version | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TiM4Rec paper | arXiv:2409.16182v3 | 0.1109 | 0.1774 | 0.3202 | 0.0611 | 0.0779 | 0.1060 |
| SSD4Rec paper v2 | arXiv:2409.01192v2 | 0.1075 | 0.1731 |  | 0.0593 | 0.0757 |  |

# Воспроизводимость опубликованных моделей

| Model | Metric | Paper | Ours | Absolute diff | Relative diff % |
| --- | --- | --- | --- | --- | --- |
| TiM4Rec | HR@10 | 0.1109 | 0.1053 | -0.0056 | -5.05% |
| TiM4Rec | HR@20 | 0.1774 | 0.1696 | -0.0078 | -4.40% |
| TiM4Rec | HR@50 | 0.3202 | 0.3031 | -0.0171 | -5.34% |
| TiM4Rec | NDCG@10 | 0.0611 | 0.0598 | -0.0013 | -2.13% |
| TiM4Rec | NDCG@20 | 0.0779 | 0.0759 | -0.0020 | -2.57% |
| TiM4Rec | NDCG@50 | 0.1060 | 0.1022 | -0.0038 | -3.58% |
| SSD4Rec | HR@10 | 0.1075 | 0.1032 | -0.0043 | -4.00% |
| SSD4Rec | HR@20 | 0.1731 | 0.1683 | -0.0048 | -2.77% |
| SSD4Rec | NDCG@10 | 0.0593 | 0.0576 | -0.0017 | -2.87% |
| SSD4Rec | NDCG@20 | 0.0757 | 0.0739 | -0.0018 | -2.38% |

# Sanity и диагностические запуски

| Run | Model | Variant | Split | Evaluation | Status | NDCG@10 | Test count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_smoke_001 | MultitaskTiM4Rec | adaptive_gradient_smoke | train | diagnostic | completed |  | 0 |
| metabalance_sanity_001 | MultitaskTiM4Rec | adaptive_metabalance_fix | validation | full_7111_items | completed | 0.0518 | 0 |
| multitask_tim4rec_sanity_001 | MultitaskTiM4Rec | sanity_5_epoch | validation | full_7111_items | completed | 0.0557 | 0 |
| optuna_smoke_001 | XGBoost LambdaMART | optuna_smoke | validation | full_7111_items | completed | 0.0150 | 0 |
| pcgrad_sanity_001 | MultitaskTiM4Rec | adaptive_pcgrad_ranking_anchored | validation | full_7111_items | completed | 0.0568 | 0 |
| smoke_20260818T132855Z | TiM4Rec | smoke_forward | validation | full_7111_items | completed |  | 0 |
| smoke_20260819T110252Z | SSD4Rec | smoke_forward | validation | full_7111_items | completed |  | 0 |
| ssd4rec_sanity_001 | SSD4Rec | sanity_5_epoch | validation | full_7111_items | completed | 0.0559 | 0 |
| target_audit_001 | Multitask target audit | target_labels_audit | train | diagnostic | completed |  | 0 |
| tim4rec_sanity_001 | TiM4Rec | sanity_5_epoch | validation | full_7111_items | completed | 0.0556 | 0 |

# Hyperparameter search

| Study | Trials complete | Trials pruned | Trials failed | Best trial | Best validation NDCG@10 | Test used? |
| --- | --- | --- | --- | --- | --- | --- |
| MultitaskTiM4Rec Optuna search | 60 | 61 | 0 | 110 | 0.0599 | no |
| XGBoost Optuna search | 40 | 0 | 1 | 16 | 0.0184 | no |

# Полный реестр запусков

| Type | Run | Source | Model | Variant | Split | Evaluation | Status | HR@10 | NDCG@10 | Test count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiment | ltr_xgb_001 | ours | XGBoost LambdaMART | sampled_history | test | sampled_100 | completed | 0.4948 | 0.2853 | 1 |
| experiment | ltr_xgb_002 | ours | XGBoost LambdaMART | baseline_full_ranking | test | full_7111_items | completed | 0.0314 | 0.0150 | 1 |
| experiment | ltr_xgb_optuna_001 | ours | XGBoost LambdaMART | tuned_optuna | test | full_7111_items | completed | 0.0333 | 0.0177 | 1 |
| experiment | mostpop_001 | ours | MostPopular | sampled_history | test | sampled_100 | completed | 0.4956 | 0.2858 | 1 |
| experiment | mostpop_002 | ours | MostPopular | full_ranking_history | test | full_7111_items | completed | 0.0295 | 0.0167 | 1 |
| experiment | multitask_tim4rec_001 | ours | MultitaskTiM4Rec | fixed_loss | test | full_7111_items | completed | 0.1041 | 0.0581 | 1 |
| experiment | multitask_tim4rec_tuned_001 | ours | MultitaskTiM4Rec | tuned_fixed_weights | test | full_7111_items | completed | 0.1071 | 0.0598 | 1 |
| experiment | random_001 | ours | Random | sampled_history | test | sampled_100 | completed | 0.1002 | 0.0453 | 1 |
| experiment | random_002 | ours | Random | full_ranking_history | test | full_7111_items | completed | 0.0013 | 0.0006 | 1 |
| experiment | ssd4rec_001 | ours | SSD4Rec | reproduction | test | full_7111_items | completed | 0.1032 | 0.0576 | 1 |
| experiment | tim4rec_001 | ours | TiM4Rec | reproduction | test | full_7111_items | completed | 0.1053 | 0.0598 | 1 |
| search | multitask_optuna_search_001 | ours | MultitaskTiM4Rec | optuna_search | validation | full_7111_items | completed | 0.1093 | 0.0599 | 0 |
| search | optuna_search_001 | ours | XGBoost LambdaMART | optuna_search | validation | full_7111_items | completed | 0.0343 | 0.0184 | 0 |
| sanity | adaptive_smoke_001 | ours | MultitaskTiM4Rec | adaptive_gradient_smoke | train | diagnostic | completed |  |  | 0 |
| sanity | metabalance_sanity_001 | ours | MultitaskTiM4Rec | adaptive_metabalance_fix | validation | full_7111_items | completed | 0.0951 | 0.0518 | 0 |
| sanity | multitask_tim4rec_sanity_001 | ours | MultitaskTiM4Rec | sanity_5_epoch | validation | full_7111_items | completed | 0.1011 | 0.0557 | 0 |
| sanity | optuna_smoke_001 | ours | XGBoost LambdaMART | optuna_smoke | validation | full_7111_items | completed | 0.0308 | 0.0150 | 0 |
| sanity | pcgrad_sanity_001 | ours | MultitaskTiM4Rec | adaptive_pcgrad_ranking_anchored | validation | full_7111_items | completed | 0.1036 | 0.0568 | 0 |
| sanity | smoke_20260818T132855Z | ours | TiM4Rec | smoke_forward | validation | full_7111_items | completed |  |  | 0 |
| sanity | smoke_20260819T110252Z | ours | SSD4Rec | smoke_forward | validation | full_7111_items | completed |  |  | 0 |
| sanity | ssd4rec_sanity_001 | ours | SSD4Rec | sanity_5_epoch | validation | full_7111_items | completed | 0.1008 | 0.0559 | 0 |
| sanity | target_audit_001 | ours | Multitask target audit | target_labels_audit | train | diagnostic | completed |  |  | 0 |
| sanity | tim4rec_sanity_001 | ours | TiM4Rec | sanity_5_epoch | validation | full_7111_items | completed | 0.1000 | 0.0556 | 0 |
| paper_reference | paper_ssd4rec_v2 | paper | SSD4Rec | official_paper | reference | paper_reference | published | 0.1075 | 0.0593 |  |
| paper_reference | paper_tim4rec | paper | TiM4Rec | official_paper | reference | paper_reference | published | 0.1109 | 0.0611 |  |
