# Master Project Status Summary

Эта таблица отвечает на вопрос, что уже сделано в проекте. Она не смешивает paper, locked test и validation-only results без явных `Source`, `Stage` и `Split`.

## A. PAPER RESULTS

| Method | Source | Family | Backbone | Stage | Split | Status | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Paper SSD4Rec v1 | SSD4Rec arXiv v1 Table 4 | SSD / SSM | published | paper_benchmark | paper_test | PUBLISHED | 0.1076 | 0.1704 |  | 0.0602 | 0.0759 |  |  | external_paper | Historical reproduction target for this project. |
| Paper SSD4Rec v2 | SSD4Rec current arXiv v2 Table 4 | SSD / SSM | published | paper_benchmark | paper_test | PUBLISHED | 0.1075 | 0.1731 |  | 0.0593 | 0.0757 |  |  | external_paper | Updated SSD4Rec paper version. Kept as published version, not treated as an error versus v1. |
| Paper TiM4Rec | TiM4Rec arXiv v3 Table 3 | Time-aware SSD | published | paper_benchmark | paper_test | PUBLISHED | 0.1109 | 0.1774 | 0.3202 | 0.0611 | 0.0779 | 0.1060 |  | external_paper | TiM4Rec Table 3 same-pipeline published row. |

## B. OUR TEST REPRODUCTIONS

| Method | Source | Family | Backbone | Stage | Split | Status | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSD4Rec | repository_artifact | sequential_reproduction | SSD4Rec | locked_test | test | CANONICAL | 0.1032 | 0.1683 | 0.3014 | 0.0576 | 0.0739 | 0.1002 | 17 | 1 | experiments/ssd4rec_baseline/runs/ssd4rec_001_notes.md |
| TiM4Rec | repository_artifact | sequential_reproduction | TiM4Rec | locked_test | test | CANONICAL | 0.1053 | 0.1696 | 0.3031 | 0.0598 | 0.0759 | 0.1022 | 12 | 1 | experiments/tim4rec_baseline/runs/tim4rec_001_notes.md |

## C. MULTITASK DEVELOPMENT

| Method | Source | Family | Backbone | Stage | Split | Status | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MultitaskTiM4Rec | repository_artifact | multitask_sequential | TiM4Rec | locked_test | test | CANONICAL | 0.1041 | 0.1663 | 0.3025 | 0.0581 | 0.0738 | 0.1006 | 14 | 1 | experiments/multitask_tim4rec/runs/multitask_tim4rec_001_notes.md |
| MultitaskTiM4Rec | repository_artifact | multitask_sequential | TiM4Rec | locked_test | test | CANONICAL | 0.1071 | 0.1746 | 0.3138 | 0.0598 | 0.0767 | 0.1042 | 16 | 1 | experiments/multitask_tim4rec_optuna/runs/multitask_tim4rec_tuned_001_notes.md |

## D. HISTORICAL ADAPTIVE METHODS

| Method | Source | Family | Backbone | Stage | Split | Status | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MultitaskTiM4Rec | repository_artifact | gradient_manipulation | MultitaskTiM4Rec | historical | validation | HISTORICAL | 0.1082 | 0.1744 | 0.3089 | 0.0586 | 0.0752 | 0.1018 | 9 | 0 | experiments/adaptive_multitask_tim4rec/runs/pcgrad_001_notes.md; historical PCGrad reference, not current family decision |
| MultitaskTiM4Rec | repository_artifact | multitask_sequential | TiM4Rec | sanity | train | IMPLEMENTED_NOT_EVALUATED | — | — | — | — | — | — |  | 0 | experiments/adaptive_multitask_tim4rec/runs/adaptive_smoke_001_notes.md |
| GradNorm | repository_artifact | adaptive_loss_balancing | MultitaskTiM4Rec | implementation_smoke | train | IMPLEMENTED_NOT_EVALUATED | — | — | — | — | — | — |  | 0 | GradNorm code and smoke diagnostics exist inside adaptive_smoke_001; no validation/test metric row is available. |

## E. 8-FAMILY LONG CONVERGENCE

| Method | Source | Family | Backbone | Stage | Split | Status | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STCH | repository_artifact | loss_balancing | MultitaskTiM4Rec | convergence_screening | validation | CANONICAL | 0.0749 | 0.1163 | 0.2082 | 0.0424 | 0.0528 | 0.0709 | 80 | 0 | experiments/moo_8families/runs/stch_convergence_001_notes.md; canonical only for validation-only MOO family screening |
| FAMO | repository_artifact | gradient_weighting | MultitaskTiM4Rec | convergence_screening | validation | CANONICAL | 0.0719 | 0.1102 | 0.1935 | 0.0412 | 0.0508 | 0.0672 | 15 | 0 | experiments/moo_8families/runs/famo_convergence_001_notes.md; canonical only for validation-only MOO family screening |
| PCGrad | repository_artifact | gradient_manipulation | MultitaskTiM4Rec | convergence_screening | validation | CANONICAL | 0.0790 | 0.1259 | 0.2253 | 0.0444 | 0.0562 | 0.0757 | 25 | 0 | experiments/moo_8families/runs/pcgrad_convergence_001_notes.md; canonical only for validation-only MOO family screening |
| EPO | repository_artifact | finite_preference_set | MultitaskTiM4Rec | convergence_screening | validation | CANONICAL | 0.1078 | 0.1767 | 0.3171 | 0.0584 | 0.0756 | 0.1033 | 15 | 0 | experiments/moo_8families/runs/epo_convergence_001_notes.md; canonical only for validation-only MOO family screening |
| HV-Gradient / GradHV-style | repository_artifact | finite_no_preference_set | MultitaskTiM4Rec | convergence_screening | validation | CANONICAL | 0.0874 | 0.1382 | 0.2440 | 0.0486 | 0.0613 | 0.0820 | 50 | 0 | experiments/moo_8families/runs/gradhv_convergence_001_notes.md; canonical only for validation-only MOO family screening |
| PHN-adapter | repository_artifact | infinite_hypernetwork | MultitaskTiM4Rec | convergence_screening | validation | CANONICAL | 0.0746 | 0.1155 | 0.2027 | 0.0423 | 0.0526 | 0.0698 | 60 | 0 | experiments/moo_8families/runs/phn_convergence_001_notes.md; canonical only for validation-only MOO family screening |
| COSMOS-style | repository_artifact | infinite_preference_conditioned | MultitaskTiM4Rec | convergence_screening | validation | CANONICAL | 0.0810 | 0.1257 | 0.2252 | 0.0453 | 0.0565 | 0.0761 | 25 | 0 | experiments/moo_8families/runs/cosmos_convergence_001_notes.md; canonical only for validation-only MOO family screening |
| PaLoRA | repository_artifact | infinite_model_combination | MultitaskTiM4Rec | convergence_screening | validation | CANONICAL | 0.0750 | 0.1159 | 0.2080 | 0.0422 | 0.0525 | 0.0706 | 35 | 0 | experiments/moo_8families/runs/palora_convergence_001_notes.md; canonical only for validation-only MOO family screening |

## F. CURRENT TUNING

| Method | Source | Family | Backbone | Stage | Split | Status | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Test evaluated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EPO | planned | finite_preference_set | MultitaskTiM4Rec | controlled_tuning | validation | READY_TO_SUBMIT_ON_CLUSTER_AFTER_REVIEW |  |  |  |  |  |  |  | 0 | planned controlled tuning, not launched from local Mac |
| HV-Gradient / GradHV-style | planned | finite_no_preference_set | MultitaskTiM4Rec | controlled_tuning | validation | READY_TO_SUBMIT_ON_CLUSTER_AFTER_REVIEW |  |  |  |  |  |  |  | 0 | planned controlled tuning, not launched from local Mac |
| COSMOS-style | planned | infinite_preference_conditioned | MultitaskTiM4Rec | controlled_tuning | validation | READY_TO_SUBMIT_ON_CLUSTER_AFTER_REVIEW |  |  |  |  |  |  |  | 0 | planned controlled tuning, not launched from local Mac |
| PCGrad | planned | gradient_manipulation | MultitaskTiM4Rec | controlled_tuning | validation | READY_TO_SUBMIT_ON_CLUSTER_AFTER_REVIEW |  |  |  |  |  |  |  | 0 | allowed only after discrepancy audit conclusion; not launched from local Mac |
