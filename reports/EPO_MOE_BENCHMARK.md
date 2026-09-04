# EPO + MoE Benchmark

KuaiRand Protocol B. Architecture selection uses validation only; TEST is used only after the frozen EPO+MoE configuration.

## Main Table

| Model | MoE | Experts | Params | Validation HR@10 | Validation NDCG@10 | Test HR@10 | Test NDCG@10 | Delta vs paper | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TiM4Rec paper | no |  |  |  |  | 0.1109 | 0.0611 | 0.0000 | published benchmark |
| Our TiM4Rec reproduction | no |  |  |  |  | 0.1053 | 0.0598 | -0.0013 | existing TEST reproduction |
| Our TiM4Rec + multitask + EPO | no | 0 | 593758 | 0.1080 | 0.0588 |  |  |  | validation-only baseline unless separately tested |
| Our TiM4Rec + multitask + EPO + MoE |  |  |  |  |  |  |  |  | validation selection |

## Validation Runs

| Run | Run ID | Experts | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Actual epochs | Params | Test evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Routing Diagnostics

| Run | Preference | Task | Dominant | Share | Entropy | Collapse |
| --- | --- | --- | --- | --- | --- | --- |

## Test Disclosure

TEST evaluations recorded for this EPO+MoE line: `0`.
No architecture, seed, learning-rate, dropout, task-set or EPO tuning is performed after TEST.
