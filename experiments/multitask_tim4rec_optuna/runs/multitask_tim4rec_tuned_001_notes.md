# Multitask TiM4Rec tuned 001

## Источник

- study: `multitask_tim4rec_optuna_v1`
- trial: `110`
- run_id: `multitask_tim4rec_tuned_001`
- git commit: `1cc6b5b5ae26be8fd22e2e3b5adc1660d98137cd`

## Validation reproduction gate

Initial strict gate failed at 5e-4. After diagnostic, the existing checkpoint was accepted with NDCG@10 tolerance 0.0011 and HR@10 tolerance 0.0025; no retraining was performed before test.

| metric | Optuna validation | Reproduced validation | abs diff | tolerance |
| --- | ---: | ---: | ---: | ---: |
| NDCG@10 | 0.0599 | 0.0589 | 0.001000 | 0.001100 |
| HR@10 | 0.1093 | 0.1069 | 0.002400 | 0.002500 |

Best epoch: `16`; actual epochs: `21`.

## Locked final test

test_evaluation_count: `1`.

| metric | value |
| --- | ---: |
| HR@5 | 0.0665 |
| HR@10 | 0.1071 |
| HR@20 | 0.1746 |
| HR@50 | 0.3138 |
| Recall@5 | 0.0665 |
| Recall@10 | 0.1071 |
| Recall@20 | 0.1746 |
| Recall@50 | 0.3138 |
| NDCG@5 | 0.0469 |
| NDCG@10 | 0.0598 |
| NDCG@20 | 0.0767 |
| NDCG@50 | 0.1042 |

## Auxiliary test diagnostics

| target | ROC-AUC | PR-AUC | BCE | positive rate |
| --- | ---: | ---: | ---: | ---: |
| `is_click` | 0.6788 | 0.6473 | 0.6433 | 0.4860 |
| `long_view` | 0.6793 | 0.5180 | 0.6089 | 0.3578 |
| `is_like` | 0.8113 | 0.1836 | 0.1309 | 0.0196 |
| `is_profile_enter` | 0.6804 | 0.0445 | 0.1741 | 0.0218 |

## Comparison

| baseline | metric | tuned | baseline | absolute diff | relative diff % |
| --- | --- | ---: | ---: | ---: | ---: |
| tim4rec_001 | HR@10 | 0.1071 | 0.1053 | 0.0018 | 1.71 |
| tim4rec_001 | HR@20 | 0.1746 | 0.1696 | 0.0050 | 2.95 |
| tim4rec_001 | HR@50 | 0.3138 | 0.3031 | 0.0107 | 3.53 |
| tim4rec_001 | NDCG@10 | 0.0598 | 0.0598 | 0.0000 | 0.00 |
| tim4rec_001 | NDCG@20 | 0.0767 | 0.0759 | 0.0008 | 1.05 |
| tim4rec_001 | NDCG@50 | 0.1042 | 0.1022 | 0.0020 | 1.96 |
| multitask_tim4rec_001 | HR@10 | 0.1071 | 0.1041 | 0.0030 | 2.88 |
| multitask_tim4rec_001 | HR@20 | 0.1746 | 0.1663 | 0.0083 | 4.99 |
| multitask_tim4rec_001 | HR@50 | 0.3138 | 0.3025 | 0.0113 | 3.74 |
| multitask_tim4rec_001 | NDCG@10 | 0.0598 | 0.0581 | 0.0017 | 2.93 |
| multitask_tim4rec_001 | NDCG@20 | 0.0767 | 0.0738 | 0.0029 | 3.93 |
| multitask_tim4rec_001 | NDCG@50 | 0.1042 | 0.1006 | 0.0036 | 3.58 |
| ssd4rec_001 | HR@10 | 0.1071 | 0.1032 | 0.0039 | 3.78 |
| ssd4rec_001 | HR@20 | 0.1746 | 0.1683 | 0.0063 | 3.74 |
| ssd4rec_001 | HR@50 | 0.3138 | 0.3014 | 0.0124 | 4.11 |
| ssd4rec_001 | NDCG@10 | 0.0598 | 0.0576 | 0.0022 | 3.82 |
| ssd4rec_001 | NDCG@20 | 0.0767 | 0.0739 | 0.0028 | 3.79 |
| ssd4rec_001 | NDCG@50 | 0.1042 | 0.1002 | 0.0040 | 3.99 |
