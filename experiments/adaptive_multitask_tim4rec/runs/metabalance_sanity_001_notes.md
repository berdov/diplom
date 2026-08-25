# metabalance_sanity_001

## Test safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Training идёт на validation-only RecBole benchmark: только `train` и `valid`.

## Метод

- Method: `MetaBalance`.
- Exact variant: `MetaBalance-Fix`.
- Shared selector: `all_backbone`.
- Epochs: `5`.

## Tuned fixed стартовая конфигурация

- Study: `multitask_tim4rec_optuna_v1`.
- Trial: `110`.
- `lambda_aux`: `0.131827407808`.
- `learning_rate`: `0.00190753706681`.
- `weight_decay`: `1.92550265674e-06`.
- `dropout_prob`: `0.0856570671989`.
- `head_lr_multiplier`: `0.458510582586`.

## Validation best

| HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|
| 0.0951 | 0.1511 | 0.2665 | 0.0518 | 0.0659 | 0.0886 |

## Epoch trajectory

| epoch | L_total | L_rank | L_click | L_long | L_like | L_profile | HR@10 | NDCG@10 | train sec | valid sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.6155 | 7.3708 | 0.6579 | 0.6827 | 0.4300 | 0.5026 | 0.0761 | 0.0431 | 160.4 | 0.3 |
| 2 | 6.9945 | 6.7734 | 0.6419 | 0.6666 | 0.3790 | 0.4805 | 0.0847 | 0.0471 | 77.8 | 0.3 |
| 3 | 6.8583 | 6.6489 | 0.6342 | 0.6588 | 0.3537 | 0.4677 | 0.0893 | 0.0498 | 77.7 | 0.3 |
| 4 | 6.7833 | 6.5822 | 0.6268 | 0.6509 | 0.3364 | 0.4564 | 0.0893 | 0.0498 | 77.3 | 0.3 |
| 5 | 6.7268 | 6.5325 | 0.6206 | 0.6443 | 0.3224 | 0.4454 | 0.0951 | 0.0518 | 77.3 | 0.3 |

## Auxiliary validation at best epoch

| target | ROC-AUC | PR-AUC | BCE | positive rate |
|---|---:|---:|---:|---:|
| `is_click` | 0.6914 | 0.6566 | 0.6394 | 0.4826 |
| `long_view` | 0.6879 | 0.5286 | 0.6109 | 0.3574 |
| `is_like` | 0.8039 | 0.1613 | 0.1785 | 0.0196 |
| `is_profile_enter` | 0.6957 | 0.0496 | 0.1997 | 0.0237 |

## Rank-aux conflicts

- До adaptive update: `0.238054`.
- После adaptive update: `0.218882`.
- Negative rank-vs-aux counts before: `{'is_click': 369, 'long_view': 398, 'is_like': 817, 'is_profile_enter': 887}`.

## Diagnostic points

| epoch | rank-click | rank-long | rank-like | rank-profile | rank-aux conflict fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.002822 | 0.005418 | -0.002681 | 0.013976 | 0.2500 |
| 3 | 0.007276 | 0.039199 | 0.010419 | 0.022221 | 0.0000 |
| 5 | 0.045849 | 0.031129 | 0.022746 | -0.010778 | 0.2500 |

## Cost

- Mean train epoch: `94.127` sec.
- Mean validation: `0.285` sec.
- Peak VRAM: `5268573184` bytes.
- Process MaxRSS: `5027608` KB.
