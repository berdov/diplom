# pcgrad_sanity_001

## Test safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Training идёт на validation-only RecBole benchmark: только `train` и `valid`.

## Метод

- Method: `PCGrad`.
- Exact variant: `ranking_anchored`.
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
| 0.1036 | 0.1656 | 0.3007 | 0.0568 | 0.0723 | 0.0990 |

## Epoch trajectory

| epoch | L_total | L_rank | L_click | L_long | L_like | L_profile | HR@10 | NDCG@10 | train sec | valid sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.4370 | 7.1839 | 0.6825 | 0.7071 | 0.4446 | 0.5147 | 0.0871 | 0.0486 | 148.4 | 0.7 |
| 2 | 6.7869 | 6.5490 | 0.6569 | 0.6816 | 0.4148 | 0.4945 | 0.0994 | 0.0548 | 62.7 | 0.3 |
| 3 | 6.6243 | 6.3932 | 0.6550 | 0.6797 | 0.3994 | 0.4919 | 0.1010 | 0.0555 | 62.7 | 0.3 |
| 4 | 6.5333 | 6.3104 | 0.6541 | 0.6788 | 0.3804 | 0.4911 | 0.1014 | 0.0556 | 62.5 | 0.3 |
| 5 | 6.4734 | 6.2567 | 0.6530 | 0.6776 | 0.3661 | 0.4903 | 0.1036 | 0.0568 | 63.0 | 0.3 |

## Auxiliary validation at best epoch

| target | ROC-AUC | PR-AUC | BCE | positive rate |
|---|---:|---:|---:|---:|
| `is_click` | 0.6837 | 0.6478 | 0.6406 | 0.4826 |
| `long_view` | 0.6805 | 0.5185 | 0.6075 | 0.3574 |
| `is_like` | 0.8034 | 0.1101 | 0.1528 | 0.0196 |
| `is_profile_enter` | 0.6856 | 0.0451 | 0.1744 | 0.0237 |

## Rank-aux conflicts

- До adaptive update: `0.277842`.
- После adaptive update: `0.130250`.
- Negative rank-vs-aux counts before: `{'is_click': 469, 'long_view': 552, 'is_like': 885, 'is_profile_enter': 978}`.

## Diagnostic points

| epoch | rank-click | rank-long | rank-like | rank-profile | rank-aux conflict fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.002822 | 0.005418 | -0.002681 | 0.013976 | 0.2500 |
| 3 | 0.070678 | 0.124876 | -0.013719 | 0.060680 | 0.2500 |
| 5 | 0.004221 | 0.034950 | -0.002634 | -0.032694 | 0.5000 |

## Cost

- Mean train epoch: `79.865` sec.
- Mean validation: `0.361` sec.
- Peak VRAM: `2472281088` bytes.
- Process MaxRSS: `5031712` KB.
