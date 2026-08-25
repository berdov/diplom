# Multitask TiM4Rec 001

## Цель

Полный fixed-loss запуск первой собственной архитектуры `MultitaskTiM4Rec` на Protocol B.

## Отличие от TiM4Rec

Backbone полностью совпадает с `tim4rec_001`; добавлены только четыре linear behavior heads.

## Данные

- Dataset: `/home/daryumin/iberdov/diplom/data/processed/protocol_b_multitask`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.
- Train/validation/test rows: 1086518 / 23951 / 23951.

## Архитектура

- Shared representation строится только из `item_id_list`, `item_length`, `timestamp_list`.
- Heads: `Linear(64, 1)` для каждого target.
- MoE/adaptive loss/new attention/Flow Matching не использовались.

## Targets

- `is_click`, `long_view`, `is_like`, `is_profile_enter`.

## Loss

- `L_total = L_rank + lambda_aux * (L_click + L_long_view + L_like + L_profile)`.
- `lambda_aux = 0.2`.
- Loss config source: `multitask_tim4rec_sanity_001`.
- `pos_weight` locked from train: `{'is_click': 1.1633097593220878, 'long_view': 1.981802114807771, 'is_like': 52.70294582839067, 'is_profile_enter': 38.18204111071042}`.

## Обучение

- Requested epochs: 300.
- Actual epochs: 25.
- Stop reason: `early_stopping_no_improvement_10`.

## Slurm

- Job ID: `4273202`.
- Partition/node/GPU: `gpu-ef-quick` / `cn-045` / `A100`.
- State/ExitCode: `COMPLETED` / `0:0`.
- Elapsed/TimeLimit: `00:15:19` / `02:00:00`.
- Batch MaxRSS: `5028368K`.

## Early stopping

- Criterion: `ndcg@10` maximize.
- Best epoch: 14.
- Best checkpoint: `/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec/multitask_tim4rec_001/checkpoints/best_validation.pth`.

## Best validation

| metric | @5 | @10 | @20 | @50 |
| --- | ---: | ---: | ---: | ---: |
| HR | 0.0653 | 0.1061 | 0.1733 | 0.3115 |
| Recall | 0.0653 | 0.1061 | 0.1733 | 0.3115 |
| NDCG | 0.0450 | 0.0580 | 0.0749 | 0.1022 |

## Final test

- `test_evaluation_count = 1`.
| metric | @5 | @10 | @20 | @50 |
| --- | ---: | ---: | ---: | ---: |
| HR | 0.0651 | 0.1041 | 0.1663 | 0.3025 |
| Recall | 0.0651 | 0.1041 | 0.1663 | 0.3025 |
| NDCG | 0.0456 | 0.0581 | 0.0738 | 0.1006 |

## Auxiliary behavior metrics

| target | valid ROC-AUC | valid PR-AUC | test ROC-AUC | test PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| `is_click` | 0.6888 | 0.6539 | 0.6840 | 0.6507 |
| `long_view` | 0.6865 | 0.5254 | 0.6846 | 0.5239 |
| `is_like` | 0.7942 | 0.1226 | 0.7976 | 0.1276 |
| `is_profile_enter` | 0.7024 | 0.0502 | 0.6872 | 0.0437 |

## Сравнение с TiM4Rec

- Test NDCG@10 delta: -0.0017 (slightly worse).
- Validation NDCG@10 delta: -0.0013 (slightly worse).

## Сравнение с остальными baseline

| model | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MostPopular | 0.0295 | 0.0601 | 0.1030 | 0.0167 | 0.0243 | 0.0327 |
| XGBoost LambdaMART | 0.0314 | 0.0557 | 0.0999 | 0.0150 | 0.0209 | 0.0297 |
| XGBoost LambdaMART tuned | 0.0333 | 0.0574 | 0.1044 | 0.0177 | 0.0237 | 0.0330 |
| SSD4Rec | 0.1032 | 0.1683 | 0.3014 | 0.0576 | 0.0739 | 0.1002 |
| TiM4Rec | 0.1053 | 0.1696 | 0.3031 | 0.0598 | 0.0759 | 0.1022 |
| MultitaskTiM4Rec | 0.1041 | 0.1663 | 0.3025 | 0.0581 | 0.0738 | 0.1006 |

## Negative transfer

- Формально significant-флаг не выставляется без repeated seeds.
- Итоговая оценка по test NDCG@10: `slightly worse`.

## Стоимость модели

- Base params: 593498.
- Multitask params: 593758.
- Delta: 260 (0.0438%).
- Runtime: 912.22 sec.
- Peak VRAM allocated: 1955903488 bytes.

## Ограничения

- Один seed; выводы о статистической значимости не делаются.
- Loss weights и target set намеренно не тюнились.

## Вывод

- Fixed-loss multitask ranking result: `slightly worse`.
- Next step: `analyze fixed-loss negative transfer before Optuna/adaptive loss/Behavior MoE`.
