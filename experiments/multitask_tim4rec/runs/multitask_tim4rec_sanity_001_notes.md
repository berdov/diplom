# Multitask TiM4Rec sanity 001

## Цель

Проверить первую собственную `MultitaskTiM4Rec` на полном Protocol B в коротком 5-epoch sanity run без test evaluation.

## Git и запуск

- Branch: `exp/multitask-tim4rec`.
- Source commit at run submit: `66e19ab1893dbb02f256b28ca0de9f95c7f7f65f`.
- Slurm job: `4273181` on `gpu-ef-quick`, node `cn-045`.
- GPU: `NVIDIA A100-SXM4-80GB`; peak allocated 1955903488 bytes, reserved 2969567232 bytes.
- Runtime: 387.02 sec total, 29.90 sec/epoch mean.

## Данные

- Dataset: `/home/daryumin/iberdov/diplom/data/processed/protocol_b_multitask`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.
- Train/validation/test rows: 1086518 / 23951 / 23951.
- Test evaluation count: `0`.
- Validation full-ranking rows: 23951; source ids match: `True`.

## Архитектура и loss

- Backbone: validated TiM4Rec, без MoE, adaptive loss, нового attention и Flow Matching.
- Shared representation строится только из `item_id_list`, `item_length`, `timestamp_list`.
- Heads: четыре `Linear(64, 1)` для `is_click`, `long_view`, `is_like`, `is_profile_enter`.
- Loss: `L_total = L_rank + lambda_aux * (L_click + L_long_view + L_like + L_profile)`.
- `lambda_aux = 0.2`; `pos_weight` used: `True`.

## Smoke и gradients

- Batch size: 2048.
- First-batch aux/rank ratio: 0.1019.
- Losses finite: `True`; combined gradients finite: `True`.
- All heads updated after optimizer step: `True`.
- Aux gradients reached shared backbone for every auxiliary loss.

## Training history

| epoch | L_total | L_rank | aux/rank | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 | HR@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.2359 | 7.4174 | 0.1104 | 0.0345 | 0.0445 | 0.0558 | 0.0755 | 0.0805 |
| 2 | 7.4912 | 6.7082 | 0.1167 | 0.0383 | 0.0494 | 0.0621 | 0.0841 | 0.0892 |
| 3 | 7.3220 | 6.5569 | 0.1167 | 0.0401 | 0.0518 | 0.0659 | 0.0896 | 0.0943 |
| 4 | 7.2141 | 6.4617 | 0.1164 | 0.0423 | 0.0543 | 0.0687 | 0.0933 | 0.0986 |
| 5 | 7.1380 | 6.3949 | 0.1162 | 0.0433 | 0.0557 | 0.0706 | 0.0959 | 0.1011 |

## Auxiliary validation на best epoch

| target | ROC-AUC | PR-AUC | BCE | random PR baseline |
| --- | ---: | ---: | ---: | ---: |
| `is_click` | 0.6874 | 0.6502 | 0.6382 | 0.4826 |
| `long_view` | 0.6841 | 0.5227 | 0.6414 | 0.3574 |
| `is_like` | 0.8073 | 0.1209 | 0.4729 | 0.0196 |
| `is_profile_enter` | 0.6956 | 0.0468 | 0.5130 | 0.0237 |

## Сравнение с TiM4Rec

- Multitask best sanity NDCG@10: 0.0557 at epoch 5.
- TiM4Rec sanity last NDCG@10: 0.0553; delta: 0.0004.
- TiM4Rec full `tim4rec_001` best NDCG@10: 0.0593; delta: -0.0036.
- Significant negative transfer flag: `False` with threshold `-0.01`.

## Стоимость модели

- Base params: 593498.
- Multitask params: 593758.
- Delta params: 260 (0.0438%).

## Решение

- Pipeline correct: `True`.
- Auxiliary tasks learn: `True`.
- Ready for full fixed-loss run: `True`.
- Next step: `full fixed-loss MultitaskTiM4Rec`.
