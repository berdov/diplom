# ltr_xgb_002 notes

## Цель

Исправить sampled evaluation из `ltr_xgb_001` и получить простой XGBoost LambdaMART baseline с full-ranking evaluation для KuaiRand Protocol B.

## Что было неправильно в 001

`ltr_xgb_001` оценивался на 101 candidate на query: 1 positive + 100 sampled negatives. Поэтому абсолютные HR/NDCG/Recall не сопоставимы с опубликованными sequential full-ranking результатами SSD4Rec/TiM4Rec.

Sanity check: при 101 candidate Random HR@10 теоретически около `10 / 101 = 0.099010`; фактически в 001 на test было `0.100246`. В 002 full-ranking Random HR@10 на test стал `0.001294`.

## Evaluation protocol

- Training candidate protocol: `sampled_100`.
- Evaluation candidate protocol: `full_7111_items`.
- Evaluation protocol: `protocol_b_full_recbole_sequential`.
- Item universe: `7111` real Protocol B items.
- Mask seen items: `False`.
- RecBole `sequential.yaml` задаёт `split: {'LS': 'valid_and_test'}`, `order: TO`, `mode: full`; default `group_by=user`.
- `leave_one_out` после timestamp ordering оставляет последние две interactions пользователя под validation и test.
- Для sequential `FullSortEvalDataLoader` возвращает `history_index=None`; `Trainer._full_sort_batch_eval` маскирует только internal item id 0. Поэтому в raw Protocol B universe history items не исключаются.
- Validation context: train history до validation target. Test context: train history + validation interaction до test target.
- Repeated target items остаются evaluable при `mask_seen_items=false`.

## Repeated targets

- Validation repeated target queries: `907`; evaluable: `907`.
- Test repeated target queries: `963`; evaluable: `963`.

## Результаты

Validation:

| Model | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.000501 | 0.001127 | 0.002923 | 0.007557 | 0.000243 | 0.000442 | 0.000887 | 0.001788 | 0.000501 | 0.001127 | 0.002923 | 0.007557 |
| MostPopular | 0.021878 | 0.029978 | 0.059079 | 0.105006 | 0.014205 | 0.016764 | 0.023990 | 0.032940 | 0.021878 | 0.029978 | 0.059079 | 0.105006 |
| XGBoost LambdaMART | 0.019790 | 0.030855 | 0.055572 | 0.098827 | 0.011351 | 0.014972 | 0.021026 | 0.029576 | 0.019790 | 0.030855 | 0.055572 | 0.098827 |

Test:

| Model | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.000668 | 0.001294 | 0.003006 | 0.007641 | 0.000365 | 0.000562 | 0.000983 | 0.001880 | 0.000668 | 0.001294 | 0.003006 | 0.007641 |
| MostPopular | 0.022170 | 0.029519 | 0.060081 | 0.103044 | 0.014354 | 0.016677 | 0.024267 | 0.032683 | 0.022170 | 0.029519 | 0.060081 | 0.103044 |
| XGBoost LambdaMART | 0.019456 | 0.031397 | 0.055697 | 0.099912 | 0.011085 | 0.015003 | 0.020946 | 0.029735 | 0.019456 | 0.031397 | 0.055697 | 0.099912 |

## Сравнение с 001

| Run | Eval candidates | Random HR@10 test | MostPopular HR@10 test | XGBoost HR@10 test |
| --- | --- | --- | --- | --- |
| ltr_xgb_001 | 1 positive + 100 sampled negatives | 0.100246 | 0.495637 | 0.494802 |
| ltr_xgb_002 | full item universe | 0.001294 | 0.029519 | 0.031397 |

## Сравнение с literature

| Source | Model | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
| --- | --- | --- | --- | --- | --- |
| SSD4Rec arXiv 2409.01192v1 Table 4 | SASRec | 0.1040 | 0.1705 | 0.0567 | 0.0733 |
| SSD4Rec arXiv 2409.01192v1 Table 4 | SSD4Rec | 0.1076 | 0.1704 | 0.0602 | 0.0759 |

Источник: SSD4Rec, arXiv:2409.01192v1, Table 4 (`https://arxiv.org/html/2409.01192v1`). Сопоставимо только если split и full-ranking semantics полностью совпадают.

## Training

- Boosting rounds requested: `1`.
- Trees trained: `1`.
- Early stopping: `None`.
- Best iteration: `None`.
- Для 002 sampled validation metric не используется для выбора test model; число деревьев фиксировано заранее, чтобы изменение 001 -> 002 отражало именно evaluation protocol.

## Slurm

| Run | Job ID | Partition | Node | CPU | GPU | Status | Runtime | MaxRSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sanity | 4257029 | cpu-e-quick / type_e | cn-044 | 8 | none | COMPLETED | 00:00:51 | 4108K |
| full | 4257031 | cpu-e-quick / type_e | cn-044 | 8 | none | COMPLETED | 00:01:53 | 766156K |

## Артефакты

- Remote artifact path: `/home/daryumin/iberdov/diplom/experiments/ltr_xgb_baseline/ltr_xgb_002`.
- Remote artifact size: `188M`; для сравнения `ltr_xgb_001` занимает `206M`.
- Full scores не сохраняются; для каждого split/model сохранён только Top-50 ranking.

## Вывод

002 измеряет ту же простую popularity/history модель при корректной full-ranking оценке. Абсолютные метрики больше не завышены sampled candidate protocol и могут служить честным слабым baseline для дальнейших sequential моделей.
