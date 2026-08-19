# TiM4Rec tim4rec_001

## Цель

Выполнить полноценный reproduction run TiM4Rec на KuaiRand Protocol B без подбора гиперпараметров. Model selection выполнялся только по validation `NDCG@10`; test split был оценен один раз после загрузки best validation checkpoint.

## Данные и протокол

- Dataset: KuaiRand Protocol B chronological leave-one-out.
- Fingerprint: users `23951`, items `7111`, interactions `1134420`, train `1086518`, validation `23951`, test `23951`.
- `MAX_ITEM_LIST_LENGTH=50`.
- Evaluation: full-ranking по `7111` items, не sampled-100.
- Validation rows: `23951`, positive targets: `23951`.
- Test rows: `23951`, positive targets: `23951`.
- Для validation и test проверено `HR@K == Recall@K` для `K in {5,10,20,50}`.
- Non-finite scores: `0` на validation и test.

## Конфигурация

- Branch: `exp/tim4rec-baseline`.
- Run metadata commit: `1301e2581295bb86f1ea9582ed38a9edfd1149fa`.
- Upstream TiM4Rec commit: `8d4a6cea6a035c249a7a13999166ba41e8924abe`.
- `is_time=True`.
- `learning_rate=0.001`, `lr_source=upstream_config`, `paper_learning_rate=0.01`.
- `seed=2026`; paper/upstream KuaiRand config seed не задают.
- Batch sizes: train `2048`, eval `4096`.
- Architecture: hidden size `64`, layers `2`, dropout `0.2`, `d_state=32`, `d_conv=4`, `expand=2`, `head_dim=32`, `chunk_size=32`, `is_ffn=True`, `p2p_residual=False`.

## Обучение

Запрошено `300` эпох. Early stopping: validation metric `NDCG@10`, `stopping_step=10`, validation после каждой эпохи.

| epoch | loss | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | es step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.4035 | 0.0787 | 0.1263 | 0.2233 | 0.0438 | 0.0558 | 0.0749 | 0 |
| 2 | 6.6913 | 0.0882 | 0.1422 | 0.2572 | 0.0495 | 0.0630 | 0.0857 | 0 |
| 3 | 6.5337 | 0.0934 | 0.1497 | 0.2698 | 0.0522 | 0.0664 | 0.0900 | 0 |
| 4 | 6.4339 | 0.1000 | 0.1599 | 0.2841 | 0.0556 | 0.0706 | 0.0952 | 0 |
| 5 | 6.3674 | 0.1002 | 0.1612 | 0.2897 | 0.0553 | 0.0707 | 0.0959 | 1 |
| 6 | 6.3227 | 0.1035 | 0.1679 | 0.2971 | 0.0572 | 0.0734 | 0.0988 | 0 |
| 7 | 6.2900 | 0.1050 | 0.1691 | 0.3004 | 0.0575 | 0.0736 | 0.0995 | 0 |
| 8 | 6.2647 | 0.1043 | 0.1702 | 0.3026 | 0.0578 | 0.0744 | 0.1005 | 0 |
| 9 | 6.2425 | 0.1050 | 0.1689 | 0.3052 | 0.0577 | 0.0737 | 0.1006 | 1 |
| 10 | 6.2251 | 0.1055 | 0.1724 | 0.3086 | 0.0578 | 0.0746 | 0.1014 | 0 |
| 11 | 6.2098 | 0.1061 | 0.1715 | 0.3074 | 0.0578 | 0.0742 | 0.1010 | 0 |
| 12 | 6.1979 | 0.1086 | 0.1740 | 0.3126 | 0.0593 | 0.0757 | 0.1030 | 0 |
| 13 | 6.1859 | 0.1072 | 0.1736 | 0.3116 | 0.0584 | 0.0751 | 0.1023 | 1 |
| 14 | 6.1760 | 0.1057 | 0.1734 | 0.3107 | 0.0579 | 0.0749 | 0.1020 | 2 |
| 15 | 6.1666 | 0.1071 | 0.1740 | 0.3105 | 0.0586 | 0.0754 | 0.1024 | 3 |
| 16 | 6.1587 | 0.1088 | 0.1733 | 0.3109 | 0.0586 | 0.0748 | 0.1020 | 4 |
| 17 | 6.1522 | 0.1048 | 0.1762 | 0.3128 | 0.0573 | 0.0752 | 0.1021 | 5 |
| 18 | 6.1452 | 0.1073 | 0.1749 | 0.3131 | 0.0581 | 0.0751 | 0.1023 | 6 |
| 19 | 6.1403 | 0.1064 | 0.1766 | 0.3149 | 0.0575 | 0.0751 | 0.1024 | 7 |
| 20 | 6.1346 | 0.1072 | 0.1761 | 0.3164 | 0.0583 | 0.0757 | 0.1033 | 8 |
| 21 | 6.1301 | 0.1056 | 0.1744 | 0.3155 | 0.0576 | 0.0748 | 0.1027 | 9 |
| 22 | 6.1249 | 0.1052 | 0.1749 | 0.3151 | 0.0576 | 0.0751 | 0.1028 | 10 |
| 23 | 6.1210 | 0.1071 | 0.1775 | 0.3166 | 0.0582 | 0.0759 | 0.1033 | 11 |

Stop reason: `early_stopping_no_improvement_10`.

## Лучшая эпоха

Best epoch: `12`.

Validation metrics на best epoch:

| metric | value |
|---|---:|
| HR/Recall@5 | 0.0654 |
| HR/Recall@10 | 0.1086 |
| HR/Recall@20 | 0.1740 |
| HR/Recall@50 | 0.3126 |
| NDCG@5 | 0.0454 |
| NDCG@10 | 0.0593 |
| NDCG@20 | 0.0757 |
| NDCG@50 | 0.1030 |

## Итоговый test

Best validation checkpoint был загружен перед test: `/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/tim4rec_001/checkpoints/best_validation.pth`, checkpoint epoch `12`.

| metric | value |
|---|---:|
| HR@5 / Recall@5 | 0.0683 |
| HR@10 / Recall@10 | 0.1053 |
| HR@20 / Recall@20 | 0.1696 |
| HR@50 / Recall@50 | 0.3031 |
| NDCG@5 | 0.0479 |
| NDCG@10 | 0.0598 |
| NDCG@20 | 0.0759 |
| NDCG@50 | 0.1022 |

## Сравнение с опубликованным TiM4Rec

| metric | ours | paper | abs diff | rel diff |
|---|---:|---:|---:|---:|
| Recall@10 | 0.1053 | 0.1109 | -0.0056 | -5.05% |
| Recall@20 | 0.1696 | 0.1774 | -0.0078 | -4.40% |
| Recall@50 | 0.3031 | 0.3202 | -0.0171 | -5.34% |
| NDCG@10 | 0.0598 | 0.0611 | -0.0013 | -2.13% |
| NDCG@20 | 0.0759 | 0.0779 | -0.0020 | -2.57% |
| NDCG@50 | 0.1022 | 0.1060 | -0.0038 | -3.58% |

## Сравнение с нашими baseline

| Model | HR/Recall@10 | HR/Recall@20 | HR/Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| Random full-ranking | 0.0013 | 0.0030 | 0.0076 | 0.0006 | 0.0010 | 0.0019 |
| MostPopular full-ranking | 0.0295 | 0.0601 | 0.1030 | 0.0167 | 0.0243 | 0.0327 |
| XGBoost LambdaMART `ltr_xgb_002` | 0.0314 | 0.0557 | 0.0999 | 0.0150 | 0.0209 | 0.0297 |
| TiM4Rec `tim4rec_001` | 0.1053 | 0.1696 | 0.3031 | 0.0598 | 0.0759 | 0.1022 |
| TiM4Rec paper | 0.1109 | 0.1774 | 0.3202 | 0.0611 | 0.0779 | 0.1060 |

## Время и ресурсы

- Slurm job: `4264125`, final partition `rocky`, node `cn-046`, constraint `type_e`.
- GPU: `NVIDIA A100-SXM4-80GB`.
- Slurm elapsed: `00:13:43`.
- Script runtime: `672.22` sec.
- Sum train time: `601.95` sec.
- Sum validation time: `6.48` sec.
- Test time: `0.25` sec.
- MaxRSS by `sacct`: `2742740K`.
- Process `ru_maxrss`: `2782980 KB`.
- Peak VRAM allocated: `1954753024` bytes.
- Peak VRAM reserved: `2696937472` bytes.
- Remote artifacts: `/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/tim4rec_001`.
- Full training log: `/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/tim4rec_001/training_log.jsonl`.
- Environment snapshot: `/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/tim4rec_001/environment.json`.
- Best checkpoint size: `7197026` bytes, sha256 `e06d20bd5f13029fd31a2ba1567833da41b9cf130ded9755ff20d2f2cfb12b41`.
- Last checkpoint size: `7187534` bytes, sha256 `aff94d1664cd3e30551424d70c9b33a773b87dcf3474611c241680a874626647`.

## Расхождения с paper/upstream

- `is_time` inconsistency: official upstream KuaiRand config содержит `is_time=False`, но paper описывает time-aware модель. В этом run использовано `is_time=True`, как в sanity.
- Learning rate mismatch: upstream config использует `0.001`, paper указывает `0.01`. Для reproduction выбран `0.001`, потому что это executable upstream config value и он уже был проверен в sanity.
- Seed situation: paper/upstream KuaiRand config не фиксируют seed; использован проектный reproducible seed `2026`.
- Software versions могут отличаться от paper: Python `3.10.14`, PyTorch `2.3.0+cu118`, RecBole `1.2.0`, mamba-ssm `2.2.2`.
- В stderr есть предупреждения RecBole/pandas `FutureWarning` и upstream warning про `n_heads=4`; обучение завершилось, finite checks чистые.

## Вывод о воспроизводимости

По test результатам TiM4Rec близко воспроизведен: отклонения от paper составляют около `-2.13%` по `NDCG@10`, `-2.57%` по `NDCG@20`, `-3.58%` по `NDCG@50` и около `-4.40%..-5.34%` по Recall@K. Результат заметно выше локальных full-ranking baseline и не требует автоматического запуска `lr=0.01` или другого эксперимента.
