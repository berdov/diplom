# TiM4Rec sanity 001

## Цель

Проверить, что TiM4Rec запускается на полном KuaiRand Protocol B, обучается с включенным time-aware механизмом и дает валидные full-ranking метрики без обращения к test split. Запуск короткий: 5 эпох, это не финальный результат модели.

## Конфигурация

- Git branch: `exp/tim4rec-baseline`
- Project commit перед запуском: `701ecf072b2b33de945248753298749f5225a3e6`
- Upstream TiM4Rec commit: `8d4a6cea6a035c249a7a13999166ba41e8924abe`
- Slurm job: `4261937`, partition `test`, constraint `type_e`, node `cn-045`
- GPU: `NVIDIA A100-SXM4-80GB`
- Runtime: `00:05:30` по `sacct`, `282.0` секунд внутри скрипта
- MaxRSS: `2739432K` по `sacct`; process `ru_maxrss=2767136 KB`
- Peak VRAM: allocated `1954753024` bytes, reserved `2694840320` bytes
- Python: `3.10.14`; PyTorch: `2.3.0+cu118`; RecBole: `1.2.0`; mamba-ssm: `2.2.2`
- Dataset fingerprint совпал с Protocol B: users `23951`, items `7111`, interactions `1134420`, train `1086518`, validation `23951`, test `23951`
- Validation: full ranking, `Hit`, `Recall`, `NDCG` for `@5/@10/@20/@50`
- Test evaluation: не выполнялась
- `is_time=True`
- `learning_rate=0.001`, `lr_source=upstream_config`, `paper_learning_rate=0.01`
- Seed: `2026`; upstream KuaiRand config и paper seed не задают
- Batch sizes: train `2048`, eval `4096`
- Remote artifact path: `/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/tim4rec_sanity_001`

## Результаты по эпохам

| epoch | train loss | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | time, sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.4036 | 0.0788 | 0.1262 | 0.2234 | 0.0439 | 0.0558 | 0.0749 | 143.58 |
| 2 | 6.6913 | 0.0882 | 0.1420 | 0.2572 | 0.0494 | 0.0629 | 0.0856 | 22.19 |
| 3 | 6.5338 | 0.0932 | 0.1497 | 0.2705 | 0.0521 | 0.0663 | 0.0901 | 21.09 |
| 4 | 6.4340 | 0.1000 | 0.1597 | 0.2835 | 0.0556 | 0.0706 | 0.0951 | 21.08 |
| 5 | 6.3675 | 0.1002 | 0.1611 | 0.2894 | 0.0553 | 0.0706 | 0.0958 | 21.04 |

Лучший validation score по `NDCG@10`: epoch `4`, `0.0556`.

## Проверка time-aware механизма

- `time_diff_present=True`
- `time_diff_shape=[2048, 4, 50]`
- `time_diff_all_finite=True`
- Trainable time parameters: `10682` параметров в `26` тензорах
- Sampled gradients: finite для всех проверенных time-тензоров
- Optimizer updates: finite, sampled time-параметры реально обновились

## Проверка full-ranking validation

- Loader: `FullSortEvalDataLoader`
- Candidate universe: `7111` items без padding
- Validation rows: `23951`
- Positive targets: `23951`, ровно один positive на строку
- Raw scores finite: `raw_nan_scores=0`, `raw_inf_scores=0`
- Для всех эпох `HR@K == Recall@K` при `K in {5,10,20,50}`

## Сравнение с ориентирами

| model | HR/Recall@10 | NDCG@10 | комментарий |
|---|---:|---:|---|
| Random full-ranking | 0.0011 | 0.0004 | validation Protocol B |
| MostPopular full-ranking | 0.0300 | 0.0168 | validation Protocol B |
| XGBoost full-ranking | 0.0309 | 0.0150 | validation Protocol B |
| TiM4Rec sanity 001, best epoch | 0.1000 | 0.0556 | 5 эпох, `lr=0.001`, `is_time=True` |
| TiM4Rec paper | 0.1109 | 0.0611 | paper reference |

Sanity-run уже существенно выше локальных full-ranking baseline и близок к paper reference, но это короткий validation-only прогон и не должен использоваться как финальная оценка.

## Проблемы

- Official upstream KuaiRand config содержит `is_time=False`, хотя paper описывает time-aware модель. В этом запуске принудительно проверялся `is_time=True`.
- Learning rate расходится между upstream config (`0.001`) и paper (`0.01`). Для sanity выбран `0.001` как значение из upstream config.
- В stderr есть предупреждения RecBole/pandas `FutureWarning`; они не остановили обучение и не повлияли на finite checks.
- Upstream `ssd.py` предупреждает, что `n_heads=4` не делится на `8`, поэтому используется `nn.Conv1d`.
- Первая эпоха заняла заметно дольше последующих, вероятно из-за прогрева/инициализации; после этого эпохи занимали около `21-22` секунд.

## Решение о полном запуске

Training pipeline считается валидированным: fingerprint Protocol B совпал, модель обучается, loss падает, validation full-ranking работает, time-aware ветка получает finite gradients и optimizer updates, checkpoints сохраняются. Можно запускать основной длинный train на кластере E/`type_e`; test split не трогать до выбора финального checkpoint по validation.
