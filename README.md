# KuaiRand — дипломный проект

Репозиторий дипломной работы по рекомендательным системам на датасете KuaiRand. Цель проекта — построить воспроизводимый исследовательский pipeline: изучить данные, подготовить последовательный benchmark-протокол и получить первые baseline-результаты для дальнейшего сравнения с более сильными sequential/causal методами.

## Данные

Используется KuaiRand: датасет коротких видео с логами взаимодействий пользователей и item metadata. В проекте важны два режима сбора:

- `standard` interactions — обычные рекомендации платформы, полезны для моделирования типичного пользовательского поведения;
- `random` exposure — случайные показы, потенциально важны для дальнейшей causal/off-policy части, потому что помогают отделять предпочтения от bias рекомендательной политики.

Raw datasets и большие промежуточные parquet/CSV не хранятся в Git. В репозиторий попадают только код, конфиги, compact summaries, manifests, отчёты и compact experiment results.

## EDA

Выполнен полный EDA для KuaiRand-27K. Интерактивный notebook находится в [notebooks/01_kuairand_eda.ipynb](notebooks/01_kuairand_eda.ipynb), batch-агрегации для полного 27K — в [src/eda_27k.py](src/eda_27k.py).

Основной отчёт: [reports/kuairand_27k_eda_report.md](reports/kuairand_27k_eda_report.md).

Compact результаты лежат в [outputs/eda/](outputs/eda/): `27k_summary.json` и небольшие `27k_*.csv`. Raw interaction logs и Slurm logs не коммитятся.

Повторный запуск EDA на кластере:

```bash
sbatch slurm/eda_27k.sh
```

## Экспериментальный протокол

Подготовлен Protocol B для последовательной рекомендации:

- фильтрация `5-core`;
- chronological ordering внутри пользователя;
- leave-one-out split;
- `train` = все interactions кроме двух последних;
- `validation` = предпоследняя interaction;
- `test` = последняя interaction.

Fingerprint Protocol B:

- users: `23,951`;
- items: `7,111`;
- interactions: `1,134,420`;
- train: `1,086,518`;
- validation: `23,951`;
- test: `23,951`.

Код подготовки: [src/prepare_kuairand_protocol_b.py](src/prepare_kuairand_protocol_b.py).
Конфиг RecBole: [configs/kuairand_protocol_b_recbole.yaml](configs/kuairand_protocol_b_recbole.yaml).
Отчёт: [reports/kuairand_protocol_b_data_report.md](reports/kuairand_protocol_b_data_report.md).
Manifest: [outputs/data/protocol_b_manifest.json](outputs/data/protocol_b_manifest.json).

Повторный запуск preprocessing на кластере:

```bash
sbatch slurm/prepare_protocol_b.sh
```

## Эксперименты

Экспериментальный код и результаты находятся в [experiments/](experiments/). Главная таблица результатов: [experiments/results.csv](experiments/results.csv).

Первый baseline — XGBoost LambdaMART с простыми leakage-safe popularity/history features:

- `ltr_xgb_001` использовал `sampled-100` evaluation: 1 positive + 100 sampled negatives. Он сохранён как sanity/exploratory experiment и не сопоставим с published full-ranking результатами.
- `ltr_xgb_002` использует full-ranking evaluation по `7,111` item Protocol B и является основным XGBoost baseline.

Test results для `ltr_xgb_002`:

| Model | HR@10 | NDCG@10 |
| --- | ---: | ---: |
| Random | 0.00129 | 0.00056 |
| MostPopular | 0.02952 | 0.01668 |
| XGBoost LambdaMART | 0.03140 | 0.01500 |

Подробности по baseline:

- [experiments/ltr_xgb_baseline/README.md](experiments/ltr_xgb_baseline/README.md);
- [experiments/ltr_xgb_baseline/runs/ltr_xgb_001_notes.md](experiments/ltr_xgb_baseline/runs/ltr_xgb_001_notes.md);
- [experiments/ltr_xgb_baseline/runs/ltr_xgb_002_notes.md](experiments/ltr_xgb_baseline/runs/ltr_xgb_002_notes.md).

Sequential baselines:

- TiM4Rec reproduction уже выполнен: [experiments/tim4rec_baseline/runs/tim4rec_001_notes.md](experiments/tim4rec_baseline/runs/tim4rec_001_notes.md).
- SSD4Rec подготовлен отдельной веткой и окружением: [experiments/ssd4rec_baseline/README.md](experiments/ssd4rec_baseline/README.md).

Повторный запуск baseline на кластере:

```bash
sbatch slurm/ltr_xgb_baseline.sh
```

Большие experiment artifacts остаются на cHARISMa: candidates, feature matrices, model files, rankings и Slurm logs не попадают в Git.

## Структура проекта

- [src/](src/) — preprocessing, EDA utilities и batch-скрипты;
- [configs/](configs/) — конфиги протоколов и инструментов;
- [slurm/](slurm/) — Slurm jobs для кластера;
- [reports/](reports/) — человекочитаемые отчёты;
- [outputs/](outputs/) — compact reproducible summaries и manifests;
- [experiments/](experiments/) — экспериментальный код, compact metrics и notes;
- [notebooks/](notebooks/) — исследовательские notebooks.

## Текущее состояние

1. EDA — завершён.
2. Protocol B — подготовлен и проверен.
3. XGBoost LambdaMART baseline — выполнен; `ltr_xgb_002` является актуальным full-ranking baseline.
4. TiM4Rec — воспроизведён на Protocol B.
5. SSD4Rec — подготовлен audit/smoke, полное обучение будет отдельным run.
