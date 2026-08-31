# KuaiRand multi-objective sequential recommendation

Дипломный проект по последовательным рекомендательным системам на KuaiRand. Основная цель - воспроизводимый Protocol B benchmark и сравнение baseline/SSD/TiM4Rec с multitask-подходом и текущим multi-objective benchmark.

## Dataset / Protocol B

Используется KuaiRand с 5-core фильтрацией, chronological ordering внутри пользователя и leave-one-out split.

| Split | Rows |
| --- | ---: |
| Users | 23,951 |
| Items | 7,111 |
| Interactions | 1,134,420 |
| Train | 1,086,518 |
| Validation | 23,951 |
| Test | 23,951 |

Код подготовки: [src/prepare_kuairand_protocol_b.py](src/prepare_kuairand_protocol_b.py). Manifest и compact fingerprints лежат в [outputs/data](outputs/data). Отчёт по протоколу: [reports/kuairand_protocol_b_data_report.md](reports/kuairand_protocol_b_data_report.md).

## Baselines

В репозитории сохранены воспроизводимые реализации и compact results для:

- SSD4Rec reproduction: [experiments/ssd4rec_baseline](experiments/ssd4rec_baseline);
- TiM4Rec reproduction: [experiments/tim4rec_baseline](experiments/tim4rec_baseline);
- XGBoost/Random/MostPopular full-ranking baselines: [experiments/ltr_xgb_baseline](experiments/ltr_xgb_baseline).

## Current Method

Текущий исследовательский фокус: `MultitaskTiM4Rec` и benchmark восьми MOO families: STCH, FAMO, PCGrad, EPO, GradHV, PHN, COSMOS и PaLoRA. Активная tuning-инфраструктура для EPO/GradHV/COSMOS/PCGrad хранится в [experiments/moo_8families](experiments/moo_8families), [configs/moo_tuning_spaces.yaml](configs/moo_tuning_spaces.yaml) и [slurm/moo_tuning.sh](slurm/moo_tuning.sh).

## Results

Machine-readable source of truth: [experiments/results.csv](experiments/results.csv).

Human-readable project summary: [reports/RESULTS.md](reports/RESULTS.md).

Published benchmark: [reports/PAPER_RESULTS.md](reports/PAPER_RESULTS.md).

8-family MOO benchmark: [reports/MOO_FAMILIES.md](reports/MOO_FAMILIES.md).

## Repository Structure

- [src](src) - preprocessing and EDA code;
- [configs](configs) - dataset and tuning configs;
- [experiments](experiments) - experiment code plus compact canonical results;
- [outputs](outputs) - dataset manifests and compact fingerprints;
- [reports](reports) - final human-readable reports;
- [slurm](slurm) - cluster entrypoints for Protocol B, canonical reproductions and current MOO work;
- [notebooks](notebooks) - EDA notebook.
