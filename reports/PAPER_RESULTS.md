# Сравнение с опубликованными результатами

## A. Опубликованный benchmark

Все числа ниже — **reported in TiM4Rec, Table 3**, KuaiRand, а не результаты
наших запусков. Ссылки в названиях ведут к исходным публикациям методов;
источник чисел — [author version arXiv v3, Table 3](https://arxiv.org/html/2409.16182v3#S4.T3).
[Journal DOI](https://doi.org/10.1016/j.neucom.2025.131270):
**author version arXiv v3; journal-table equivalence not verified**.
Журнальная таблица недоступна для проверки; препринт не выдаётся за publisher PDF.

| Метод | Год | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Caser](https://dl.acm.org/doi/10.1145/3159652.3159656) | 2018 | 0.0801 | 0.1344 | 0.2561 | 0.0395 | 0.0531 | 0.0770 |
| [GRU4Rec](https://openreview.net/forum?id=yoffK5KZSgQ) | 2016 | 0.1020 | 0.1659 | 0.3017 | 0.0564 | 0.0724 | 0.0911 |
| [SASRec](https://doi.org/10.1109/ICDM.2018.00035) | 2018 | 0.1055 | 0.1704 | 0.3074 | 0.0584 | 0.0747 | 0.1016 |
| [BERT4Rec](https://dl.acm.org/doi/10.1145/3357384.3357895) | 2019 | 0.0938 | 0.1537 | 0.2873 | 0.0510 | 0.0660 | 0.0923 |
| [TiSASRec](https://dl.acm.org/doi/10.1145/3336191.3371786) | 2020 | 0.1057 | 0.1710 | 0.3060 | 0.0590 | 0.0753 | 0.1019 |
| [LRURec](https://dl.acm.org/doi/10.1145/3616855.3635760) | 2024 | 0.1036 | 0.1663 | 0.3078 | 0.0570 | 0.0727 | 0.1005 |
| [Mamba4Rec](https://arxiv.org/abs/2403.03900) | 2024 | 0.1094 | 0.1768 | 0.3154 | 0.0608 | 0.0777 | 0.1050 |
| [SSD4Rec*](https://dl.acm.org/doi/10.1145/3773038) | 2024 | 0.1055 | 0.1717 | 0.3088 | 0.0588 | 0.0754 | 0.1024 |
| [TiM4Rec](https://www.sciencedirect.com/science/article/abs/pii/S0925231225019423) | 2025 | 0.1109 | 0.1774 | 0.3202 | 0.0611 | 0.0779 | 0.1060 |

R@K в Table 3 означает Recall@K; обозначение сохранено. SSD4Rec* — вариант
воспроизведения авторов TiM4Rec, без заявленных в полном SSD4Rec bidirectional
и variable-length enhancements. [Машиночитаемые строки и provenance](evidence/tim4rec_table3_kuairand.json).

## B. Наши завершённые TEST

Внешний reference — опубликованный **TiM4Rec**, не наша репродукция `tim4rec_001`.
Наш setup: KuaiRand — хронологический leave-one-out, оценка по полному каталогу.
Один relevant target: наши HR=Recall. Все @10/20/50 показаны, включая проигрыши.
`delta_abs=ours-published`; `delta_rel_pct=100*(ours/published-1)`.

| Наш run | Metric | Published TiM4Rec | Our value | Absolute delta | Relative delta | Source JSON | Seeds / aggregation |
|---|---|---:|---:|---:|---:|---|---|
| Vanilla | Recall@10 | 0.1109 | 0.1062 | -0.0047 | -4.2381% | [JSON](../experiments/mamba3_baseline/runs/mamba3_final_test_001.json) | 1, seed 2026 |
| Vanilla | Recall@20 | 0.1774 | 0.1708 | -0.0066 | -3.7204% | [JSON](../experiments/mamba3_baseline/runs/mamba3_final_test_001.json) | 1, seed 2026 |
| Vanilla | Recall@50 | 0.3202 | 0.3053 | -0.0149 | -4.6533% | [JSON](../experiments/mamba3_baseline/runs/mamba3_final_test_001.json) | 1, seed 2026 |
| Vanilla | NDCG@10 | 0.0611 | 0.0590 | -0.0021 | -3.4370% | [JSON](../experiments/mamba3_baseline/runs/mamba3_final_test_001.json) | 1, seed 2026 |
| Vanilla | NDCG@20 | 0.0779 | 0.0752 | -0.0027 | -3.4660% | [JSON](../experiments/mamba3_baseline/runs/mamba3_final_test_001.json) | 1, seed 2026 |
| Vanilla | NDCG@50 | 0.1060 | 0.1017 | -0.0043 | -4.0566% | [JSON](../experiments/mamba3_baseline/runs/mamba3_final_test_001.json) | 1, seed 2026 |
| Shared RT | Recall@10 | 0.1109 | 0.1116 | +0.0007 | +0.6312% | [JSON](../experiments/mamba3_timeaware/runs/mamba3_timeaware_final_test_001.json) | 1, seed 2026 |
| Shared RT | Recall@20 | 0.1774 | 0.1764 | -0.0010 | -0.5637% | [JSON](../experiments/mamba3_timeaware/runs/mamba3_timeaware_final_test_001.json) | 1, seed 2026 |
| Shared RT | Recall@50 | 0.3202 | 0.3136 | -0.0066 | -2.0612% | [JSON](../experiments/mamba3_timeaware/runs/mamba3_timeaware_final_test_001.json) | 1, seed 2026 |
| Shared RT | NDCG@10 | 0.0611 | 0.0613 | +0.0002 | +0.3273% | [JSON](../experiments/mamba3_timeaware/runs/mamba3_timeaware_final_test_001.json) | 1, seed 2026 |
| Shared RT | NDCG@20 | 0.0779 | 0.0776 | -0.0003 | -0.3851% | [JSON](../experiments/mamba3_timeaware/runs/mamba3_timeaware_final_test_001.json) | 1, seed 2026 |
| Shared RT | NDCG@50 | 0.1060 | 0.1046 | -0.0014 | -1.3208% | [JSON](../experiments/mamba3_timeaware/runs/mamba3_timeaware_final_test_001.json) | 1, seed 2026 |

Regression check: shared TEST NDCG@10 0.0613 против paper 0.0611:
**+0.0002 / +0.3273%**, не +2.51%. Последнее число относится только к нашей
внутренней репродукции TiM4Rec 0.0598, сохранённой в [реестре](../experiments/results.csv).
Округлённые метрики одного seed не доказывают статистическую значимость.

Separate (VALID 0.0633), decay_only, scan_only и Proto-Mamba3 не имеют нового
TEST: TEST и delta к paper TEST **не оценены**.

## C. Сопоставимость и ограничения

| Аспект | Наши Mamba3 runs | Что подтверждено в author v3 |
|---|---|---|
| Источник | standard log 4_08_to_4_21, is_rand=0 | Точный файл/версия не установлены; описание randomized не подтверждает наш источник |
| Фильтрация | iterative 5-core, duplicates сохранены | minimum 5; детали duplicates неизвестны |
| Counts | 23951 / 7111 / 1134420 | Совпадают с Table 2, но это не доказательство идентичности |
| Split | user-wise chronological leave-one-out | Сортировка по времени; exact held-out split не подтверждён |
| Candidates | full catalog 7111 | exact full/sampled protocol не подтверждён текстом |
| Masking | padding исключён, seen items НЕ исключены | Не установлено |
| Повторные targets / ties | разрешены; source_row_id для равного времени | Не установлено |
| History | до 50; target timestamp не используется | maxlen 50; детали time leakage/masks не установлены |
| Aggregation | один seed 2026, без seed averaging | seeds/aggregation/SD не установлены |
| Optimization | Adam 0.001 | В тексте указано 0.01: parity не заявляется |

Подробный [Experimental setup](EVALUATION_SETUP.md) основан на manifest и
установленном коде. Это оговорённый reported benchmark, **не verified apples-to-apples
comparison и не заявление SOTA**. Наши данные/маски для сходства с paper не менялись.

## D. Отдельные VALID-абляции

[Таблица и графики временных механизмов](MAMBA3_TIME_MECHANISMS_RESULTS.md),
[все внутренние результаты](RESULTS.md). VALID не сравнивается с paper TEST.

## Связанные работы 2026 года

- [Multi-Task Multi-Behavior Sequential Recommendation](https://dl.acm.org/doi/10.1145/3774904.3792187) (WWW 2026) тематически близка к проекту, но её результаты не добавлены в таблицу: текущий экспериментальный протокол не подтверждён как прямо сопоставимый с KuaiRand benchmark из TiM4Rec.
- [Automated Information Flow Selection for Multi-scenario Multi-task Recommendation (AutoIFS)](https://dl.acm.org/doi/10.1145/3773966.3777992) (WSDM 2026) также релевантна тематически, но не используется как численная строка сравнения без отдельной проверки совместимости протокола.

Не следует делать заявления о новизне только из этой таблицы. Она показывает внешний контекст, а не доказывает преимущество текущего метода проекта.
