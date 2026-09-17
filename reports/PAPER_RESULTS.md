# Сравнение с опубликованными результатами

## Опубликованные числа (reported)

Источник чисел: [TiM4Rec, author version arXiv v3, Table 3](https://arxiv.org/html/2409.16182v3#S4.T3), KuaiRand. Это результаты из статьи, не наших запусков. Ссылки в названиях ведут к публикациям методов. [Журнальный DOI](https://doi.org/10.1016/j.neucom.2025.131270) указан отдельно: соответствие журнальной таблицы версии arXiv не проверено.

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

R@K означает Recall@K. SSD4Rec* воспроизведён авторами TiM4Rec без bidirectional и variable-length enhancements полного SSD4Rec. [Числа и происхождение](evidence/tim4rec_table3_kuairand.json).

## Наши TEST (our run)

Один seed 2026; checkpoint выбран по VALID. Хронологический leave-one-out, полный каталог, один relevant target: HR=Recall. Источники чисел указаны в названиях. Наша репродукция TiM4Rec (TEST NDCG@10 0.0598) находится в [RESULTS](RESULTS.md), а внешнее сравнение ниже использует **reported TiM4Rec 0.0611**.

| Модель | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| [Vanilla Mamba3](../experiments/mamba3_baseline/runs/mamba3_final_test_001.json) | 0.1062 | 0.1708 | 0.3053 | 0.0590 | 0.0752 | 0.1017 |
| [Shared RT-Mamba3](../experiments/mamba3_timeaware/runs/mamba3_timeaware_final_test_001.json) | 0.1116 | 0.1764 | 0.3136 | 0.0613 | 0.0776 | 0.1046 |

Shared RT выше reported TiM4Rec по NDCG@10 на +0.0002 (+0.3273%), но ниже по NDCG@20/@50 и Recall@20/@50. Округлённые числа одного seed не доказывают статистическую значимость.

<details>
<summary>Все разницы с reported TiM4Rec</summary>

Абсолютная разница = ours − reported; относительная = 100×(ours/reported − 1). В каждой ячейке: абсолютная / относительная.

| Модель | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | -0.0047 / -4.2381% | -0.0066 / -3.7204% | -0.0149 / -4.6533% | -0.0021 / -3.4370% | -0.0027 / -3.4660% | -0.0043 / -4.0566% |
| Shared RT | +0.0007 / +0.6312% | -0.0010 / -0.5637% | -0.0066 / -2.0612% | +0.0002 / +0.3273% | -0.0003 / -0.3851% | -0.0014 / -1.3208% |

</details>

## Ограничения сопоставимости

| Аспект | Наши запуски | Что известно из author v3 |
|---|---|---|
| Данные | standard 4_08_to_4_21, is_rand=0, iterative 5-core, повторы сохранены | minimum 5; точный файл/версия и обработка повторов не установлены; совпадение counts 23951/7111/1134420 недостаточно |
| Split и время | user-wise leave-one-out; history до 50, без target timestamp; ties по source_row_id | сортировка по времени и maxlen 50; exact split, ties и временная доступность не подтверждены |
| Кандидаты и маски | 7111 реальных items, padding исключён, seen items и повторные targets разрешены | full/sampled, masking и повторные targets не установлены |
| Обучение и агрегация | Adam 0.001, один seed 2026 | learning rate 0.01; seeds/aggregation/SD не установлены |

Это внешний ориентир с оговорками, **не доказанная идентичность протоколов и не заявление SOTA**. [Подробные условия наших экспериментов](EVALUATION_SETUP.md).

У separate, decay_only, scan_only и Proto-Mamba3 нет TEST: их [VALID-результаты](MAMBA3_TIME_MECHANISMS_RESULTS.md) не сравниваются с paper TEST.

## Связанные работы 2026 года

- [Multi-Task Multi-Behavior Sequential Recommendation](https://dl.acm.org/doi/10.1145/3774904.3792187) (WWW 2026): тематически релевантна, но результаты не добавлены из-за неподтверждённой сопоставимости с протоколом TiM4Rec.
- [Automated Information Flow Selection for Multi-scenario Multi-task Recommendation (AutoIFS)](https://dl.acm.org/doi/10.1145/3773966.3777992) (WSDM 2026): тематически релевантна, но результаты не добавлены по той же причине.

Эта таблица не устанавливает новизну метода.
