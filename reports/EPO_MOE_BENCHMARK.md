# Эксперимент EPO + MoE

Этот отчёт описывает текущий этап проекта: сравнение EPO без MoE и EPO со смесью экспертов (Mixture of Experts, MoE) только по валидационной выборке поверх той же базовой архитектуры `MultitaskTiM4Rec`. TEST должен использоваться только после фиксации EPO+MoE конфигурации.

## Постановка эксперимента

Сравниваются четыре варианта:

| Вариант | Смысл | Job ID |
| --- | --- | ---: |
| M0 | TiM4Rec + MTL + EPO без MoE | 4300861 |
| M2 | TiM4Rec + MTL + EPO + MoE с 2 experts | 4300862 |
| M4 | TiM4Rec + MTL + EPO + MoE с 4 experts | 4300863 |
| M8 | TiM4Rec + MTL + EPO + MoE с 8 experts | 4300864 |

Во всех вариантах сохраняются:

- один набор данных и один протокол B;
- один TiM4Rec backbone;
- один набор задач: `rank`, `is_click`, `long_view`, `is_like`, `is_profile_enter`;
- одна настройка EPO и одна метрика выбора модели: `ranking_operating_point.NDCG@10`;
- выбор архитектуры только по валидационной выборке.

Меняется только наличие и число experts. Текущая реализация MoE является dense: вычисляются все experts, а task-specific gate смешивает их outputs. Это не sparse top-k MoE.

## Главная таблица

| Модель | MoE | Experts | Params | Validation HR@10 | Validation NDCG@10 | Test HR@10 | Test NDCG@10 | Delta vs paper | Статус |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TiM4Rec paper | no |  |  |  |  | 0.1109 | 0.0611 | 0.0000 | опубликованный benchmark |
| Наше воспроизведение TiM4Rec | no |  |  |  |  | 0.1053 | 0.0598 | -0.0013 | существующая TEST reproduction |
| Наш TiM4Rec + multitask + EPO | no | 0 | 593758 | 0.1080 | 0.0588 |  |  |  | validation-only baseline, если не тестировался отдельно |
| Наш TiM4Rec + multitask + EPO + MoE |  |  |  |  |  |  |  |  | выбор по валидационной выборке |

Зафиксированный summary [../experiments/epo_moe/summary.json](../experiments/epo_moe/summary.json) пока не содержит validation results для M0/M2/M4/M8: все соответствующие source JSON отмечены как `missing`. Поэтому этот отчёт не выбирает архитектуру и не добавляет новых metrics.

## Валидационные запуски

| Run | Run ID | Experts | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Actual epochs | Params | TEST evals |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | `epo_moe_m0_validation_001` | 0 |  |  |  |  |  |  |  |  |  | 0 |
| M2 | `epo_moe_m2_validation_001` | 2 |  |  |  |  |  |  |  |  |  | 0 |
| M4 | `epo_moe_m4_validation_001` | 4 |  |  |  |  |  |  |  |  |  | 0 |
| M8 | `epo_moe_m8_validation_001` | 8 |  |  |  |  |  |  |  |  |  | 0 |

Пустые ячейки с метриками означают отсутствие committed artifacts, а не нулевой результат.

## Диагностика маршрутизации

| Run | Preference | Task | Dominant expert | Share | Entropy | Collapse |
| --- | --- | --- | --- | ---: | ---: | --- |

Диагностика маршрутизации будет заполняться только после появления зафиксированных validation artifacts. Для MoE важно отслеживать не только метрику ранжирования, но и использование experts: есть ли dominant expert, не возникает ли коллапс экспертов, различаются ли task-specific gates.

## Гигиена TEST

Количество TEST evaluations для текущей EPO+MoE-линии: `0`.

Архитектура, seed, learning rate, dropout, набор задач и настройка EPO не должны изменяться после frozen TEST evaluation.

## Отдельный TODO

[../experiments/epo_moe/summary.json](../experiments/epo_moe/summary.json) сейчас содержит абсолютный локальный путь в поле `config`. Поле генерируется в `experiments/epo_moe/summarize.py`, поэтому исправление требует изменения генератора и намеренно не включено в этот documentation PR.
