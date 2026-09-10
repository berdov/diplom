# Сводка результатов проекта

Сводка обновлена 10 сентября 2026 по сохранённым артефактам. Машиночитаемый реестр — [../experiments/results.csv](../experiments/results.csv): 28 исторических строк сохранены, добавлены 3 challenger convergence и 16 target-combination validation screening. Новое обучение, tuning и оценка TEST не проводились. [Аудит обновления и происхождение данных](CANONICAL_RESULTS_AUDIT.md).

## Как читать этапы

| Этап | `record_type` в canonical CSV | Состав и границы сравнения | Подробности |
| --- | --- | --- | --- |
| Stage 1 convergence | `convergence_screening` | 8 исходных представителей; VALID, seed 2026; smoke/sanity не входят | [Обоснование семейств](MOO_FAMILIES.md), [история запусков](MOO_EXPERIMENT_HISTORY.md) |
| Stage 2 tuned MOO | `tuning_budgeted_validation_only` | 4 итога поиска, разный фактический бюджет; VALID | [История и ограничения tuning](MOO_EXPERIMENT_HISTORY.md) |
| Stage 3 auxiliary diagnostics | `stage3_auxiliary_analysis` | 6 исторических диагностических запусков; VALID | [Диагностика задач и градиентов](STAGE3_AUXILIARY_ANALYSIS.md) |
| Challenger convergence | `challenger_convergence` | 3 завершённых запуска; сравнение с Stage 1, один seed; fairness MosT/GradHV открыта | [Сравнение представителей](MOO_REPRESENTATIVE_CHALLENGERS.md) |
| Target-combination validation screening | `target_combination_validation_screening` | 16/16 subsets, fixed hyperparameters, один seed; собственный primary-only контроль | [Полная таблица и эффекты](TARGET_COMBINATION_ANALYSIS.md) |

Обоснование включения **восьми семейств**, литературные кандидаты, критерии выбора представителей и ограничения адаптаций находятся в [MOO_FAMILIES.md](MOO_FAMILIES.md). Это основной исследовательский документ по выбору методов; результаты и последующие решения приведены в нём отдельно. Исходная версия обоснования вошла в main через [PR #3](https://github.com/berdov/diplom/pull/3). Метрики разных этапов и historical TEST не объединяются в общий рейтинг.

## 1. Данные

Основной экспериментальный протокол — протокол B на KuaiRand-Pure / KuaiRand-27K. После 5-core фильтрации получен контрольный отпечаток `23 951 users / 7 111 items / 1 134 420 interactions`; он совпадает с ожидаемым fingerprint из [../outputs/data/protocol_b_manifest.json](../outputs/data/protocol_b_manifest.json).

| Split | Interactions | Users |
| --- | ---: | ---: |
| Train | 1 086 518 | 23 951 |
| Validation | 23 951 | 23 951 |
| Test | 23 951 | 23 951 |

Разбиение хронологическое: обучающая выборка содержит историю до двух последних взаимодействий пользователя, валидационная выборка — предпоследнее взаимодействие, тестовая выборка — последнее взаимодействие. Максимальная длина последовательности в RecBole-конфигурации равна `50`.

## 2. Базовые модели

Ниже приведены зафиксированные TEST-воспроизведения. Эти строки уже использовали тестовую выборку и не относятся к последующему выбору MOO или EPO + MoE по валидационной выборке.

| Run | Модель | Вариант | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | TEST evals |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_002` | Random | full_ranking_history | 0.0013 | 0.0030 | 0.0076 | 0.0006 | 0.0010 | 0.0019 | 1 |
| `mostpop_002` | MostPopular | full_ranking_history | 0.0295 | 0.0601 | 0.1030 | 0.0167 | 0.0243 | 0.0327 | 1 |
| `ltr_xgb_002` | XGBoost LambdaMART | baseline_full_ranking | 0.0314 | 0.0557 | 0.0999 | 0.0150 | 0.0209 | 0.0297 | 1 |
| `ltr_xgb_optuna_001` | XGBoost LambdaMART | tuned_optuna | 0.0333 | 0.0574 | 0.1044 | 0.0177 | 0.0237 | 0.0330 | 1 |
| `ssd4rec_001` | SSD4Rec | reproduction | 0.1032 | 0.1683 | 0.3014 | 0.0576 | 0.0739 | 0.1002 | 1 |

## 3. TiM4Rec

Опубликованный бенчмарк и наше воспроизведение не смешиваются:

| Источник | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TiM4Rec paper | 0.1109 | 0.1774 | 0.3202 | 0.0611 | 0.0779 | 0.1060 |
| `tim4rec_001`, наше воспроизведение | 0.1053 | 0.1696 | 0.3031 | 0.0598 | 0.0759 | 0.1022 |

Опубликованная таблица для KuaiRand benchmark находится в [PAPER_RESULTS.md](PAPER_RESULTS.md).

## 4. Многозадачное обучение

Многозадачная версия TiM4Rec добавляет выходные головы для вспомогательных сигналов `is_click`, `long_view`, `is_like` и `is_profile_enter`; основной задачей остаётся `next_item`.

| Run | Вариант | Split | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | TEST evals |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multitask_tim4rec_001` | fixed_loss | TEST | 0.1041 | 0.1663 | 0.3025 | 0.0581 | 0.0738 | 0.1006 | 1 |
| `multitask_tim4rec_tuned_001` | tuned_fixed_weights | TEST | 0.1071 | 0.1746 | 0.3138 | 0.0598 | 0.0767 | 0.1042 | 1 |
| `multitask_optuna_search_001` | optuna_search | validation | 0.1093 | 0.1722 | 0.3136 | 0.0599 | 0.0757 | 0.1036 | 0 |

`multitask_tim4rec_tuned_001` — лучший зафиксированный TEST-результат внутри текущей MTL-линии: NDCG@10 `0.0598`, HR@20 `0.1746`, NDCG@50 `0.1042`. Строка `multitask_optuna_search_001` относится к поиску по валидационной выборке и не использовала TEST.

<a id="stage1-convergence"></a>

## 5. Этап 1 MOO — Stage 1 convergence

Этап 1 — завершённые convergence-запуски представителей и адаптаций восьми MOO-семейств только по валидационной выборке. Smoke и sanity служили техническими проверками и в эту таблицу не входят. Исходный состав обоснован в [MOO_FAMILIES.md](MOO_FAMILIES.md); последующие challengers его исторические строки не заменяют.

| Run | Метод | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Actual epochs | TEST evals |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `epo_convergence_001` | EPO | 0.1078 | 0.1767 | 0.3171 | 0.0584 | 0.0756 | 0.1033 | 15 | 30 | 0 |
| `gradhv_convergence_001` | GradHV-style | 0.0874 | 0.1382 | 0.2440 | 0.0486 | 0.0613 | 0.0820 | 50 | 65 | 0 |
| `cosmos_convergence_001` | COSMOS-style | 0.0810 | 0.1257 | 0.2252 | 0.0453 | 0.0565 | 0.0761 | 25 | 40 | 0 |
| `pcgrad_convergence_001` | PCGrad | 0.0790 | 0.1259 | 0.2253 | 0.0444 | 0.0562 | 0.0757 | 25 | 40 | 0 |
| `stch_convergence_001` | STCH | 0.0749 | 0.1163 | 0.2082 | 0.0424 | 0.0528 | 0.0709 | 80 | 95 | 0 |
| `phn_convergence_001` | PHN-adapter | 0.0746 | 0.1155 | 0.2027 | 0.0423 | 0.0526 | 0.0698 | 60 | 75 | 0 |
| `palora_convergence_001` | PaLoRA | 0.0750 | 0.1159 | 0.2080 | 0.0422 | 0.0525 | 0.0706 | 35 | 50 | 0 |
| `famo_convergence_001` | FAMO | 0.0719 | 0.1102 | 0.1935 | 0.0412 | 0.0508 | 0.0672 | 15 | 30 | 0 |

Исторический `pcgrad_001` сохранён только как валидационный ориентир для ранней PCGrad-реализации. Текущая строка этапа 1 `pcgrad_convergence_001` получена другим кодом запуска и другой постановкой целевых функций.

<a id="stage2-tuned-moo"></a>

## 6. Этап 2 MOO — Stage 2 tuned MOO

Этап 2 — настройка гиперпараметров четырёх лучших методов этапа 1: EPO, GradHV, COSMOS и PCGrad. Эксперимент ограничен вычислительным бюджетом и временем, поэтому его нельзя читать как бенчмарк с равным числом завершённых запусков.

| Метод | Запланировано успешных запусков | Завершено успешных запусков | Неуспешные | Устаревшие | Лучший запуск | Best epoch | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EPO | 10 | 5 | 1 | 1 | 0 | 20 | 0.1080 | 0.1778 | 0.3198 | 0.0588 | 0.0763 | 0.1043 | исчерпан 36-часовой лимит времени |
| GradHV | 12 | 12 | 0 | 0 | 1 | 90 | 0.0877 | 0.1370 | 0.2460 | 0.0488 | 0.0612 | 0.0827 | бюджет завершён |
| COSMOS | 12 | 9 | 1 | 0 | 0 | 40 | 0.0819 | 0.1274 | 0.2289 | 0.0455 | 0.0569 | 0.0769 | остановлен защитным условием `preference_sensitivity` |
| PCGrad | 12 | 12 | 0 | 0 | 9 | 75 | 0.0828 | 0.1298 | 0.2317 | 0.0464 | 0.0581 | 0.0783 | бюджет завершён |

EPO дал лучший наблюдавшийся NDCG@10 на валидационной выборке среди четырёх методов Stage 2 в рамках его бюджета: `0.0588`. Это не доказывает абсолютное превосходство EPO над всеми MOO-методами. Устаревший незавершённый запуск EPO `0006` и неуспешный запуск COSMOS `0009` не используются как финальные результаты. Значения Stage 2 не подставляются вместо Stage 1 при сравнении с challengers; [подробная история](MOO_EXPERIMENT_HISTORY.md).

## 7. Этап 3

Этап 3 — диагностика вспомогательных задач и градиентных взаимодействий только на валидационной выборке. Основная задача — ранжирование следующего объекта.

| Run | Вспомогательная задача | HR@10 | NDCG@10 | Delta NDCG@10 |
| --- | --- | ---: | ---: | ---: |
| `stage3_primary_only_001` | нет | 0.1080 | 0.0586 | 0.0000 |
| `stage3_aux_click_001` | `is_click` | 0.1086 | 0.0593 | +0.0007 |
| `stage3_aux_long_view_001` | `long_view` | 0.1083 | 0.0586 | +0.0000 |
| `stage3_aux_like_001` | `is_like` | 0.1080 | 0.0587 | +0.0001 |
| `stage3_aux_profile_enter_001` | `is_profile_enter` | 0.1088 | 0.0590 | +0.0004 |

`is_click` дал лучший результат среди вариантов с одной вспомогательной задачей в текущем диагностическом запуске. При этом связь между частотой конфликта градиентов и приростом метрики ранжирования оказалась слабой или неоднозначной; подробнее см. [STAGE3_AUXILIARY_ANALYSIS.md](STAGE3_AUXILIARY_ANALYSIS.md). Количество TEST evaluations для этапа 3: `0`.

Отдельный historical diagnostic `stage3_all_current_aux_diagnostic_001` уже есть в canonical CSV: HR@10 **0.1089**, NDCG@10 **0.0597**, best/actual epochs **17/22**. Он использовал `tuned_task_weights` и служил диагностике градиентов. Это другая конфигурация, чем all-four в новом screening с `uniform_normalized_aux`; значение 0.0597 сохранено и не заменяется на 0.0589.

<a id="challenger-convergence"></a>

## 8. Challenger convergence — отдельное сравнение представителей

Все три запуска завершены, gates passed, только VALID, seed 2026, `TEST evals = 0`. Это convergence после технических smoke/sanity, без нового tuning. Источник каждого результата указан через `source_json` в CSV и в [специализированном отчёте](MOO_REPRESENTATIVE_CHALLENGERS.md).

| Run | Представитель | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Best epoch | Actual epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ferero_convergence_001` | FERERO-adapter | 0.1083 | 0.1757 | 0.3157 | 0.0579 | 0.0748 | 0.1024 | 15 | 30 |
| `most_convergence_001` | MosT-style | 0.0947 | 0.1554 | 0.2741 | 0.0522 | 0.0674 | 0.0908 | 10 | 25 |
| `phn_hvi_convergence_001` | PHN-HVI-adapter | 0.0789 | 0.1252 | 0.2241 | 0.0443 | 0.0560 | 0.0754 | 20 | 35 |

| Семейство | Historical Stage 1 NDCG@10 | Challenger NDCG@10 | Решение |
| --- | --- | --- | --- |
| Конечный набор с preferences | EPO 0.0584 | FERERO-adapter 0.0579 | Оставить EPO; auxiliary BCE FERERO 2.75–3.95 — отдельное ограничение, не изменение primary ranking |
| Конечный набор без preferences в оптимизаторе | GradHV-style 0.0486 | MosT-style 0.0522 | Выбор открыт; MosT — предварительный кандидат, окончательная замена GradHV отложена |
| Гиперсетевое представление | PHN-adapter 0.0423 | PHN-HVI-adapter 0.0443 | Выбрать PHN-HVI-adapter для следующего этапа, с ограничениями адаптации и одного seed |

**Fairness MosT/GradHV не закрыта:** GradHV выбирал max VALID NDCG@10 среди трёх решений; MosT — минимум фиксированной многокритериальной оценки. Правило влияет также на выбор epoch и early stopping; пересчёта только финальных точек недостаточно. Эти наблюдения не доказывают превосходство семейства или статистическую значимость различий.

Рабочий состав восьми семейств: **STCH, FAMO, PCGrad, EPO, GradHV-style / MosT-style (выбор открыт), PHN-HVI-adapter, COSMOS-style, PaLoRA**. Полные основания и статус каждого выбора: [MOO_FAMILIES.md](MOO_FAMILIES.md).

<a id="target-combination-screening"></a>

## 9. Target-combination validation screening

Завершены **16/16** уникальных subsets четырёх auxiliary, все gates passed; smoke исключён. Primary `next_item` присутствует во всех строках. Только VALID, seed 2026, `TEST evals = 0`; fixed hyperparameters, максимум 80 эпох, validation каждую эпоху, patience 5. Loss: `L_rank + 0.13182740780834337 * mean(active auxiliary BCE)`; пустой subset — `L_rank`. Расписание отличается от MOO convergence, и это отдельный этап отбора комбинаций.

| Категория | Auxiliary combination | NDCG@10 | HR@10 | Δ NDCG@10 к текущему primary-only | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| Primary-only | — | 0.0588 | 0.1076 | 0.0000 | 11 |
| Best single | click | 0.0592 | 0.1086 | +0.0004 | 17 |
| Best pair | like + profile_enter | 0.0595 | 0.1093 | +0.0007 | 17 |
| Best triple | click + like + profile_enter | 0.0595 | 0.1092 | +0.0007 | 17 |
| All-four | click + long_view + like + profile_enter | 0.0589 | 0.1088 | +0.0001 | 9 |

Best pair и best triple делят первое место при сохранённой точности метрик (4 знака). **One-seed descriptive screening:** различие +0.0007 не подтверждено несколькими seed, статистическая значимость не оценивалась. При изменении subset меняются веса отдельных auxiliary из-за усреднения; эффекты относятся ко всей конфигурации, а не к изолированной причинной полезности метки. Delta считается от primary-only **0.0588 этой попытки**, а не от Stage 3 **0.0586** или tuned EPO **0.0588**.

Все 16 строк внесены в [canonical CSV](../experiments/results.csv). [Полная таблица, marginal effects и pairwise interactions](TARGET_COMBINATION_ANALYSIS.md); [summary и исходные run JSON с контрольными суммами](evidence/README.md).

## 10. EPO + MoE — статус сохранённых артефактов

Отдельная линия экспериментов — сравнение EPO без MoE и EPO со смесью экспертов (Mixture of Experts, MoE) только по валидационной выборке:

| Вариант | Эксперты | Состояние зафиксированных artifacts |
| --- | ---: | --- |
| M0 | 0 | missing |
| M2 | 2 | missing |
| M4 | 4 | missing |
| M8 | 8 | missing |

В [../experiments/epo_moe/summary.json](../experiments/epo_moe/summary.json) результаты валидационных запусков пока отсутствуют. Таблица и правила раскрытия TEST описаны в [EPO_MOE_BENCHMARK.md](EPO_MOE_BENCHMARK.md).

## 11. Использование TEST

Этапы 1, 2, 3, challenger convergence и target-combination screening не использовали TEST для выбора методов, гиперпараметров или конфигураций. Выбор архитектуры EPO + MoE также должен выполняться только по валидационной выборке. При этом обновлении читались сохранённые результаты; новые оценки на TEST не запускались, historical TEST не использован для новых выводов.

TEST уже использовался в исторических строках базовых моделей и воспроизведений, перечисленных выше. Для нового финального метода TEST должен оставаться закрытым до frozen evaluation.

## 12. Что завершено и что остаётся

Завершены Stage 1 convergence, Stage 2 как срез с ограниченным бюджетом, Stage 3 diagnostics, три challenger convergence и screening 16/16. Литературное обоснование выбора восьми семейств зафиксировано в [MOO_FAMILIES.md](MOO_FAMILIES.md). Завершение запусков не означает окончательную фиксацию всех методологических решений.

- Закрыть fairness MosT/GradHV общим заранее заданным operating-point rule с учётом checkpoint/early stopping; пока окончательная замена не выполнена.
- Подтвердить primary-only, like + profile_enter и click + like + profile_enter несколькими seed по общему VALID-протоколу. Эти проверки ещё не выполнены.
- Разобрать auxiliary BCE FERERO отдельно от primary ranking, не меняя сохранённую рабочую точку задним числом.
- Для EPO + MoE сначала получить подтверждённые артефакты M0/M2/M4/M8; текущий cluster status этим обновлением не проверялся. Архитектура и финальный собственный метод ещё не зафиксированы.
