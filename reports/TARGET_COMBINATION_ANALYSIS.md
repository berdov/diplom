# Итоги 16 комбинаций вспомогательных задач

Проверено 10 сентября 2026: **completeness = 16/16**. Все 16 уникальных subsets завершены, gates passed; таблица сверена с каждым run JSON. Marginal effects и pairwise interactions независимо пересчитаны и совпали с summary.

Источник: [неизменённый cluster summary](evidence/target_combinations/summary.json), [индивидуальные результаты](evidence/target_combinations/runs/), [контрольные суммы копий](evidence/completed_run_hashes.json). Скопированы байт-в-байт из `/home/daryumin/iberdov/diplom_exp_target_combinations_002`; настройки и результаты на кластере не изменялись.

Code SHA: `599bcdb6e50dceea73233169b8834eaceef083a8`. Jobs: smoke `4315279`, array `4315280_0–15`, summary `4315281`. Попытка 002; предыдущая попытка сохранена отдельно.

KuaiRand-Pure / Protocol B, seed 2026, только TRAIN/VALID, полный каталог 7111 items. Loss: `L_rank + 0.13182740780834337 * mean(active auxiliary BCE)`, пустой subset — `L_rank`. Максимум 80 эпох, validation каждую эпоху, patience 5. Это отдельный factorial screening; его расписание отличается от MOO Stage 1/challengers. Нового tuning не было, TEST не загружался и не оценивался.

## Итоговая таблица

Во всех строках primary ranking присутствует. `combination` перечисляет только auxiliary; delta — абсолютная разность NDCG@10 с primary-only текущей попытки (0.0588).

| combination | n_aux | NDCG@10 | HR@10 | delta vs primary-only | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary_only | 0 | 0.0588 | 0.1076 | +0.0000 | 11 |
| click | 1 | 0.0592 | 0.1086 | +0.0004 | 17 |
| long_view | 1 | 0.0589 | 0.1081 | +0.0001 | 12 |
| like | 1 | 0.0590 | 0.1075 | +0.0002 | 11 |
| profile_enter | 1 | 0.0588 | 0.1071 | +0.0000 | 11 |
| click + long_view | 2 | 0.0593 | 0.1081 | +0.0005 | 12 |
| click + like | 2 | 0.0589 | 0.1077 | +0.0001 | 6 |
| click + profile_enter | 2 | 0.0590 | 0.1088 | +0.0002 | 12 |
| long_view + like | 2 | 0.0584 | 0.1069 | -0.0004 | 12 |
| long_view + profile_enter | 2 | 0.0590 | 0.1087 | +0.0002 | 21 |
| like + profile_enter | 2 | 0.0595 | 0.1093 | +0.0007 | 17 |
| click + long_view + like | 3 | 0.0589 | 0.1078 | +0.0001 | 17 |
| click + long_view + profile_enter | 3 | 0.0587 | 0.1077 | -0.0001 | 13 |
| click + like + profile_enter | 3 | 0.0595 | 0.1092 | +0.0007 | 17 |
| long_view + like + profile_enter | 3 | 0.0594 | 0.1099 | +0.0006 | 12 |
| click + long_view + like + profile_enter | 4 | 0.0589 | 0.1088 | +0.0001 | 9 |

## Лучшие комбинации

| Категория | Комбинация | NDCG@10 | Δ |
| --- | --- | ---: | ---: |
| Best single | click | 0.0592 | +0.0004 |
| Best pair | like + profile_enter | 0.0595 | +0.0007 |
| Best triple | click + like + profile_enter | 0.0595 | +0.0007 |
| All-four | click + long_view + like + profile_enter | 0.0589 | +0.0001 |

**Best overall: ничья** между `like + profile_enter` и `click + like + profile_enter`: 0.0595, +0.0007 (+1.19%) к primary-only. В summary пара стоит первой по порядку сортировки; это не уникальный победитель. Различия оцениваются по сохранённой точности метрик (4 знака).

## Marginal effects

Для каждой задачи t усредняется `f(S ∪ {t}) − f(S)` по всем 8 subsets остальных задач, где f — VALID NDCG@10.

| Target | Matched comparisons | Mean Δ NDCG@10 | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| is_click | 8/8 | +0.000075 | -0.000500 | +0.000500 |
| long_view | 8/8 | -0.000150 | -0.000600 | +0.000200 |
| is_like | 8/8 | +0.000100 | -0.000500 | +0.000700 |
| is_profile_enter | 8/8 | +0.000175 | -0.000600 | +0.001000 |

## Pairwise interactions

Усредняется `f(S ∪ {a,b}) − f(S ∪ {a}) − f(S ∪ {b}) + f(S)` по 4 subsets двух остальных задач.

| Pair | Matched comparisons | Mean interaction | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| is_click + long_view | 4/4 | -0.000100 | -0.000500 | +0.000600 |
| is_click + is_like | 4/4 | -0.000200 | -0.000500 | +0.000100 |
| is_click + is_profile_enter | 4/4 | -0.000450 | -0.001000 | +0.000100 |
| long_view + is_like | 4/4 | -0.000350 | -0.000700 | -0.000100 |
| long_view + is_profile_enter | 4/4 | -0.000100 | -0.000600 | +0.000500 |
| is_like + is_profile_enter | 4/4 | +0.000700 | +0.000500 | +0.000900 |

## Интерпретация и ограничения

Наибольший средний marginal effect у profile_enter (+0.000175), отрицательный — у long_view (−0.000150). Взаимодействие like/profile_enter положительно во всех четырёх matched backgrounds (среднее +0.000700); long_view/like отрицательно во всех четырёх (−0.000350). All-four (0.0589) не улучшает лучший subset.

Это описательные эффекты одного seed при фиксированных гиперпараметрах, без оценки статистической значимости. При добавлении задачи меняется знаменатель mean и вес каждой активной auxiliary: эффекты характеризуют всю зафиксированную конфигурацию, а не изолированную причинную полезность метки.

Historical regression checks summary: primary-only и четыре одиночные задачи отклонились от Stage 3 на −0.0002…+0.0003; ни одна проверка не требует расследования по зафиксированному порогу 0.001 (`interpretation_blocked_by_regression=false`). Baseline для delta в таблице — текущий primary-only, не исторический результат.
