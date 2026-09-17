# Mamba3: результаты временных механизмов

KuaiRand — хронологический leave-one-out, оценка по полному каталогу.
[Experimental setup](EVALUATION_SETUP.md). Один seed 2026; новые результаты
только VALID, `test_evaluation_count=0`.

## Гипотеза

Нужна ли одинаковая функция physical gap для decay (`ADT`) и scan/input-weight/
rotary dynamics (`DT`)? `decay_only` и `scan_only` включают один путь,
`shared` использует одну функцию для обоих, `separate` — две независимые.
Остальные backbone, CE, scorer, reference 838393 ms и bounds [0.5,2] frozen.
Подробные формулы — [implementation README](../experiments/mamba3_time_mechanisms/README.md).

## VALID

Каждая ячейка содержит @5 / @10 / @20 / @50. Один relevant target: HR=Recall.
Числа извлечены из source JSON; original execution commits не переписаны.

| Mode | HR / Recall @5/10/20/50 | NDCG @5/10/20/50 | Best epoch (0-based) | Actual epochs | Parameters |
|---|---|---|---:|---:|---:|
| vanilla | .0648 / .1078 / .1738 / .3114 | .0446 / .0584 / .0749 / .1021 | 15 | 27 | 610440 |
| shared RT | .0682 / .1111 / .1800 / .3204 | .0468 / .0605 / .0778 / .1055 | 15 | 27 | 610506 |
| decay_only | .0688 / .1132 / .1797 / .3200 | .0470 / .0612 / .0779 / .1056 | 15 | 27 | 610506 |
| scan_only | .0684 / .1131 / .1807 / .3221 | .0468 / .0611 / .0780 / .1059 | 15 | 27 | 610506 |
| separate | .0692 / .1176 / .1896 / .3374 | .0478 / .0633 / .0813 / .1105 | 51 | 63 | 610572 |

| Mode | NDCG@10 delta vs vanilla | Relative | Delta vs shared | Relative |
|---|---:|---:|---:|---:|
| shared | +.0021 | +3.5959% | .0000 | 0.0000% |
| decay_only | +.0028 | +4.7945% | +.0007 | +1.1570% |
| scan_only | +.0027 | +4.6233% | +.0006 | +0.9917% |
| separate | +.0049 | +8.3904% | +.0028 | +4.6281% |

**Separate: best observed single-seed VALID; confirmation pending.** Не объявляем
статистическую значимость или устойчивое превосходство. Все три новых TEST не оценены.

## Обучение и бюджет

![Истории VALID](assets/time_mechanisms/learning_curves.svg)

Истории всех пяти runs сохранились. Separate в первых 27 эпохах (0–26):
**0.0614 на эпохе 23**, delta к shared +.0009 (+1.4876%). Итоговые .0633
получены позже, на эпохе 51 (52-я эпоха). Поэтому сравнение .0633 с максимумом
shared за 27 эпох не изолирует влияние дополнительного времени обучения.
Правила max300/patience10 одинаковы, фактический бюджет различается.
Vanilla имеет равные округлённые максимумы на 11 и 15; сохранён checkpoint 15,
что подтверждает existing final-test provenance. График отмечает выбранный checkpoint.

| Mode | Actual epochs | Время между timestamps run JSON, секунды |
|---|---:|---:|
| vanilla | 27 | 386.96 |
| shared | 27 | 414.00 |
| decay_only | 27 | 697.01 |
| scan_only | 27 | 668.43 |
| separate | 63 | 1150.84 |

Это время runner, не Slurm allocation и не только training kernel time.
Per-epoch training/VALID timings сохранены в [данных графиков](assets/time_mechanisms/extracted.json).

## Функции calibrator

![Scale против physical gap](assets/time_mechanisms/calibrator_curves.svg)

На CPU загружены только наши доверенные best checkpoints; извлечены параметры
calibrator. Сетка: 121 log-spaced положительных gaps между frozen TRAIN min/max,
плюс reference и отдельный active zero. Ни Mamba forward, ни ranking evaluation
не выполнялись. Все пять checkpoint доступны; пути, SHA256, source/log SHA,
маленькие веса и значения функций — в [provenance](assets/time_mechanisms/extracted.json).
[Извлечение](assets/time_mechanisms/extract.py), [рендер](assets/time_mechanisms/render.py).

Active zero-gap отмечен отдельной точкой; first event и padding всегда дают 1,
не смешиваются с такими точками. Кривые функций на сетке не являются
распределением коэффициентов на реальных VALID histories.

## Распределения на VALID

Именно `best_diagnostics` исходных JSON: 724401 active history-gap occurrences,
padding/first исключены. Mean/std/min/max точные; quantiles приблизительные,
reservoir 8192 с собственным RNG. Повторяющиеся gaps в prefixes учитываются повторно.

| Mode/path | Head | Mean | Std | p10 | p50 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| decay_only/decay | 0 | 1.999910 | .000102 | 1.999745 | 1.999971 | 2.000000 |
| decay_only/decay | 1 | .836214 | .439993 | .502807 | .612231 | 1.642123 |
| scan_only/scan | 0 | .765376 | .405650 | .505826 | .543320 | 1.513760 |
| scan_only/scan | 1 | .787525 | .126271 | .593296 | .823963 | .933462 |
| separate/decay | 0 | 1.138654 | .450420 | .519576 | 1.300120 | 1.628739 |
| separate/decay | 1 | .908804 | .495633 | .504781 | .637370 | 1.794623 |
| separate/scan | 0 | .632056 | .303651 | .501674 | .502632 | 1.014333 |
| separate/scan | 1 | .800682 | .052129 | .766373 | .786225 | .878568 |

Decay-only head 0 почти насыщен у upper bound 2; separate scan head 0 часто
близок к lower bound. Это описательная диагностика, не повод менять bounds
после наблюдения результата. Для shared сохранились checkpoint functions,
но отдельное VALID scale distribution в исходном run отсутствует.

Separate: pooled log-correlation **−0.536664**, mean absolute log-difference
**0.635848**, доля decay>scan **0.566871**. Pooled означает объединение heads;
per-head correlation на VALID не сохранена и не восстанавливается по grid.
Эта корреляция не доказывает независимые физические механизмы или периодичность.

## Источники и ограничения

- [Decay JSON](../experiments/mamba3_time_mechanisms/runs/mamba3_decay_only_validation_001.json), job 4332918.
- [Scan JSON](../experiments/mamba3_time_mechanisms/runs/mamba3_scan_only_validation_001.json), job 4332919.
- [Separate JSON](../experiments/mamba3_time_mechanisms/runs/mamba3_separate_time_validation_001.json), job 4332920.
- Submission commit `9334bd93c8fa30fbacabaeb220cef17e336c261b`.
- Core fingerprint `460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f`.
- [Неизменённый GPU evidence](../experiments/mamba3_time_mechanisms/runs/gpu_equivalence_001.json).

Один seed, разные early-stopping горизонты и наблюдавшийся ранее shared TEST
ограничивают выводы. Следующий этап — заранее фиксированное matched-seed
подтверждение shared/separate и отдельный exploratory constant-gap control.
Новых TEST нет. Внешние paper TEST сравниваются только с завершёнными нашими
TEST в [PAPER_RESULTS.md](PAPER_RESULTS.md), не с .0633 VALID.
