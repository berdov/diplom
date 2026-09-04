# История MOO-экспериментов

Этот файл сохраняет историю этапов многокритериальной оптимизации (MOO) в дипломном проекте. Все результаты Stage 1 и Stage 2 относятся к протоколу B на KuaiRand и были получены только на валидационной выборке. TEST не использовался для выбора метода, настройки гиперпараметров или выбора конфигурации.

Внутреннее имя `Stage A` в скриптах настройки соответствует научному `Stage 2` в этом отчёте: первому ограниченному по времени бюджету для настройки top-4 MOO-подходов.

Компактный источник фактов: [../experiments/moo_8families/runs/moo_stage_history_summary.json](../experiments/moo_8families/runs/moo_stage_history_summary.json). Summary был собран `2026-09-02 12:38:24` MSK. Зафиксированные SHA из summary:

| Поле | SHA |
| --- | --- |
| `local_head_before_documentation` | `ad3d08c48b5782e7fcbcb74479bb850b7ff340c4` |
| `origin_head_before_documentation` | `ad3d08c48b5782e7fcbcb74479bb850b7ff340c4` |
| `cluster_worktree_head_at_check` | `ad3d08c48b5782e7fcbcb74479bb850b7ff340c4` |

Stage 3 добавлен позднее как отдельный диагностический слой; его источник — [../experiments/stage3_auxiliary_analysis/stage3_summary.json](../experiments/stage3_auxiliary_analysis/stage3_summary.json).

## Этапы

| Этап | Научная роль | Компактный источник |
| --- | --- | --- |
| Stage 0 — контрольная MTL-модель | Настроенная fixed-weight версия `MultitaskTiM4Rec` как контекст для MOO. | `multitask_tim4rec_tuned_001` в [RESULTS.md](RESULTS.md) |
| Stage 1 — первичный отбор 8 семейств MOO | Широкий первичный отбор представителей и адаптаций восьми MOO-семейств только по валидационной выборке. | `experiments/moo_8families/runs/*_convergence_001.json` |
| Stage 2 — настройка top-4 | EPO, GradHV, COSMOS и PCGrad при ограниченном вычислительном и временном бюджете. | cHARISMa Optuna DB/logs и [moo_stage_history_summary.json](../experiments/moo_8families/runs/moo_stage_history_summary.json) |
| Stage 3 — анализ вспомогательных задач | Аудит целевых сигналов, ablations с одной вспомогательной задачей и диагностика градиентных взаимодействий для текущих MTL heads. | [STAGE3_AUXILIARY_ANALYSIS.md](STAGE3_AUXILIARY_ANALYSIS.md) |

## Stage 1 — первичный отбор 8 семейств MOO

Stage 1 был первичным отбором с ранжированием по полному набору объектов на валидационной выборке. Это не результаты TEST и не заявление о точном воспроизведении каждого опубликованного метода.

| Метод | Семейство / категория | Run | Status | Git commit | Best epoch | Actual epochs | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | TEST evals |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STCH | loss_balancing | `stch_convergence_001` | completed | `40a748b6a4f75a041d82951c61739c7f2d2f78bf` | 80 | 95 | 0.0749 | 0.1163 | 0.2082 | 0.0424 | 0.0528 | 0.0709 | 0 |
| FAMO | gradient_weighting | `famo_convergence_001` | completed | `40a748b6a4f75a041d82951c61739c7f2d2f78bf` | 15 | 30 | 0.0719 | 0.1102 | 0.1935 | 0.0412 | 0.0508 | 0.0672 | 0 |
| PCGrad | gradient_manipulation | `pcgrad_convergence_001` | completed | `40a748b6a4f75a041d82951c61739c7f2d2f78bf` | 25 | 40 | 0.0790 | 0.1259 | 0.2253 | 0.0444 | 0.0562 | 0.0757 | 0 |
| EPO | finite_preference_set | `epo_convergence_001` | completed | `40a748b6a4f75a041d82951c61739c7f2d2f78bf` | 15 | 30 | 0.1078 | 0.1767 | 0.3171 | 0.0584 | 0.0756 | 0.1033 | 0 |
| GradHV-style | finite_no_preference_set | `gradhv_convergence_001` | completed | `40a748b6a4f75a041d82951c61739c7f2d2f78bf` | 50 | 65 | 0.0874 | 0.1382 | 0.2440 | 0.0486 | 0.0613 | 0.0820 | 0 |
| PHN-adapter | infinite_hypernetwork | `phn_convergence_001` | completed | `40a748b6a4f75a041d82951c61739c7f2d2f78bf` | 60 | 75 | 0.0746 | 0.1155 | 0.2027 | 0.0423 | 0.0526 | 0.0698 | 0 |
| COSMOS-style | infinite_preference_conditioned | `cosmos_convergence_001` | completed | `40a748b6a4f75a041d82951c61739c7f2d2f78bf` | 25 | 40 | 0.0810 | 0.1257 | 0.2252 | 0.0453 | 0.0565 | 0.0761 | 0 |
| PaLoRA | infinite_model_combination | `palora_convergence_001` | completed | `40a748b6a4f75a041d82951c61739c7f2d2f78bf` | 35 | 50 | 0.0750 | 0.1159 | 0.2080 | 0.0422 | 0.0525 | 0.0706 | 0 |

Вывод Stage 1: EPO дал лучший наблюдавшийся validation NDCG@10 (`0.0584`), поэтому вместе с GradHV, COSMOS и PCGrad был передан в Stage 2. Этот выбор не означает, что остальные семейства исчерпаны или что выбранные representatives являются top-1 внутри своих семейств.

## Stage 2 — настройка top-4 MOO-подходов

Stage 2 — исследовательский срез настройки top-4 под доступный вычислительный и временной бюджет. Эксперимент считается завершённым как ограниченный по бюджету validation-only этап, но не как равный бенчмарк с одинаковым числом успешных запусков. Фактический результат каждого метода — лучший COMPLETE trial, полученный до остановки соответствующего job.

| Метод | Planned complete trials | Actual complete trials | Failed trials | Stale trials | Best trial | Best epoch | Actual epochs | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Причина остановки | Slurm status | Study/run identifier | Git commit |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| EPO | 10 | 5 | 1 (`0001`) | 1 (`0006`) | 0 | 20 | 35 | 0.1080 | 0.1778 | 0.3198 | 0.0588 | 0.0763 | 0.1043 | 36-hour Slurm walltime exhausted; stale partial trial is not counted. | `4295022 TIMEOUT` | `epo_tuning_001` | `24b7d14f01424817fc0134bd199ca69b1ad59202` |
| GradHV | 12 | 12 | 0 | 0 | 1 | 90 | 100 | 0.0877 | 0.1370 | 0.2460 | 0.0488 | 0.0612 | 0.0827 | Planned internal Stage A budget completed. | `4295023 COMPLETED 0:0` | `gradhv_tuning_001` | `24b7d14f01424817fc0134bd199ca69b1ad59202` |
| COSMOS | 12 | 9 | 1 (`0009`) | 0 | 0 | 40 | 55 | 0.0819 | 0.1274 | 0.2289 | 0.0455 | 0.0569 | 0.0769 | Scientific `preference_sensitivity` guard failed; collapsed trial is not accepted as a valid result. | `4295024 FAILED 1:0` | `cosmos_tuning_001` | `24b7d14f01424817fc0134bd199ca69b1ad59202` |
| PCGrad | 12 | 12 | 0 | 0 | 9 | 75 | 90 | 0.0828 | 0.1298 | 0.2317 | 0.0464 | 0.0581 | 0.0783 | Planned internal Stage A budget completed. | `4295025 COMPLETED 0:0` | `pcgrad_tuning_001` | `unknown` |

Удалённые source paths для лучших результатов записаны в полях `best_result_json` внутри [moo_stage_history_summary.json](../experiments/moo_8families/runs/moo_stage_history_summary.json); raw Optuna storage и большие tuning artifacts намеренно остаются вне Git.

## Переход от первичного отбора к настройке гиперпараметров

| Метод | NDCG@10 после первичного отбора | NDCG@10 после настройки | Абсолютное изменение | Относительное изменение |
| --- | ---: | ---: | ---: | ---: |
| EPO | 0.0584 | 0.0588 | +0.0004 | +0.68% |
| GradHV | 0.0486 | 0.0488 | +0.0002 | +0.41% |
| COSMOS | 0.0453 | 0.0455 | +0.0002 | +0.44% |
| PCGrad | 0.0444 | 0.0464 | +0.0020 | +4.50% |

## Интерпретация Stage 2

EPO показал лучший наблюдавшийся NDCG@10 среди настроенных MOO-методов (`0.0588`), но job упёрся в 36-hour Slurm walltime до изначально запланированных 10 COMPLETE trials. Stale partial trial `0006` не считается COMPLETE и не используется как best.

GradHV завершил полный запланированный внутренний бюджет Stage A. PCGrad также завершил бюджет и дал самый большой относительный прирост относительно конфигурации первичного отбора.

COSMOS остановился, потому что trial `0009` не прошёл научный guard `preference_sensitivity`: preference conditioning collapsed. Эта конфигурация исключена из valid best-trial selection.

Итог Stage 2 поддерживает только осторожное утверждение: в рамках имеющегося validation-only бюджета лучший наблюдавшийся MOO-результат дал EPO. Эксперимент не доказывает абсолютное превосходство одного MOO-алгоритма над другими, потому что методы имели разное число completed trials и разные причины остановки. Он также не поддерживает TEST, Stage B, multiseed или claims о новом методе.

## Stage 3 — анализ вспомогательных задач

Stage 3 — диагностический validation-only этап, а не таблица лидеров. Он проверяет доступные поведенческие targets, ablations с одной вспомогательной задачей и градиентные взаимодействия относительно основной задачи рекомендации следующего объекта. Полные детали находятся в [STAGE3_AUXILIARY_ANALYSIS.md](STAGE3_AUXILIARY_ANALYSIS.md).

Primary-only достиг validation NDCG@10 `0.0586`. Изменения запусков с одной вспомогательной задачей: `is_click` `+0.0007`, `long_view` `+0.0000`, `is_like` `+0.0001`, `is_profile_enter` `+0.0004`. Доказательства в пользу primary-aware метода остаются слабыми и смешанными: они поддерживают идею защиты ranking-задачи от неравномерных auxiliary gradients, но не показывают явно вредной текущей вспомогательной задачи по NDCG@10.

## Гигиена TEST

Проверены committed run artifacts Stage 1, 41 artifact `result.json` / `result.partial.json` из Stage 2 и Stage 3 artifacts. Во всех случаях `test_evaluation_count = 0`.
