# Stage 3 — анализ вспомогательных задач

Stage 3 выполнен на KuaiRand в рамках протокола B только по валидационной выборке. TEST не использовался.

## Область анализа

Stage 3 проверяет доступные поведенческие метки и вклад отдельных вспомогательных задач для уже установленной схемы `MultitaskTiM4Rec`. Основная задача остаётся прежней: ранжирование следующего объекта. Вспомогательные метрики приводятся только как диагностика и не используются для выбора рекомендательной модели.

Артефакты были получены с infrastructure commit `2177bc4e0d082ad6ccb8532f04b9e015fa80e9a0`. Validation-only датасет RecBole использует `benchmark_filename = ["train", "valid"]`; каждый artifact содержит `test_evaluation_count = 0`.

## Аудит целевых сигналов

| Сигнал | Поле | Тип | Train observations | Train positives / rate | Validation rate | Missing rate | Текущий head | Проверен отдельно |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| click | `is_click` | binary | 1 086 518 | 502 248 / 0.4623 | 0.4826 | 0.0000 | yes | yes |
| long_view | `long_view` | binary | 1 086 518 | 364 383 / 0.3354 | 0.3574 | 0.0000 | yes | yes |
| like | `is_like` | binary | 1 086 518 | 20 232 / 0.0186 | 0.0196 | 0.0000 | yes | yes |
| profile_enter | `is_profile_enter` | binary | 1 086 518 | 27 730 / 0.0255 | 0.0237 | 0.0000 | yes | yes |
| follow | `is_follow` | binary | 1 086 518 | 1 046 / 0.0010 | 0.0012 | 0.0000 | no | no |
| comment | `is_comment` | binary | 1 086 518 | 2 818 / 0.0026 | 0.0021 | 0.0000 | no | no |
| forward | `is_forward` | binary | 1 086 518 | 1 065 / 0.0010 | 0.0010 | 0.0000 | no | no |
| hate | `is_hate` | binary | 1 086 518 | 458 / 0.0004 | 0.0005 | 0.0000 | no | no |
| play_time | `play_time_ms` | continuous | 1 086 518 | n/a | n/a | 0.0000 | no | no |
| play_ratio | `play_ratio` | continuous | 1 086 518 | n/a | n/a | 0.0209 | no | no |

Распределения continuous-сигналов на train:

| Сигнал | Mean | Median | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| play_time | 23 109.0289 | 4 938.0000 | 69 410.3000 | 106 473.0000 | 211 966.8300 | 964 774.0000 |
| play_ratio | 0.3696 | 0.1021 | 1.0377 | 1.1561 | 2.0957 | 58.0329 |

## Построение и допустимость сигналов

| Сигнал | Как построен в репозитории | Доступность в момент interaction и решение по leakage |
| --- | --- | --- |
| `is_click` | Raw поле после показа сохраняется в `experiments/multitask_tim4rec/audit_targets.py`; в RecBole files для валидационной схемы экспортируется через `experiments/multitask_tim4rec_optuna/prepare_validation_only.py`. | Это целевая метка той же строки; `MultitaskTiM4Rec.input_fields_used` исключает её из input features. Утечка будущего target для Stage 3 не найдена. |
| `long_view` | Raw/derived watch label после показа сохраняется в audit-коде и экспортируется как float target. | Это целевая метка той же строки, а не input feature. Сигнал сильно связан с watch time, поэтому он допустим как auxiliary target, но не является независимым доказательством пользы для ранжирования. |
| `is_like` | Raw explicit positive action сохраняется в audit-коде и моделируется существующим `like_head`. | Это целевая метка той же строки, а не input feature. Есть сильный class imbalance, но утечка будущего target не найдена. |
| `is_profile_enter` | Raw profile-entry action сохраняется в audit-коде и моделируется существующим `profile_enter_head`. | Это целевая метка той же строки, а не input feature. Есть class imbalance; Stage 3 оставляет сигнал, потому что head уже существует. |
| `is_follow` | Raw explicit action присутствует в Protocol B multitask audit data. | В принципе допустим как label, но исключён из первого ablation pass: prevalence всего 0.0010, а добавление требует изменения области модели. |
| `is_comment` | Raw explicit action присутствует в Protocol B multitask audit data. | В принципе допустим как label, но исключён из первого ablation pass: prevalence 0.0026, добавление требует изменения области модели. |
| `is_forward` | Raw explicit action присутствует в Protocol B multitask audit data. | В принципе допустим как label, но исключён из первого ablation pass: prevalence 0.0010, добавление требует изменения области модели. |
| `is_hate` | Raw negative feedback присутствует в Protocol B multitask audit data. | Не смешивается с positive engagement labels; исключён из-за редкости и отрицательной семантики. Для него нужен отдельный дизайн целевой функции. |
| `play_time_ms` | Raw watch-time field сохраняется в Protocol B multitask audit data. | Continuous signal после показа; исключён, потому что корректное использование требует regression, survival, clipping или отдельного дизайна преобразования вне первого pass с одной вспомогательной задачей. |
| `play_ratio` | Рассчитывается как `play_time_ms / duration_ms`, если `duration_ms > 0`. | Continuous signal после показа с missing values при неположительной duration; исключён по той же причине, что и watch time. |

Первый ablation pass использует ровно четыре существующих heads: `is_click`, `long_view`, `is_like`, `is_profile_enter`.

## Связь и избыточность сигналов

Для binary-сигналов использованы co-occurrence на train split, conditional probabilities, Jaccard overlap и phi coefficient. Эти статистики описывают связи между labels; связь target-target сама по себе не доказывает, что auxiliary target улучшает ранжирование следующего объекта.

| Pair | Phi | Jaccard | P(right=1 \| left=1) | P(left=1 \| right=1) | Интерпретация |
| --- | ---: | ---: | ---: | ---: | --- |
| `is_click` / `long_view` | 0.7601 | 0.7202 | 0.7224 | 0.9958 | Сильно связаны; `long_view` почти вложен в click/valid-play, но не идентичен ему. |
| `is_click` / `is_like` | 0.1078 | 0.0341 | 0.0343 | 0.8524 | Likes редкие и обычно происходят вместе с click, но click редко означает like. |
| `is_click` / `is_profile_enter` | 0.1569 | 0.0520 | 0.0522 | 0.9455 | Profile enter редок и обычно происходит вместе с click. |
| `long_view` / `is_like` | 0.0997 | 0.0369 | 0.0376 | 0.6770 | Likes часто связаны с long views, но большинство long views не являются likes. |
| `long_view` / `is_profile_enter` | 0.1473 | 0.0572 | 0.0582 | 0.7649 | Profile enter пересекается с `long_view`, но остаётся отдельным поведением. |
| `is_like` / `is_profile_enter` | 0.0597 | 0.0412 | 0.0939 | 0.0685 | Явные positive actions пересекаются слабо. |

Связи continuous/watch-time сигналов носят описательный характер. `play_time_ms` и `play_ratio` сильно связаны с `long_view` и `is_click`, что ожидаемо: `long_view` сам является watch-time-derived behavior. Например, Pearson(`play_ratio`, `long_view`) равен `0.6480`, Pearson(`play_ratio`, `is_click`) равен `0.5544` на non-missing train rows.

## Запуски с одной вспомогательной задачей

| Run | Вспомогательная задача | Best epoch | Actual epochs | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | Delta NDCG@10 | TEST evals |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stage3_primary_only_001` | `none` | 11 | 16 | 0.1080 | 0.1755 | 0.3124 | 0.0586 | 0.0755 | 0.1025 | 0.0000 | 0 |
| `stage3_aux_click_001` | `is_click` | 17 | 22 | 0.1086 | 0.1779 | 0.3183 | 0.0593 | 0.0767 | 0.1043 | +0.0007 | 0 |
| `stage3_aux_long_view_001` | `long_view` | 9 | 14 | 0.1083 | 0.1736 | 0.3117 | 0.0586 | 0.0750 | 0.1022 | +0.0000 | 0 |
| `stage3_aux_like_001` | `is_like` | 11 | 16 | 0.1080 | 0.1777 | 0.3150 | 0.0587 | 0.0762 | 0.1033 | +0.0001 | 0 |
| `stage3_aux_profile_enter_001` | `is_profile_enter` | 11 | 16 | 0.1088 | 0.1737 | 0.3141 | 0.0590 | 0.0752 | 0.1030 | +0.0004 | 0 |

Диагностический запуск со всеми четырьмя текущими auxiliary heads достиг validation NDCG@10 `0.0597` на epoch 17. Здесь он используется для матрицы градиентов между вспомогательными задачами, а не как новый результат TEST.

## Диагностика градиентных взаимодействий

Градиенты измерялись на одном и том же наборе общих параметров TiM4Rec backbone, без task-specific heads. Runner восстанавливает RNG state перед training step после диагностики, поэтому diagnostic pass не должен намеренно менять ход оптимизации.

| Вспомогательная задача | Batches | Median norm ratio | Mean cosine | Median cosine | Q25 cosine | Q75 cosine | Conflict rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `is_click` | 5 | 0.1413 | 0.0452 | 0.0428 | 0.0275 | 0.0565 | 0.2000 |
| `long_view` | 3 | 0.1564 | 0.0223 | 0.0219 | 0.0143 | 0.0301 | 0.0000 |
| `is_like` | 4 | 0.4343 | 0.0265 | 0.0308 | -0.0075 | 0.0649 | 0.5000 |
| `is_profile_enter` | 4 | 0.3254 | -0.0082 | 0.0007 | -0.0125 | 0.0049 | 0.2500 |

## Совместная интерпретация вспомогательных сигналов

| Вспомогательная задача | Доля в train | Delta NDCG@10 | Mean cosine with primary | Conflict rate | Median norm ratio | Auxiliary validation metric | Интерпретация |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `is_click` | 0.4623 | +0.0007 | 0.0452 | 0.2000 | 0.1413 | BCE 0.6382, accuracy 0.6376 | Лучший результат с одной вспомогательной задачей в этом pass: небольшой положительный ranking delta и умеренно согласованные gradients. |
| `long_view` | 0.3354 | +0.0000 | 0.0223 | 0.0000 | 0.1564 | BCE 0.6070, accuracy 0.6679 | Нет primary conflict, но почти нет прироста ranking; это согласуется с избыточностью относительно click/watch-time behavior. |
| `is_like` | 0.0186 | +0.0001 | 0.0265 | 0.5000 | 0.4343 | BCE 0.1545, accuracy 0.9708 | Редкий label с большими относительными gradients и частым conflict; ranking gain очень мал и должен считаться нестабильным. |
| `is_profile_enter` | 0.0255 | +0.0004 | -0.0082 | 0.2500 | 0.3254 | BCE 0.1814, accuracy 0.9760 | Небольшой ranking gain при near-zero/negative mean cosine; positive predictions очень разрежены, поэтому evidence mixed. |

В текущем single-seed pass ни одна текущая auxiliary target не оказалась явно вредной по validation NDCG@10. Вторичные метрики ранжирования для `long_view` и `is_profile_enter` смешанные, поэтому результат нельзя читать как широкое доминирование.

## Матрица градиентов между вспомогательными задачами

Матрица ниже взята из `stage3_all_current_aux_diagnostic_001`.

| Left | Right | Batches | Mean cosine | Median cosine | Conflict rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `is_click` | `is_like` | 5 | 0.0795 | 0.0723 | 0.2000 |
| `is_click` | `is_profile_enter` | 5 | 0.0139 | 0.0212 | 0.4000 |
| `is_click` | `long_view` | 5 | 0.5976 | 0.7392 | 0.2000 |
| `is_like` | `is_profile_enter` | 5 | 0.0270 | 0.0517 | 0.2000 |
| `long_view` | `is_like` | 5 | 0.0288 | -0.0058 | 0.6000 |
| `long_view` | `is_profile_enter` | 5 | -0.0074 | -0.0006 | 0.6000 |

Самая сильная согласованность между вспомогательными задачами наблюдается у `is_click` и `long_view`; это совпадает с избыточностью на уровне данных. `long_view` чаще конфликтует с редкими explicit-action heads, поэтому симметричный multi-objective method может тратить capacity на разрешение disagreement между вспомогательными задачами, который не обязательно улучшает primary ranking.

## Связь со Stage 1/2 MOO

Stage 3 согласуется со следующей интерпретацией Stage 1/2, но не доказывает её причинно:

- EPO мог оказаться сильным, потому что finite preference search способен включать operating points, защищающие primary ranking objective и одновременно использующие полезный auxiliary signal. Его Stage 2 NDCG@10 `0.0588` близок к Stage 3 primary-only `0.0586`, но ниже all-current diagnostic `0.0597`.
- PCGrad улучшился после настройки, что согласуется с mild-to-moderate gradient conflicts. При этом конфликты Stage 3 не являются catastrophic для каждой auxiliary task, поэтому projection alone не гарантирует достаточный результат; важны weights и operating point.
- GradHV и COSMOS остались слабее в Stage 1/2. Stage 3 совместим с идеей, что rare heads и auxiliary-auxiliary conflicts могут отвлекать методы, слишком симметрично относящиеся к objectives, но это не proof of mechanism.
- Preference-collapse failures COSMOS из Stage 2 совместимы с этой картиной: conditioning on preferences полезен только если модель действительно создаёт distinct primary-relevant trade-offs.

## Гипотеза о приоритете основной задачи

Гипотеза: в этой рекомендательной задаче next-item ranking является основной задачей, а поведенческие tasks — вспомогательными. Поэтому будущий метод должен защищать primary gradient и использовать auxiliary gradients только тогда, когда они не вредят основной ranking objective.

Аргументы в пользу:

- лучший single auxiliary (`is_click`) имеет максимальный положительный NDCG@10 delta и mildly positive primary cosine;
- `is_like` и `is_profile_enter` имеют большие relative gradient norms и больше conflicts, а их ranking gains малы;
- all-current diagnostic улучшает NDCG@10, значит auxiliary information полезна, но паттерн запусков с одной вспомогательной задачей показывает, что полезность неравномерна.

Аргументы против и ограничения:

- в этом single-seed Stage 3 pass нет current auxiliary target с отрицательным NDCG@10 delta;
- correlation между median primary cosine и NDCG@10 delta равна только `0.2626`, поэтому gradient alignment alone не объясняет ranking impact;
- sample diagnostic batches намеренно компактный и не должен трактоваться как доказательство full training trajectory.

Точный failure mode для primary-aware метода: большие или конфликтующие auxiliary updates от sparse behavior heads меняют shared ranking representations, когда это не улучшает validation NDCG@10.

Существующие методы частично покрывают эту идею. PCGrad может удалять conflicting components, EPO ищет Pareto trade-offs, GradHV использует hypervolume objective, COSMOS и PHN condition on preferences. Дипломный вклад должен быть конкретнее, чем "use gradients": нужна воспроизводимая primary-priority rule, ясное отличие от PCGrad/EPO scalarization, validation-only protocol выбора и позднее locked TEST evaluation после выбора метода.

## Дополнительные целевые сигналы

`is_follow`, `is_comment`, `is_forward` и `is_hate` доступны, но слишком разрежены для первого single-head pass без дополнительного objective design. `is_hate` должен оставаться отдельным negative-feedback target, а не positive engagement label. `play_time_ms` и `play_ratio` выглядят перспективными consumption signals, но их корректное использование требует continuous-objective design, например `log1p(play_time_ms)` или clipped `play_ratio`; это отдельный эксперимент.

Рекомендуемый следующий научный эксперимент: validation-only primary-aware gradient gating на существующих четырёх heads, сначала в сравнении с `primary_only`, `is_click` и all-current fixed-weight diagnostic при том же seed и бюджете. Multiseed и TEST должны ждать до фиксации метода и протокола выбора.

## Slurm-происхождение запусков

| Роль | Job ID | Partition | Status | Notes |
| --- | ---: | --- | --- | --- |
| Target audit | 4299312 | `cpu-e-quick` | COMPLETED | CPU-only audit, `TEST=0`. |
| Sanity run | 4299311 | `test` | COMPLETED | One epoch, two train batches, finite diagnostics, `TEST=0`. |
| Primary-only ablation | 4299376 | `gpu-ef-quick` | COMPLETED | Produced `stage3_primary_only_001.json`. |
| Click ablation | 4299449 | `test` | TIMEOUT | Final JSON was written as `COMPLETE`; partial file removed; `TEST=0`. |
| Long-view ablation | 4299450 | `test` | COMPLETED | Produced `stage3_aux_long_view_001.json`. |
| Like ablation | 4299451 | `test` | COMPLETED | Produced `stage3_aux_like_001.json`. |
| Profile-enter ablation | 4299452 | `test` | COMPLETED | Produced `stage3_aux_profile_enter_001.json`. |
| All-current diagnostic | 4299453 | `test` | COMPLETED | Produced auxiliary-auxiliary gradient matrix. |

## Гигиена TEST

- `test_evaluation_count = 0` для каждого Stage 3 artifact.
- Stage 3 artifacts содержат `test_dataset_loaded = false` и `test_metrics_present = false`.
- TEST metrics не используются для выбора target, интерпретации ablations или проверки гипотезы о приоритете основной задачи.
