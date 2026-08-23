# Multitask KuaiRand Protocol B audit

## Цель

Подготовить первый data/audit слой для будущей TiM4Rec-based multitask архитектуры без обучения новой модели. Проверка выполняется именно на 1 134 420 interactions Protocol B, а не на full KuaiRand-27K EDA.

## Источник данных

- Protocol B parquet: `/home/daryumin/iberdov/diplom/data/processed/protocol_b`.
- Raw source log: `/home/daryumin/iberdov/Corpora/KuaiRand-Pure/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv`.
- Используется ранний KuaiRand-Pure standard log; random interactions в Protocol B не входят.
- Семантика полей взята из README KuaiRand, локальная копия: `data/KuaiRand-1K/README.md`.

## Связь с существующим Protocol B

- Split не пересоздавался: train/validation/test прочитаны из существующих Protocol B parquet.
- Каждая строка Protocol B сохраняет `source_row_id`, нулевой номер строки в raw CSV.
- Join выполнен по `source_row_id`; `(user_id, item_id, timestamp)` использовался только как проверка соответствия.
- Exact row identity hash `user_id,item_id,timestamp,split`: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Доступные behavior labels

| field | dtype | missing | unique values | available in source | kind | possible target | leakage risk |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `user_id` | `Int64` | 0 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22] ... (+23931)` | True | identifier | no | identity/linkage field; not a behavior target |
| `video_id` | `Int64` | 0 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21] ... (+7091)` | True | identifier | no | identity/linkage field; not a behavior target |
| `date` | `Int64` | 0 | `[20220409, 20220410, 20220411, 20220412, 20220413, 20220414, 20220415, 20220416, 20220417, 20220418, 20220419, 20220420, 20220421]` | True | categorical | no | context field; future use needs time-aware policy review |
| `hourmin` | `Int64` | 0 | `[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900] ... (+4)` | True | categorical | no | context field; future use needs time-aware policy review |
| `time_ms` | `Int64` | 0 | `[1649475963278, 1649476012164, 1649476421916, 1649476523839, 1649476630340, 1649476742423, 1649476787404, 1649476800158, 1649476886158, 1649476886308, 1649476925117, 1649476951047, 1649476971601, 1649476978588, 1649477146477, 1649477155751, 1649477278239, 1649477446035, 1649477488922, 1649477559363] ... (+1046066)` | True | identifier | no | identity/linkage field; not a behavior target |
| `is_click` | `Int64` | 0 | `[0, 1]` | True | binary | yes | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `is_like` | `Int64` | 0 | `[0, 1]` | True | binary | yes | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `is_follow` | `Int64` | 0 | `[0, 1]` | True | binary | yes | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `is_comment` | `Int64` | 0 | `[0, 1]` | True | binary | yes | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `is_forward` | `Int64` | 0 | `[0, 1]` | True | binary | yes | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `is_hate` | `Int64` | 0 | `[0, 1]` | True | binary | yes | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `long_view` | `Int64` | 0 | `[0, 1]` | True | binary | yes | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `play_time_ms` | `Int64` | 0 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] ... (+141979)` | True | continuous | possible_regression_or_auxiliary | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `duration_ms` | `Int64` | 0 | `[0, 5000, 5047, 5200, 6000, 6041, 6133, 6233, 6266, 6300, 6366, 6400, 6466, 6483, 6520, 6533, 6573, 6600, 6633, 6666] ... (+5442)` | True | continuous | source_for_derived_watch_targets | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `profile_stay_time` | `Int64` | 0 | `[0, 39, 65, 118, 213, 369, 611, 756, 867, 963, 1003, 1051, 1161, 1179, 1205, 1252, 1357, 1359, 1370, 1403] ... (+100)` | True | continuous | possible_regression_or_auxiliary | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `comment_stay_time` | `Int64` | 0 | `[0, 2, 3, 4, 8, 10, 11, 12, 13, 14, 15, 16, 22, 24, 26, 29, 49, 70, 130, 131] ... (+21903)` | True | continuous | possible_regression_or_auxiliary | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `is_profile_enter` | `Int64` | 0 | `[0, 1]` | True | binary | yes | post-exposure label/current interaction: target-only, forbidden as same-row input |
| `is_rand` | `Int64` | 0 | `[0]` | True | binary | no | constant standard-policy flag in Protocol B; not useful as target |
| `tab` | `Int64` | 0 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]` | True | categorical | no | context field; future use needs time-aware policy review |
| `play_ratio` | `Float64` | 23 810 | `[0.0, 2.866019328434351e-06, 3.3780017768289346e-06, 4.139758238118894e-06, 4.284490145672665e-06, 5e-06, 5.417617006983308e-06, 5.584002948353557e-06, 5.90261279155218e-06, 5.926979611190138e-06, 6.4047959111782904e-06, 7.720040453011974e-06, 9.095043201455207e-06, 9.172208209126346e-06, 9.230797633223486e-06, 9.578544061302682e-06, 1.0635469290082425e-05, 1.1544722330186755e-05, 1.2382674162312094e-05, 1.4411990776325903e-05] ... (+897938)` | False | continuous | possible_regression_or_auxiliary | post-exposure label/current interaction: target-only, forbidden as same-row input |

## Семантика labels

- `is_click` является post-exposure бинарным feedback. В single-column UI это derived `valid_play`, поэтому он не является чистым click во всех сценариях.
- `long_view` является derived post-exposure target из watch-time и duration: threshold 18 секунд или полная длительность для коротких видео.
- `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_profile_enter` являются явными positive engagement actions после показа.
- `is_hate` является negative feedback и не смешивается с positive engagement.
- `play_ratio` в audit вычислен как `play_time_ms / duration_ms` только при `duration_ms > 0`; в raw CSV отдельной колонки `play_ratio` нет.

## Статистика train/validation/test

| target | train positive rate | validation positive rate | test positive rate | train positives | train positive users | train positive items | missing train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `is_click` | 46.225% | 48.261% | 48.603% | 502 248 | 98.497% | 96.723% | 0 |
| `long_view` | 33.537% | 35.744% | 35.781% | 364 383 | 96.881% | 91.731% | 0 |
| `is_like` | 1.862% | 1.958% | 1.958% | 20 232 | 22.976% | 48.010% | 0 |
| `is_follow` | 0.096% | 0.117% | 0.188% | 1 046 | 3.273% | 9.886% | 0 |
| `is_comment` | 0.259% | 0.209% | 0.209% | 2 818 | 7.657% | 16.130% | 0 |
| `is_forward` | 0.098% | 0.096% | 0.121% | 1 065 | 3.315% | 10.533% | 0 |
| `is_hate` | 0.042% | 0.046% | 0.046% | 458 | 1.027% | 5.133% | 0 |
| `is_profile_enter` | 2.552% | 2.367% | 2.179% | 27 730 | 41.915% | 51.146% | 0 |
| `strong_positive` | 2.217% | 2.296% | 2.367% | 24 092 | 29.339% | 52.552% | 0 |
| `explicit_positive` | 4.552% | 4.451% | 4.300% | 49 458 | 53.217% | 66.221% | 0 |
| `deep_engagement` | 34.754% | 37.046% | 36.942% | 377 604 | 97.094% | 93.067% | 0 |

## Class imbalance

- `is_click` и `long_view` достаточно частые и стабильные по split.
- `is_like` и `is_profile_enter` редкие, но имеют заметную user/item coverage и отражают разные действия.
- `is_follow`, `is_comment`, `is_forward`, `is_hate` сильно разрежены; отдельные heads для них в первом эксперименте рискованны.
- `explicit_positive` и `strong_positive` могут снизить sparse-проблему, но это derived targets, поэтому их надо явно фиксировать в protocol/config.

## User coverage

| target | positive rows train | positive users train | share users with positive | median positives/user | p95 positives/user |
| --- | ---: | ---: | ---: | ---: | ---: |
| `is_click` | 502 248 | 23 591 | 98.497% | 15.0 | 61.0 |
| `long_view` | 364 383 | 23 204 | 96.881% | 11.0 | 45.0 |
| `is_like` | 20 232 | 5 503 | 22.976% | 2.0 | 14.0 |
| `is_follow` | 1 046 | 784 | 3.273% | 1.0 | 3.0 |
| `is_comment` | 2 818 | 1 834 | 7.657% | 1.0 | 4.0 |
| `is_forward` | 1 065 | 794 | 3.315% | 1.0 | 2.0 |
| `is_hate` | 458 | 246 | 1.027% | 1.0 | 5.0 |
| `is_profile_enter` | 27 730 | 10 039 | 41.915% | 2.0 | 8.0 |
| `strong_positive` | 24 092 | 7 027 | 29.339% | 2.0 | 13.0 |
| `explicit_positive` | 49 458 | 12 746 | 53.217% | 2.0 | 12.0 |
| `deep_engagement` | 377 604 | 23 255 | 97.094% | 12.0 | 47.0 |

## Item coverage

Покрытие item positives считается только по TRAIN, так как будущая модель должна учиться без validation/test labels.

| target | positive rows train | positive items train | share items with positive | median positives/item | p95 positives/item |
| --- | ---: | ---: | ---: | ---: | ---: |
| `is_click` | 502 248 | 6 878 | 96.723% | 17.0 | 313.1 |
| `long_view` | 364 383 | 6 523 | 91.731% | 12.0 | 240.0 |
| `is_like` | 20 232 | 3 414 | 48.010% | 2.0 | 22.0 |
| `is_follow` | 1 046 | 703 | 9.886% | 1.0 | 4.0 |
| `is_comment` | 2 818 | 1 147 | 16.130% | 1.0 | 7.0 |
| `is_forward` | 1 065 | 749 | 10.533% | 1.0 | 3.0 |
| `is_hate` | 458 | 365 | 5.133% | 1.0 | 2.0 |
| `is_profile_enter` | 27 730 | 3 637 | 51.146% | 3.0 | 30.0 |
| `strong_positive` | 24 092 | 3 737 | 52.552% | 2.0 | 24.0 |
| `explicit_positive` | 49 458 | 4 709 | 66.221% | 3.0 | 42.0 |
| `deep_engagement` | 377 604 | 6 618 | 93.067% | 13.0 | 245.1 |

## Temporal distribution

| target | train | validation | test | test-train | max-min |
| --- | ---: | ---: | ---: | ---: | ---: |
| `is_click` | 46.225% | 48.261% | 48.603% | 2.378% | 2.378% |
| `long_view` | 33.537% | 35.744% | 35.781% | 2.245% | 2.245% |
| `is_like` | 1.862% | 1.958% | 1.958% | 0.096% | 0.096% |
| `is_follow` | 0.096% | 0.117% | 0.188% | 0.092% | 0.092% |
| `is_comment` | 0.259% | 0.209% | 0.209% | -0.051% | 0.051% |
| `is_forward` | 0.098% | 0.096% | 0.121% | 0.023% | 0.025% |
| `is_hate` | 0.042% | 0.046% | 0.046% | 0.004% | 0.004% |
| `is_profile_enter` | 2.552% | 2.367% | 2.179% | -0.373% | 0.373% |
| `strong_positive` | 2.217% | 2.296% | 2.367% | 0.150% | 0.150% |
| `explicit_positive` | 4.552% | 4.451% | 4.300% | -0.252% | 0.252% |
| `deep_engagement` | 34.754% | 37.046% | 36.942% | 2.189% | 2.293% |

- `is_click` и `long_view` выше в validation/test примерно на 2.2-2.4 п.п. относительно train.
- `is_profile_enter` немного снижается к test; sparse labels дают более шумные split-level rates.

## Co-occurrence

| condition | probability | count both | jaccard |
| --- | ---: | ---: | ---: |
| P(`long_view`=1 / `is_click`=1) | 72.243% | 362 837 | 0.7202 |
| P(`is_like`=1 / `is_click`=1) | 3.434% | 17 245 | 0.0341 |
| P(`is_profile_enter`=1 / `is_click`=1) | 5.220% | 26 218 | 0.0520 |
| P(`is_hate`=1 / `is_click`=1) | 0.041% | 206 | 0.0004 |
| P(`is_follow`=1 / `is_like`=1) | 2.195% | 444 | 0.0213 |
| P(`is_comment`=1 / `is_click`=1) | 0.545% | 2 738 | 0.0055 |
| P(`is_forward`=1 / `is_click`=1) | 0.187% | 937 | 0.0019 |
| P(`explicit_positive`=1 / `is_like`=1) | 100.000% | 20 232 | 0.4091 |

- В train P(`long_view`=1 / `is_click`=1) = 72.243%. Это показывает сильную, но не полную связь consumption и click/valid_play.
- В train P(`is_like`=1 / `is_click`=1) = 3.434%; like несет более редкий explicit-positive сигнал.
- В train P(`is_hate`=1 / `is_click`=1) = 0.041%; negative feedback не надо объединять с positive engagement.

## Correlations

- Spearman(`play_ratio`, `long_view`) в train = 0.7629; это ожидаемо, потому что `long_view` derived из watch-time/duration.
- Spearman(`play_ratio`, `is_click`) в train = 0.7418; click/valid_play также связан с watch-time, но не тождественен long_view.
- Correlation не интерпретируется как причинность. Полная матрица сохранена в `target_correlations.csv`.

## Rare behaviors

- `is_follow`, `is_comment`, `is_forward` лучше не брать отдельными heads в `multitask_tim4rec_001`: они слишком sparse для первого устойчивого запуска.
- Их можно включить через прозрачный aggregate `strong_positive` или `explicit_positive`, либо оставить для второго этапа после базовой multitask проверки.
- `is_hate` тоже sparse, но семантически отдельный negative-preference сигнал; его стоит держать как option, а не смешивать с positive labels.

## Negative feedback

- `is_hate` в train: 0.042%, positives 458, positive users 1.027%.
- Для первого stable multitask run это слишком sparse как обязательная head, но поле подготовлено и валидно как будущая negative-preference task.
- `is_hate` нельзя объединять с `strong_positive` или `explicit_positive`: это отдельная семантика.

## Watch-time variables

- В full filtered `duration_ms <= 0`: 23 810 строк (2.099%).
- В train `duration_ms <= 0`: 22 741 строк (2.093%).
- В train `play_ratio > 1`: 164 268 строк среди valid ratio (15.442%).
- `play_ratio > 1` не помечается как ошибка: возможны пересмотры/повторы. Для regression head разумнее анализировать `log1p(play_time_ms)` или clipped `play_ratio`, но не в первом data-only шаге.

## Возможные derived targets

- `strong_positive = is_like OR is_follow OR is_comment OR is_forward`.
- `explicit_positive = is_like OR is_follow OR is_comment OR is_forward OR is_profile_enter`.
- `deep_engagement = long_view OR is_like OR is_follow OR is_comment OR is_forward OR is_profile_enter`.
- Derived targets не материализованы как обязательные labels будущей модели; они посчитаны для выбора первого experiment.

## Candidate task sets

### OPTION A - minimal

- `is_click` (raw): train rate 46.225%, users+ 98.497%, items+ 96.723%.
- `long_view` (raw): train rate 33.537%, users+ 96.881%, items+ 91.731%.
- `is_like` (raw): train rate 1.862%, users+ 22.976%, items+ 48.010%.

### OPTION B - balanced derived

- `is_click` (raw): train rate 46.225%, users+ 98.497%, items+ 96.723%.
- `long_view` (raw): train rate 33.537%, users+ 96.881%, items+ 91.731%.
- `explicit_positive` (derived; formula: `is_like OR is_follow OR is_comment OR is_forward OR is_profile_enter`): train rate 4.552%, users+ 53.217%, items+ 66.221%.
- `is_hate` (raw): train rate 0.042%, users+ 1.027%, items+ 5.133%.

### OPTION C - richer raw

- `is_click` (raw): train rate 46.225%, users+ 98.497%, items+ 96.723%.
- `long_view` (raw): train rate 33.537%, users+ 96.881%, items+ 91.731%.
- `is_like` (raw): train rate 1.862%, users+ 22.976%, items+ 48.010%.
- `is_profile_enter` (raw): train rate 2.552%, users+ 41.915%, items+ 51.146%.
- `is_hate` (raw): train rate 0.042%, users+ 1.027%, items+ 5.133%.


## Рекомендуемый первый multitask набор

Для `multitask_tim4rec_001` рекомендуется начать с raw targets:

- `is_click`
- `long_view`
- `is_like`
- `is_profile_enter`

Причина: это 4 задачи без спорной preprocessing-логики, с разными behavioral meanings и приемлемой плотностью. `is_hate` лучше оставить как заранее подготовленный negative option для следующего запуска или ablation, потому что он слишком sparse для первого stability check.

## Предварительные behavior experts

- Interest/exposure response: `is_click`.
- Consumption/watch-time: `long_view`, позднее `log1p(play_time_ms)` или clipped `play_ratio`.
- Positive engagement: `is_like`, `is_profile_enter`, позднее aggregate `explicit_positive`.
- Social/amplification: `is_follow`, `is_comment`, `is_forward` как отдельная группа после sparse-aware настройки.
- Negative preference: `is_hate` отдельно от positive heads.

## Leakage policy

- Labels текущего candidate interaction не используются как input features для предсказания этого же interaction.
- Historical behavior предыдущих interactions можно использовать позже только после time-aware sequence construction.
- Full-period item statistics и target-derived aggregates запрещены как inputs без перестроения по train-window.
- `duration_ms` может быть item/context field, но любые derived watch targets из текущего interaction являются target-only.

## Dataset preparation

- Multitask dataset path: `/home/daryumin/iberdov/diplom/data/processed/protocol_b_multitask`.
- Сохранены `train.parquet`, `validation.parquet`, `test.parquet`, `full_filtered.parquet`.
- В строках сохранены `user_id`, `item_id`, `timestamp`, `source_row_id`, `split`, raw behavior labels и watch-time поля.

## Join validation

- rows expected: 1 134 420.
- rows matched: 1 134 420.
- unmatched: 0.
- multiple matched: 0.
- source_row duplicate extra rows внутри Protocol B: 0.
- Hidden many-to-many joins отсутствуют, потому что join key - stable raw row index.

## Ограничения

- Аудит фиксирует labels для KuaiRand-Pure Protocol B, а не для full KuaiRand-27K.
- `is_click` и `long_view` частично derived из watch behavior и UI-specific semantics, поэтому они не являются независимыми чистыми actions.
- Continuous watch-time targets требуют отдельной нормализации/clipping политики перед обучением.

## Следующий эксперимент

Следующий шаг - реализовать минимальный TiM4Rec multitask head поверх того же sequence backbone для targets `is_click`, `long_view`, `is_like`, `is_profile_enter`. Не менять split и не открывать новые evaluation protocol variants.
