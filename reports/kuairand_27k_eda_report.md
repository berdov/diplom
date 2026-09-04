# Исследовательский анализ KuaiRand-27K

Отчёт построен по фактическим результатам полного EDA KuaiRand-27K, выполненного на cHARISMa. Компактный fingerprint сохранён в `outputs/eda/27k_summary.json`; подробные aggregate CSV можно повторно сгенерировать через `src/eda_27k.py`. Raw interaction logs и per-user CSV не коммитятся.

## 1. Обзор датасета

- Полный EDA-запуск: job `4253874`, состояние `COMPLETED`, ExitCode `0:0`.
- Slurm: partition `test`, constraint `type_a`, GRES `gpu:v100:1`, CPUs `10`, memory `mem=0 (в Slurm для этих узлов RealMemory отображается как 1M; фактическая свободная RAM на type_a порядка 700+ GB)`.
- Узел: `cn-012`; runtime `00:09:45`; MaxRSS `46537000K`.
- Сгенерировано UTC: `2026-08-16T16:01:15.653041+00:00`.
- Размер директории KuaiRand-27K по inventory: `45.6 GiB`.
- Перед финальным запуском была проверена `normal/type_c/gpu:v100:1`, но job 4253860 упал за 3 секунды из-за несовместимости GLIBC старой ОС normal с текущим `.conda`. Финальный успешный запуск выполнен на Rocky-compatible partition `test`.

## 2. Структура файлов

Логи взаимодействий прочитаны как пять исходных файлов: один random log и четыре standard parts за два календарных периода.

| policy | period | source_log | interactions | users | items | date_min | date_max | unique_tabs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 2022-04-22_to_2022-05-08 | log_random_4_22_to_5_08_27k | 1,186,059 | 27,285 | 7,583 | 20220422 | 20220508 | 4 |
| standard | 2022-04-08_to_2022-04-21 | log_standard_4_08_to_4_21_27k_part1 | 68,148,288 | 26,858 | 10,221,515 | 20220408 | 20220421 | 15 |
| standard | 2022-04-08_to_2022-04-21 | log_standard_4_08_to_4_21_27k_part2 | 68,148,288 | 26,857 | 10,414,229 | 20220408 | 20220421 | 15 |
| standard | 2022-04-22_to_2022-05-08 | log_standard_4_22_to_5_08_27k_part1 | 93,062,020 | 27,285 | 13,214,491 | 20220422 | 20220508 | 15 |
| standard | 2022-04-22_to_2022-05-08 | log_standard_4_22_to_5_08_27k_part2 | 92,919,789 | 27,285 | 13,347,412 | 20220422 | 20220508 | 15 |

Файлы признаков имеют полные ID-таблицы без пропущенных и повторяющихся ID:

| feature_table | duplicate_id_rows | files | missing_ids | rows | unique_ids |
| --- | --- | --- | --- | --- | --- |
| user_features | 0 | 1 | 0 | 27,285 | 27,285 |
| video_basic | 0 | 1 | 0 | 32,038,725 | 32,038,725 |
| video_statistic | 0 | 3 | 0 | 32,038,725 | 32,038,725 |

## 3. Количество взаимодействий

| policy | interactions | users | items | date_min | date_max | time_ms_min | time_ms_max | unique_tabs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 1,186,059 | 27,285 | 7,583 | 20220422 | 20220508 | 1650595332831 | 1652025128763 | 4 |
| standard | 322,278,385 | 27,285 | 32,038,693 | 20220408 | 20220508 | 1649339966592 | 1652025130589 | 15 |

Standard содержит 322,278,385 взаимодействий, random содержит 1,186,059 взаимодействий. Доля random-взаимодействий равна 0.367%.

## 4. Пользователи и видео

| metric | count |
| --- | --- |
| README standard items | 32,038,725 |
| Unique standard interaction video_id | 32,038,693 |
| Unique random interaction video_id | 7,583 |
| Unique video_features_basic video_id | 32,038,725 |
| Items in features not in standard logs | 32 |
| Items in standard logs not in basic features | 0 |
| Unique video_features_statistic video_id | 32,038,725 |
| Items in statistic features not in standard logs | 32 |
| Items in standard logs not in statistic features | 0 |

Расхождение 32 items с README объясняется разницей между item universe в feature-файлах и реально наблюдаемыми `video_id` в standard-логах взаимодействий. `video_features_basic_27k.csv` и `video_features_statistic_27k_part*.csv` содержат ровно 32,038,725 уникальных `video_id`, а в standard-логах встречается 32,038,693. Все наблюдаемые standard items покрыты feature universe; 32 item IDs есть в признаках, но не встретились в interaction logs. Поэтому это не доказанная ошибка датасета, а различие определения: документированная item universe против наблюдаемых interaction items.

## 5. Сравнение standard и random policy

| metric | standard | random | absolute_difference_standard_minus_random | relative_ratio_standard_over_random |
| --- | --- | --- | --- | --- |
| interactions | 322,278,385.0000 | 1,186,059.0000 | 321,092,326.0000 | 271.7221 |
| users | 27,285.0000 | 27,285.0000 | 0.0000 | 1.0000 |
| items | 32,038,693.0000 | 7,583.0000 | 32,031,110.0000 | 4,225.0683 |
| is_click_rate | 0.3787 | 0.1762 | 0.2026 | 2.1499 |
| is_like_rate | 0.0167 | 0.0048 | 0.0119 | 3.4705 |
| is_follow_rate | 0.0011 | 0.0003 | 0.0008 | 4.0759 |
| is_comment_rate | 0.0025 | 0.0003 | 0.0022 | 7.2582 |
| is_forward_rate | 0.0009 | 0.0003 | 0.0005 | 2.5722 |
| is_hate_rate | 0.0005 | 0.0011 | -0.0007 | 0.4256 |
| long_view_rate | 0.2604 | 0.0850 | 0.1755 | 3.0655 |
| is_profile_enter_rate | 0.0179 | 0.0056 | 0.0124 | 3.2232 |
| play_time_ms_median | 3,114.0000 | 2,091.0000 | 1,023.0000 | 1.4892 |
| play_time_ms_mean | 14,709.6647 | 6,935.5113 | 7,774.1534 | 2.1209 |
| duration_ms_median | 28,933.0000 | 76,833.0000 | -47,900.0000 | 0.3766 |
| duration_ms_mean | 72,456.8564 | 104,438.8331 | -31,981.9768 | 0.6938 |
| valid_play_ratio_median | 0.1144 | 0.0348 | 0.0796 | 3.2882 |
| valid_play_ratio_mean | 0.4171 | 0.1413 | 0.2758 | 2.9519 |
| interactions_per_user_median | 8,594.0000 | 22.0000 | 8,572.0000 | 390.6364 |
| interactions_per_user_p95 | 32,590.8000 | 152.0000 | 32,438.8000 | 214.4132 |
| interactions_per_user_p99 | 56,230.3200 | 309.0000 | 55,921.3200 | 181.9751 |
| item_popularity_interactions_median | 1.0000 | 162.0000 | -161.0000 | 0.0062 |
| item_popularity_interactions_p95 | 30.0000 | 184.0000 | -154.0000 | 0.1630 |
| item_popularity_interactions_p99 | 168.0000 | 194.0000 | -26.0000 | 0.8660 |
| item_popularity_interactions_max | 15,380.0000 | 214.0000 | 15,166.0000 | 71.8692 |

Это наблюдаемое различие распределений, а не причинная интерпретация. Standard-policy traffic радикально отличается от random exposure по candidate distribution: standard видит около 32.0M items, random только 7,583 items.

## 6. Сигналы обратной связи

| policy | signal | positive_count | positive_rate | missing_count |
| --- | --- | --- | --- | --- |
| random | is_click | 208,934 | 17.616% | 0 |
| random | is_comment | 411 | 0.035% | 0 |
| random | is_follow | 310 | 0.026% | 0 |
| random | is_forward | 402 | 0.034% | 0 |
| random | is_hate | 1,351 | 0.114% | 0 |
| random | is_like | 5,691 | 0.480% | 0 |
| random | is_profile_enter | 6,597 | 0.556% | 0 |
| random | long_view | 100,769 | 8.496% | 0 |
| standard | is_click | 122,052,542 | 37.872% | 0 |
| standard | is_comment | 810,580 | 0.252% | 0 |
| standard | is_follow | 343,328 | 0.107% | 0 |
| standard | is_forward | 280,970 | 0.087% | 0 |
| standard | is_hate | 156,233 | 0.048% | 0 |
| standard | is_like | 5,366,646 | 1.665% | 0 |
| standard | is_profile_enter | 5,777,660 | 1.793% | 0 |
| standard | long_view | 83,936,743 | 26.045% | 0 |

`positive_rate` считается как `positive_count / interactions`; `positive_rate_non_null` также сохранён в CSV. Количество пропусков равно 0 для всех feedback-сигналов, поэтому обе версии доли совпадают.

## 7. Время просмотра и длительность видео

| policy | duration_zero | duration_negative | duration_null | duration_non_positive_share | play_time_median | duration_median | play_ratio_median | play_ratio_p99 | play_ratio_p999 | play_ratio_max | share_gt_1 | share_gt_2 | share_gt_5 | share_gt_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 36,920 | 0 | 0 | 3.113% | 2,091 | 76,833 | 0.0348 | 1.2362 | 3.3328 | 103.13 | 3.318% | 0.348% | 0.044% | 0.012% |
| standard | 27,274,006 | 0 | 0 | 8.463% | 3,114 | 28,933 | 0.1144 | 3.0600 | 7.8394 | 348.95 | 16.289% | 2.822% | 0.303% | 0.056% |

`duration_ms <= 0` полностью состоит из нулей: negative/null count равен 0. В standard это 27,274,006 строк (8.463%), затрагивает 27,277 users и 6,761,960 videos. В random это 36,920 строк (3.113%), 15,461 users и 239 videos.

`play_ratio` считается только при `duration_ms > 0`. Значения `play_ratio > 1` не считаются автоматической ошибкой: они могут соответствовать пересмотрам, автоповторам или суммарному play time больше длительности видео. Для визуализаций можно применять clipping, но исходную статистику нужно сохранять.

`duration_ms <= 0` не объясняется отсутствием video metadata:

| policy | has_video_basic | interactions | duration_non_positive_count | unique_videos | duration_non_positive_share |
| --- | --- | --- | --- | --- | --- |
| random | True | 1186059 | 36920 | 7583 | 0.0311282996882954 |
| standard | True | 322278385 | 27274006 | 32038693 | 0.0846287162572196 |

## 8. Статистика последовательностей

| scope | quantity | count | min | mean | median | p75 | p90 | p95 | p99 | p999 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | interactions_per_user | 27,285 | 110 | 11,855.03 | 8,646 | 15,279 | 24,650 | 32,641 | 56,245 | 115,002 | 228,030 |
| all | unique_videos_per_user | 27,285 | 109 | 11,511.97 | 8,434 | 14,867 | 23,923 | 31,529 | 53,788 | 111,330 | 218,796 |
| random | interactions_per_user | 27,285 | 10 | 43.47 | 22 | 45 | 96 | 152 | 309 | 653 | 1,580 |
| random | unique_videos_per_user | 27,285 | 10 | 43.47 | 22 | 45 | 96 | 152 | 309 | 651 | 1,580 |
| standard | interactions_per_user | 27,285 | 100 | 11,811.56 | 8,594 | 15,227 | 24,587 | 32,591 | 56,230 | 114,979 | 228,000 |
| standard | unique_videos_per_user | 27,285 | 99 | 11,468.53 | 8,384 | 14,807 | 23,846 | 31,478 | 53,723 | 111,308 | 218,767 |

Самая длинная объединённая последовательность относится к `user_id=21695`: 228,030 взаимодействий, 218,796 уникальных видео, период 20220408-20220508. Для этого пользователя точная сводка дублей по `(video_id, time_ms)`:

| user_id | distinct_video_time_keys | duplicate_video_time_keys | extra_rows_over_unique_video_time_keys | max_rows_per_video_time_key |
| --- | --- | --- | --- | --- |
| 21,695 | 225,562 | 2,468 | 2,468 | 2 |

## 9. Длинный хвост и популярность видео

| policy | items | interactions_mean | interactions_std | interactions_cv | interactions_gini | interactions_median | interactions_p95 | interactions_p99 | interactions_p999 | interactions_max | unique_users_mean | unique_users_p99 | unique_users_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 7,583 | 156.4103 | 30.8296 | 0.1971 | 0.0808 | 162 | 184 | 194 | 205 | 214 | 156.4033 | 194 | 214 |
| standard | 32,038,693 | 10.0590 | 66.1035 | 6.5716 | 0.8271 | 1 | 30 | 168 | 791 | 15,380 | 9.7669 | 163 | 11,441 |

Популярность видео в standard имеет сильный long tail: median=1, p99=168, p99.9=791, max=15,380. Random exposure гораздо ровнее: median=162, p99=194, max=214.

## 10. Равномерность random exposure

Random exposure ближе к равномерному показу, но не perfectly uniform: CV=0.197, Gini=0.081, min=1, p90=179, p95=184, p99=194, max=214. Для standard CV=6.572 и Gini=0.827, то есть exposure намного более концентрирован.

## 11. Временная структура

| policy | period | interactions | users | items | date_min | date_max | is_click_rate | long_view_rate | is_like_rate | play_time_ms_mean | play_time_ms_median | duration_ms_mean | duration_ms_median | duration_non_positive_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 2022-04-22_to_2022-05-08 | 1,186,059 | 27,285 | 7,583 | 20220422 | 20220508 | 17.616% | 8.496% | 0.480% | 6,935.5 | 2,091.0 | 104,438.8 | 76,833.0 | 36,920 |
| standard | 2022-04-08_to_2022-04-21 | 136,296,576 | 26,904 | 15,293,075 | 20220408 | 20220421 | 38.077% | 26.285% | 1.631% | 14,968.8 | 3,119.0 | 72,464.8 | 29,560.0 | 11,638,395 |
| standard | 2022-04-22_to_2022-05-08 | 185,981,809 | 27,285 | 19,865,544 | 20220422 | 20220508 | 37.721% | 25.869% | 1.690% | 14,519.8 | 3,110.0 | 72,451.0 | 28,466.0 | 15,635,611 |

| policy | dates | min_daily_interactions | min_date | max_daily_interactions | max_date | click_rate_range | long_view_rate_range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 20220422-20220508 | 24,849 | 20220422 | 114,925 | 20220508 | 16.112%-18.466% | 7.761%-9.102% |
| standard | 20220408-20220508 | 8,994,629 | 20220419 | 12,573,232 | 20220430 | 36.668%-40.163% | 25.127%-27.903% |

Между standard-периодами 08.04-21.04 и 22.04-08.05 есть небольшой наблюдаемый сдвиг: click rate снижается с 38.077% до 37.721%, long_view rate с 26.285% до 25.869%, like rate растёт с 1.631% до 1.690%. Это наблюдаемое изменение распределений, а не утверждение о статистической значимости.

## 12. Анализ tab/scenario

| policy | tab | interactions | policy_share | users | items | is_click_rate | long_view_rate | is_like_rate | is_hate_rate | play_time_ms_median | duration_non_positive_count | duration_non_positive_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 1 | 1,178,025 | 99.323% | 27,000 | 7,537 | 17.729% | 8.535% | 0.481% | 0.113% | 2,102 | 36,632 | 3.110% |
| random | 11 | 6,755 | 0.570% | 304 | 4,378 | 0.000% | 2.369% | 0.237% | 0.163% | 0 | 239 | 3.538% |
| random | 2 | 764 | 0.064% | 190 | 725 | 11.387% | 4.581% | 0.000% | 0.785% | 1,573 | 30 | 3.927% |
| random | 14 | 515 | 0.043% | 38 | 498 | 0.000% | 6.214% | 0.583% | 0.000% | 1,565 | 19 | 3.689% |
| standard | 1 | 215,925,023 | 67.000% | 27,088 | 20,231,074 | 44.364% | 31.237% | 2.081% | 0.047% | 4,959 | 16,767,533 | 7.765% |
| standard | 0 | 63,923,841 | 19.835% | 26,458 | 15,544,845 | 11.598% | 3.803% | 0.473% | 0.050% | 0 | 8,838,776 | 13.827% |
| standard | 4 | 24,510,543 | 7.605% | 22,219 | 4,296,333 | 54.448% | 41.191% | 1.311% | 0.043% | 8,147 | 682,697 | 2.785% |
| standard | 2 | 9,949,649 | 3.087% | 4,536 | 3,501,714 | 42.421% | 33.314% | 1.985% | 0.099% | 5,000 | 747,285 | 7.511% |
| standard | 6 | 5,268,393 | 1.635% | 22,626 | 287,024 | 17.772% | 8.464% | 0.794% | 0.029% | 2,128 | 157,726 | 2.994% |
| standard | 3 | 957,605 | 0.297% | 17,674 | 219,637 | 2.114% | 0.360% | 0.070% | 0.037% | 0 | 49,582 | 5.178% |
| standard | 8 | 478,050 | 0.148% | 4,348 | 132,353 | 4.048% | 2.029% | 0.077% | 0.034% | 0 | 0 | 0.000% |
| standard | 5 | 429,129 | 0.133% | 2,713 | 107,251 | 23.651% | 12.276% | 0.787% | 0.121% | 2,450 | 11,980 | 2.792% |
| standard | 12 | 332,086 | 0.103% | 2,389 | 95,321 | 12.237% | 8.883% | 0.182% | 0.001% | 0 | 0 | 0.000% |
| standard | 11 | 217,754 | 0.068% | 358 | 159,996 | 0.059% | 11.243% | 0.479% | 0.034% | 0 | 12,122 | 5.567% |
| standard | 7 | 114,704 | 0.036% | 1,915 | 82,673 | 44.139% | 23.132% | 2.747% | 0.011% | 0 | 5,568 | 4.854% |
| standard | 13 | 72,217 | 0.022% | 95 | 63,398 | 36.306% | 29.337% | 0.363% | 0.000% | 2,551 | 0 | 0.000% |
| standard | 9 | 60,678 | 0.019% | 2,081 | 37,047 | 99.234% | 23.585% | 1.389% | 0.002% | 2,202 | 0 | 0.000% |
| standard | 10 | 23,722 | 0.007% | 1,281 | 18,451 | 99.966% | 62.052% | 2.074% | 0.000% | 53,798 | 0 | 0.000% |
| standard | 14 | 14,991 | 0.005% | 55 | 14,572 | 0.027% | 27.123% | 1.394% | 0.020% | 2,692 | 737 | 4.916% |

Крупнейший standard-сценарий - `tab=1` с 67.000% standard-взаимодействий; далее `tab=0` с 19.835% и `tab=4` с 7.605%. Random почти полностью находится в `tab=1` (99.323%). `duration_ms=0` в standard особенно заметен в `tab=0` (13.827% строк) и `tab=1` (16.77M строк, 7.765%). В random доля `duration_ms=0` около 3.0-3.9% по всем четырем tabs.

## 13. Покрытие пользовательских и видео-признаков

| feature_table | policy | interactions | interactions_with_feature | interaction_coverage_share | unique_interaction_entities | unique_interaction_entities_with_feature | unique_entity_coverage_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| user_features | random | 1,186,059 | 1,186,059 | 100.000% | 27,285 | 27,285 | 100.000% |
| user_features | standard | 322,278,385 | 322,278,385 | 100.000% | 27,285 | 27,285 | 100.000% |
| video_basic | random | 1,186,059 | 1,186,059 | 100.000% | 7,583 | 7,583 | 100.000% |
| video_basic | standard | 322,278,385 | 322,278,385 | 100.000% | 32,038,693 | 32,038,693 | 100.000% |
| video_statistic | random | 1,186,059 | 1,186,059 | 100.000% | 7,583 | 7,583 | 100.000% |
| video_statistic | standard | 322,278,385 | 322,278,385 | 100.000% | 32,038,693 | 32,038,693 | 100.000% |

Все пользователи из interaction logs покрыты `user_features_27k.csv`; все наблюдаемые видео из interaction logs покрыты `video_features_basic_27k.csv` и `video_features_statistic_27k_part*.csv`. Это покрытие признаков, а не разрешение использовать statistic features без leakage-проверки.

## 14. Проверки качества данных

| policy | signal | unique_values | min_value | max_value | missing_count | invalid_count |
| --- | --- | --- | --- | --- | --- | --- |
| random | is_click | 2 | 0 | 1 | 0 | 0 |
| random | is_comment | 2 | 0 | 1 | 0 | 0 |
| random | is_follow | 2 | 0 | 1 | 0 | 0 |
| random | is_forward | 2 | 0 | 1 | 0 | 0 |
| random | is_hate | 2 | 0 | 1 | 0 | 0 |
| random | is_like | 2 | 0 | 1 | 0 | 0 |
| random | is_profile_enter | 2 | 0 | 1 | 0 | 0 |
| random | long_view | 2 | 0 | 1 | 0 | 0 |
| standard | is_click | 2 | 0 | 1 | 0 | 0 |
| standard | is_comment | 2 | 0 | 1 | 0 | 0 |
| standard | is_follow | 2 | 0 | 1 | 0 | 0 |
| standard | is_forward | 2 | 0 | 1 | 0 | 0 |
| standard | is_hate | 2 | 0 | 1 | 0 | 0 |
| standard | is_like | 2 | 0 | 1 | 0 | 0 |
| standard | is_profile_enter | 2 | 0 | 1 | 0 | 0 |
| standard | long_view | 2 | 0 | 1 | 0 | 0 |

Проверки дублирования ключей:

| key | policy | distinct_keys | duplicate_keys | extra_rows_over_unique_keys | max_rows_per_key |
| --- | --- | --- | --- | --- | --- |
| user_id+time_ms | random | 1,186,049 | 10 | 10 | 2 |
| user_id+time_ms | standard | 55,717,007 | 51,570,391 | 266,561,378 | 70 |
| user_id+video_id+time_ms | random | 1,186,049 | 10 | 10 | 2 |
| user_id+video_id+time_ms | standard | 319,382,274 | 2,878,498 | 2,896,111 | 6 |

Повторный просмотр нельзя считать дубликатом только по `(user_id, video_id)`. Более строгий ключ `(user_id, video_id, time_ms)` показывает 2,896,111 лишних строк сверх уникальных ключей в standard и 10 в random. Ключ `(user_id, time_ms)` сильно менее специфичен: он ловит несколько видео/событий в один timestamp и поэтому дает гораздо больше совпадений.

## 15. Риски утечки

Главный риск - `video_features_statistic_27k_part*.csv`: это агрегированные item-level statistics (`show_cnt`, `play_cnt`, `like_cnt`, `comment_cnt`, `follow_cnt`, `share_cnt`, `collect_cnt` и т.д.). В файлах нет time-aware версии признаков для момента каждого interaction. Если эти statistics рассчитаны на полном периоде, то при chronological/sequential training они могут содержать future exposure/feedback относительно train interaction. Без реконструкции train-window-only или time-aware statistics эти признаки нельзя автоматически включать в baseline.

Дополнительные риски: popularity features, рассчитанные на full log; filtering users/items до split; preprocessing fitted on full dataset; mixing standard/random logs без policy-aware design; target-derived aggregates.

## 16. Выводы для train/validation/test

- Leave-one-out: sequence median около 8,646 combined interactions/user и 8,594 standard interactions/user делает user-level sequential split практически применимым; нужно явно задать truncation/windowing, потому что p99=56,230 и max=228,000 очень велики.
- Chronological split: датасет имеет два явных standard-периода; split по времени естественен для production-like evaluation, но нужно контролировать changing item universe, feature windows и eligibility.
- Standard training + randomized evaluation: random log полезен для debiased/off-policy evaluation, но candidate distribution радикально другая: 7,583 random items против 32,038,693 standard items, median random interactions/user=22. Random test не является drop-in заменой standard test для любой задачи.

## 17. Выводы для рекомендательного моделирования

- Для full-scale sequential/top-K baseline использовать standard logs 27K после выбора опубликованного протокола.
- Для разработки и sanity/debugging использовать 1K/Pure; полный 27K запускать batch/lazy scripts.
- `duration_ms=0` не удалять автоматически: это массовый паттерн, особенно в standard `tab=0/1`; для watch-time targets нужно явное правило обработки.
- `play_ratio > 1` сохранить как raw signal; для plots можно clip, но в training targets нужна осознанная нормализация или отдельный capped feature.
- Random exposure подходит для задач exposure bias / off-policy / causal design, но требует отдельного candidate-policy protocol.

## 18. Рекомендуемые следующие шаги

1. Выбрать конкретную опубликованную KuaiRand benchmark paper и воспроизвести её preprocessing/split/evaluation.
2. Зафиксировать eligibility, negative sampling, repeated-item policy и truncation для длинных histories.
3. Решить, используется ли random exposure как отдельный evaluation component или только как источник bias diagnostics.
4. Не использовать `video_features_statistic_27k_part*.csv` в baseline до доказанной time-aware безопасности.
5. После воспроизведения базовой модели переходить к новому методу и ablations.
