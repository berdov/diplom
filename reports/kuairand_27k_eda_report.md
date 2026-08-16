# Отчёт EDA KuaiRand-27K

Отчёт построен по реальным результатам cHARISMa из `outputs/eda/27k_summary.json` и компактных CSV-файлов в `outputs/eda/`.

## 1. Обзор датасета

- Путь к данным на cHARISMa: `/home/daryumin/iberdov/Corpora/KuaiRand-27K/KuaiRand-27K`.
- Размер директории KuaiRand-27K, просканированной для инвентаризации: 45.6 GiB.
- Просканировано исходных файлов взаимодействий: 5.
- Финальное Slurm-задание: `4253788`, состояние `COMPLETED`, узел `cn-046`, время выполнения `00:03:35`.
- Ресурсы: `partition=rocky`, `constraint=type_e`, `cpus-per-task=16`, `gres=gpu:a100:1`, `mem=0`.
- Peak RSS по `sacct`: `46069996K`.

Важно: `video_features_statistic_27k_part*.csv` не сканировались в этом скрипте. Это агрегированные статистики видео, а не индивидуальные взаимодействия; такие признаки потенциально создают temporal leakage.

## 2. Документация / README vs вычисленные значения

| Показатель | Документация / README | Вычислено по файлам | Комментарий |
| --- | --- | --- | --- |
| Пользователи | 27,285 | standard=27,285; random=27,285 | совпадает для обеих политик |
| Взаимодействия standard | 322,278,385 | 322,278,385 | совпадает |
| Взаимодействия random | 1,186,059 | 1,186,059 | совпадает |
| Items standard | 32,038,725 | 32,038,693 | вычислено на 32 меньше |
| Items random | 7,583 | 7,583 | совпадает |

Документированные числа взаимодействий совпали точно. Единственное расхождение: вычисленное число уникальных `video_id` в standard-логах взаимодействий меньше README-числа standard items на 32 items. Это может означать различие между задокументированным standard-policy universe и реально наблюдаемыми item IDs в просканированных log-файлах. Для benchmark reporting это нужно явно фиксировать.

## 3. Исходные файлы и временные диапазоны

| Политика | Исходный лог | Взаимодействия | Пользователи | Items | Диапазон дат | Уникальные tabs |
| --- | --- | --- | --- | --- | --- | --- |
| random | log_random_4_22_to_5_08_27k | 1,186,059 | 27,285 | 7,583 | 20220422-20220508 | 4 |
| standard | log_standard_4_08_to_4_21_27k_part1 | 68,148,288 | 26,858 | 10,221,515 | 20220408-20220421 | 15 |
| standard | log_standard_4_08_to_4_21_27k_part2 | 68,148,288 | 26,857 | 10,414,229 | 20220408-20220421 | 15 |
| standard | log_standard_4_22_to_5_08_27k_part1 | 93,062,020 | 27,285 | 13,214,491 | 20220422-20220508 | 15 |
| standard | log_standard_4_22_to_5_08_27k_part2 | 92,919,789 | 27,285 | 13,347,412 | 20220422-20220508 | 15 |

Standard-трафик покрывает `20220408`-`20220508`; random exposure покрывает `20220422`-`20220508`.

## 4. Сводка standard vs random

- Standard-взаимодействия: 322,278,385; пользователей: 27,285; items: 32,038,693; уникальных `tab`: 15.
- Random-взаимодействия: 1,186,059; пользователей: 27,285; items: 7,583; уникальных `tab`: 4.
- Доля random-взаимодействий: 0.367%.

## 5. Частоты feedback-сигналов

| Сигнал | Positive в standard | Rate в standard | Positive в random | Rate в random |
| --- | --- | --- | --- | --- |
| is_click | 122,052,542 | 37.872% | 208,934 | 17.616% |
| is_comment | 810,580 | 0.252% | 411 | 0.035% |
| is_follow | 343,328 | 0.107% | 310 | 0.026% |
| is_forward | 280,970 | 0.087% | 402 | 0.034% |
| is_hate | 156,233 | 0.048% | 1,351 | 0.114% |
| is_like | 5,366,646 | 1.665% | 5,691 | 0.480% |
| is_profile_enter | 5,777,660 | 1.793% | 6,597 | 0.556% |
| long_view | 83,936,743 | 26.045% | 100,769 | 8.496% |

В standard-policy трафике выше rates для `is_click`, `long_view`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_profile_enter`. В random exposure выше rate для `is_hate`. Это наблюдаемая разница распределений, а не causal effect.

## 6. Статистика длины последовательностей

| Срез | Показатель | Пользователи | Min | Mean | Median | P75 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | interactions_per_user | 27,285 | 110 | 11855.03 | 8,646 | 15,279 | 24,650 | 32,640 | 56,244 | 228,030 |
| all | unique_videos_per_user | 27,285 | 109 | 11511.97 | 8,434 | 14,867 | 23,923 | 31,528 | 53,787 | 218,796 |
| random | interactions_per_user | 27,285 | 10 | 43.47 | 22 | 45 | 96 | 152 | 309 | 1,580 |
| random | unique_videos_per_user | 27,285 | 10 | 43.47 | 22 | 45 | 96 | 152 | 309 | 1,580 |
| standard | interactions_per_user | 27,285 | 100 | 11811.56 | 8,594 | 15,227 | 24,587 | 32,590 | 56,230 | 228,000 |
| standard | unique_videos_per_user | 27,285 | 99 | 11468.53 | 8,384 | 14,807 | 23,845 | 31,478 | 53,722 | 218,767 |

KuaiRand-27K имеет длинные пользовательские истории: median all-policy interactions per user = 8,646, p99 = 56,244. Для sequential recommendation это полезно, но heavy-tail требует заранее заданного протокола усечения.

## 7. Статистика watch-time

| Policy | Duration <= 0 | Duration mean | Duration median | Duration p99 | Play mean | Play median | Play p99 | Ratio mean | Ratio median | Ratio p99 | Ratio max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 36,920 | 104438.8 | 76,833 | 507,666 | 6935.5 | 2,091 | 91,334 | 0.1413 | 0.0348 | 1.2362 | 103.13 |
| standard | 27,274,006 | 72456.9 | 28,933 | 483,480 | 14709.7 | 3,114 | 168,051 | 0.4171 | 0.1144 | 3.0600 | 348.95 |

`play_ratio` считался только для `duration_ms > 0`. В обеих политиках есть строки с `duration_ms <= 0`; их нужно обрабатывать явно. Значения `play_ratio > 1` встречаются и не должны silently clipped в preprocessing без документированного protocol.

## 8. Распределение `tab` / сценариев

Топ `tab` по числу взаимодействий внутри каждой policy:

| Policy | Tab | Взаимодействия | Доля внутри policy | Пользователи | Items | Click rate | Long-view rate | Like rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 1 | 1,178,025 | 99.323% | 27,000 | 7,537 | 17.729% | 8.535% | 0.481% |
| random | 11 | 6,755 | 0.570% | 304 | 4,378 | 0.000% | 2.369% | 0.237% |
| random | 2 | 764 | 0.064% | 190 | 725 | 11.387% | 4.581% | 0.000% |
| random | 14 | 515 | 0.043% | 38 | 498 | 0.000% | 6.214% | 0.583% |
| standard | 1 | 215,925,023 | 67.000% | 27,088 | 20,231,074 | 44.364% | 31.237% | 2.081% |
| standard | 0 | 63,923,841 | 19.835% | 26,458 | 15,544,845 | 11.598% | 3.803% | 0.473% |
| standard | 4 | 24,510,543 | 7.605% | 22,219 | 4,296,333 | 54.448% | 41.191% | 1.311% |
| standard | 2 | 9,949,649 | 3.087% | 4,536 | 3,501,714 | 42.421% | 33.314% | 1.985% |
| standard | 6 | 5,268,393 | 1.635% | 22,626 | 287,024 | 17.772% | 8.464% | 0.794% |
| standard | 3 | 957,605 | 0.297% | 17,674 | 219,637 | 2.114% | 0.360% | 0.070% |
| standard | 8 | 478,050 | 0.148% | 4,348 | 132,353 | 4.048% | 2.029% | 0.077% |
| standard | 5 | 429,129 | 0.133% | 2,713 | 107,251 | 23.651% | 12.276% | 0.787% |

Random exposure почти полностью сосредоточен в `tab=1`; standard-трафик использует 15 значений `tab` и доминируется `tab=1`, `tab=0`, `tab=4`.

## 9. Наблюдения по long tail

| Policy | Items | Среднее interactions/item | Median | P90 | P99 | Max | Среднее unique users/item | P99 unique users | Max unique users |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 7,583 | 156.41 | 162 | 179 | 194 | 214 | 156.40 | 194 | 214 |
| standard | 32,038,693 | 10.06 | 1 | 12 | 168 | 15,380 | 9.77 | 163 | 11,441 |

Standard-policy item popularity имеет выраженный long tail: медиана interactions per item = 1, p99 = 168, max = 15,380. Random exposure значительно более равномерный внутри малого candidate pool.

## 10. Найденные проблемы качества данных

- Бинарные feedback-колонки содержат только `{0, 1}` в обеих policies; invalid count = 0 для всех проверенных feedback-сигналов.
- Missing count = 0 для всех проверенных feedback-сигналов.
- `duration_ms <= 0`: 27,274,006 standard-строк и 36,920 random-строк.
- `play_ratio` имеет экстремальные значения выше 1; это важно для визуализации, filtering и построения target.

## 11. Риски leakage

Для 27K особенно важны:

1. Агрегированные video statistic features могут содержать future information относительно training interaction.
2. Popularity features, рассчитанные на полном логе, будут leak future exposure/feedback.
3. Filtering пользователей/items до temporal split может leak future eligibility.
4. Смешивание standard и random logs без policy-aware design искажает evaluation.
5. Агрегаты, построенные от target, и preprocessing, fitted on full dataset, могут leak validation/test data.

## 12. Выводы для train / validation / test

Сырой KuaiRand-27K не задаёт один универсальный train/validation/test split. Для будущего baseline нужно определить:

- задачу: sequential next-item, top-K ranking, feedback prediction или debiased/random-exposure evaluation;
- семейство split: user-level leave-one-out, global chronological split или standard-training + random-exposure evaluation;
- item/user eligibility;
- candidate set и negative sampling;
- роль `tab` и `policy`: feature, filter или stratification key.

## 13. Что использовать для baselines

- KuaiRand-1K: отладка protocol и проверки воспроизведения baseline.
- KuaiRand-27K standard logs: full-scale sequential/top-K baselines после выбора published protocol.
- Random exposure: debiased evaluation/off-policy component только если protocol явно это поддерживает.
- Aggregated video statistic features: не включать в baseline до проверки временного окна агрегации.

## 14. Следующий шаг

Выбрать конкретную опубликованную benchmark-статью по KuaiRand, воспроизвести preprocessing/split/evaluation protocol, сверить baseline metrics с опубликованными результатами, затем переходить к новому методу и ablations.
