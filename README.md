# Дипломный проект

Исследуем последовательную рекомендацию видео на KuaiRand: предсказываем следующее взаимодействие по истории пользователя. Сравниваем vanilla Mamba3 с вариантами, использующими интервалы между историческими событиями.

## Реализация

- [Vanilla Mamba3Rec](experiments/mamba3_baseline/README.md): primary-only, без временных признаков.
- [RT-Mamba3](experiments/mamba3_timeaware/README.md): общий временной множитель.
- [Раздельные временные механизмы](experiments/mamba3_time_mechanisms/README.md): `decay_only`, `scan_only`, `separate`.
- [Входы, разбиение и оценка](reports/EVALUATION_SETUP.md): до 50 событий, полный каталог, точные ограничения временной корректности.

Текущая [модель](experiments/mamba3_time_mechanisms/model.py), [конфигурация](experiments/mamba3_time_mechanisms/config.py) и [кластерный launcher](slurm/mamba3_time_mechanisms_validation.sh) описаны в README эксперимента. Launcher привязан к HSE, существующему окружению и проверенному GPU evidence; это не универсальная команда запуска.

## Результаты

- [Наши VALID и TEST](reports/RESULTS.md), включая vanilla TEST NDCG@10 **0.0590**.
- [Опубликованные ориентиры и наши TEST](reports/PAPER_RESULTS.md): все cutoff и ограничения сопоставимости.
- [Подтверждение временных механизмов: таблицы и график](reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#confirmation).
- [Реестр запусков](experiments/results.csv) с источниками метрик.

Separate превысил shared по VALID NDCG@10 во всех пяти парных seeds: +1,91% по средним, на четырёх новых seeds +1,25%. Constant-gap control выполнен на одном seed; новых TEST нет.

Завершённые исследования: [MTL/MOO](reports/MTL_MOO_STUDY.md) и [Proto-Mamba3](experiments/mamba3_prototypes/README.md). Их результаты и ограничения сохранены, но они не выбраны основной линией модели.
