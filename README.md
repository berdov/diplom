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
- [Временные абляции: таблицы и графики](reports/MAMBA3_TIME_MECHANISMS_RESULTS.md).
- [Реестр запусков](experiments/results.csv) с источниками метрик.

Завершённые исследования: [MTL/MOO](reports/MTL_MOO_STUDY.md) и [Proto-Mamba3](experiments/mamba3_prototypes/README.md). Их результаты и ограничения сохранены, но они не выбраны основной линией модели.
