# Происхождение сохранённых результатов

Снимок завершённых запусков получен 10 сентября 2026 через read-only SSH в экспериментальной ветке. Для canonical обновления он перенесён из commit `47953484490964febf47c05733545bdc29308702` с повторной проверкой SHA-256 всех 25 файлов; повторное обучение и оценка моделей не проводились. [Canonical summary](../RESULTS.md) · [Аудит обновления](../CANONICAL_RESULTS_AUDIT.md).

- Challenger source checkout: `/home/daryumin/iberdov/diplom_exp_moo_challengers`, code SHA `1a98ef966a2923a8f234b71602291b1c71ab82c3`. Три convergence JSON сохранены в `experiments/moo_representative_challengers/runs/`.
- Target-combination source checkout: `/home/daryumin/iberdov/diplom_exp_target_combinations_002`, code SHA `599bcdb6e50dceea73233169b8834eaceef083a8`. В `target_combinations/` сохранены неизменённые summary, config, combinations, два CSV и 17 run JSON попытки 002 (16 scientific combinations и отдельный smoke). Smoke не входит в completeness 16/16.
- `completed_run_hashes.json`: полный список скопированных файлов и SHA-256 исходных байтов; копии после записи проверены повторно.
- `completed_audit.json`: результат read-only проверки completeness, отдельных метрик и пересчёта эффектов. Обучение/оценка модели не запускались.

При подготовке исходного снимка текст `reports/MOO_FAMILIES.md` синхронизирован выборочно с `origin/main` на `16ba99b69929a711137f616df28e18e7f92b76c1`, затем дополнен выводами; рабочая ветка тогда оставалась `exp/moo-representative-challengers`. Canonical обновление подготовлено отдельно от main `16ba99b69929a711137f616df28e18e7f92b76c1` в ветке `docs/canonical-moo-mtl-results`. Исходная таблица Stage 1 и байты evidence сохранены. Код экспериментов из challenger-ветки не переносился; ссылки на него в отчёте закреплены на конкретных commits.

Схема `experiments/results.csv` сохранена. Добавлены только 3 строки `challenger_convergence` и 16 строк `target_combination_validation_screening`; в каждой указан локальный `source_json`, `notes_path`, исходный code SHA, VALID и нулевое число TEST evaluations. Исходные два CSV target screening используют свою схему и остаются неизменёнными источниками, а не заменой общего реестра.
