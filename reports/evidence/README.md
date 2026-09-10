# Происхождение сохранённых результатов

Снимок завершённых запусков получен 10 сентября 2026 через read-only SSH.

- Challenger source checkout: `/home/daryumin/iberdov/diplom_exp_moo_challengers`, code SHA `1a98ef966a2923a8f234b71602291b1c71ab82c3`. Три convergence JSON сохранены в `experiments/moo_representative_challengers/runs/`.
- Target-combination source checkout: `/home/daryumin/iberdov/diplom_exp_target_combinations_002`, code SHA `599bcdb6e50dceea73233169b8834eaceef083a8`. В `target_combinations/` сохранены неизменённые summary, config, combinations, два CSV и 17 run JSON попытки 002 (16 scientific combinations и отдельный smoke). Smoke не входит в completeness 16/16.
- `completed_run_hashes.json`: полный список скопированных файлов и SHA-256 исходных байтов; копии после записи проверены повторно.
- `completed_audit.json`: результат read-only проверки completeness, отдельных метрик и пересчёта эффектов. Обучение/оценка модели не запускались.

Текст `reports/MOO_FAMILIES.md` синхронизирован выборочно с `origin/main` на `16ba99b69929a711137f616df28e18e7f92b76c1`, затем дополнен выводами. Рабочая ветка осталась `exp/moo-representative-challengers`; merge, rebase и обновление code checkout кластера не выполнялись. Исходная таблица Stage 1 сохранена буквально.
