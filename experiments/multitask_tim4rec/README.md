# KuaiRand multitask TiM4Rec preparation

Этот каталог фиксирует data/audit этап перед первой собственной multitask/behavior-aware моделью. На этом этапе модель не обучается: код только связывает существующий Protocol B split с raw behavior labels KuaiRand-Pure, проверяет join и сохраняет компактные статистики.

Основной вход:

- Protocol B parquet: `/home/daryumin/iberdov/diplom/data/processed/protocol_b`
- raw source log: `/home/daryumin/iberdov/Corpora/KuaiRand-Pure/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv`

Основной выход:

- multitask parquet: `/home/daryumin/iberdov/diplom/data/processed/protocol_b_multitask`
- manifest: `outputs/data/protocol_b_multitask_manifest.json`
- audit: `experiments/multitask_tim4rec/AUDIT.md`
- compact CSV: `experiments/multitask_tim4rec/*.csv`

Запуск аудита на cHARISMa:

```bash
/home/daryumin/iberdov/diplom/.conda/bin/python \
  experiments/multitask_tim4rec/audit_targets.py
```

Leakage policy: labels текущего interaction (`is_click`, `long_view`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `is_profile_enter`, watch-time поля) могут быть supervised targets, но не input features для предсказания этого же interaction.
