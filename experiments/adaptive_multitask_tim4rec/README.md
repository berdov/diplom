# Адаптивный Multitask TiM4Rec

Эта ветка готовит следующий исследовательский этап после `multitask_tim4rec_tuned_001`: адаптивную multitask-оптимизацию для KuaiRand Protocol B.

Границы этапа:

- backbone, Protocol B split, targets, recommendation objective и evaluation не меняются;
- `rank` остаётся главной задачей с фиксированным весом `1`;
- adaptive методы управляют только contribution/gradient четырёх auxiliary задач;
- используются реальные train batches KuaiRand, не synthetic data;
- test split не загружается и не оценивается, `test_evaluation_count=0`;
- full training, 5-epoch sanity, Optuna и MoE на этом этапе не запускаются.

Структура директории:

- `config.yaml` - единый smoke/config для будущих sanity-запусков;
- `methods/gradnorm.py` - GradNorm для auxiliary losses;
- `methods/pcgrad.py` - PCGrad, включая ranking-anchored вариант;
- `methods/metabalance.py` - MetaBalance-Fix для recommender-style auxiliary gradient magnitude balancing;
- `methods/common.py` - shared-gradient diagnostics;
- `smoke_test.py` - один compact smoke run на train batches;
- `runs/adaptive_smoke_001.json` - smoke-артефакт после запуска.

Запуск на cluster E:

```bash
sbatch slurm/adaptive_multitask_tim4rec.sh
```

Скрипт использует `experiments/multitask_tim4rec_optuna/prepare_validation_only.py`, чтобы RecBole benchmark содержал только `train` и `valid`. Smoke runner берёт только `train_data`.
