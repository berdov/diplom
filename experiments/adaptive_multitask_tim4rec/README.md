# Адаптивный Multitask TiM4Rec

Эта ветка готовит следующий исследовательский этап после `multitask_tim4rec_tuned_001`: адаптивную multitask-оптимизацию для KuaiRand Protocol B.

Границы этапа:

- backbone, Protocol B split, targets, recommendation objective и evaluation не меняются;
- `rank` остаётся главной задачей с фиксированным весом `1`;
- adaptive методы управляют только contribution/gradient четырёх auxiliary задач;
- используются реальные train batches KuaiRand, не synthetic data;
- test split не загружается и не оценивается, `test_evaluation_count=0`;
- full training, Optuna и MoE на этом этапе не запускаются;
- 5-epoch sanity запускаются только для PCGrad ranking-anchored и MetaBalance-Fix.

Структура директории:

- `config.yaml` - единый smoke/config для будущих sanity-запусков;
- `methods/gradnorm.py` - GradNorm для auxiliary losses;
- `methods/pcgrad.py` - PCGrad, включая ranking-anchored вариант;
- `methods/metabalance.py` - MetaBalance-Fix для recommender-style auxiliary gradient magnitude balancing;
- `methods/common.py` - shared-gradient diagnostics;
- `smoke_test.py` - один compact smoke run на train batches;
- `sanity_train.py` - 5-epoch validation-only sanity для PCGrad/MetaBalance;
- `build_comparison.py` - comparison report после двух sanity runs;
- `runs/adaptive_smoke_001.json` - smoke-артефакт после запуска;
- `runs/pcgrad_sanity_001.json` - 5-epoch validation-only sanity для ranking-anchored PCGrad;
- `runs/metabalance_sanity_001.json` - 5-epoch validation-only sanity для MetaBalance-Fix;
- `runs/adaptive_sanity_comparison_001.md` - сравнение двух sanity runs с validation reference.

Запуск на cluster E:

```bash
sbatch slurm/adaptive_multitask_tim4rec.sh
```

Скрипт использует `experiments/multitask_tim4rec_optuna/prepare_validation_only.py`, чтобы RecBole benchmark содержал только `train` и `valid`. Smoke runner берёт только `train_data`.

Запуск 5-epoch sanity на cluster E:

```bash
ADAPTIVE_MTL_METHOD=pcgrad sbatch slurm/adaptive_multitask_sanity.sh
ADAPTIVE_MTL_METHOD=metabalance sbatch slurm/adaptive_multitask_sanity.sh
```

Скрипт настроен на короткую non-preemptive очередь `test` и GPU constraint `type_e`. Финальные sanity runs `pcgrad_sanity_001` и `metabalance_sanity_001` были выполнены через явный Slurm override `--constraint=type_h`, потому что `gpu-ef-quick/type_e` дважды вытеснил PCGrad, а `test/type_e` имел поздний старт.

Эти sanity runs используют tuned fixed trial `110`, полный train split и full-ranking validation. GradNorm, Optuna, full training и test на этом этапе не запускаются.
