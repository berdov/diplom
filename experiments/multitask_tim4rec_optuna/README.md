# Optuna search для Multitask TiM4Rec

Validation-only Optuna search для эксперимента `MultitaskTiM4Rec` с
фиксированной архитектурой.

Study тюнит только политику multitask-оптимизации поверх
`multitask_tim4rec_001`: общий масштаб auxiliary loss, нормированные веса задач,
imbalance exponents, learning rate, weight decay, существующий dropout и
множитель learning rate для auxiliary heads.

Test на этом этапе закрыт. Search использует физический RecBole dataset только
из train и validation, `eval_args.split.LS=valid_only`, full-ranking validation
NDCG@10 как единственную objective и `test_evaluation_count=0`.

Команды для кластера:

```bash
sbatch slurm/multitask_tim4rec_optuna.sh
MULTITASK_OPTUNA_STAGE=search sbatch slurm/multitask_tim4rec_optuna.sh
```

Persistent storage для study:

```text
/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/optuna.db
```
