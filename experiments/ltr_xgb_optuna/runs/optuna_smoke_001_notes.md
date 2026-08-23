# Optuna smoke 001

## Цель

Проверить pipeline подбора гиперпараметров XGBoost LambdaMART поверх `ltr_xgb_002` без изменения признаков, candidates и evaluation protocol.

## Test policy

- Test evaluation count: `0`.
- Test split не загружался и не использовался в objective, pruning, model selection или отчетной метрике.
- Objective: validation full-ranking `NDCG@10`.

## Baseline

- Base run: `ltr_xgb_002`.
- Training candidates: `sampled_100`.
- Validation objective candidates: `full_7111_items`.
- Feature set version: `8abcf619f10433225767952859d305ec507d2b00eaca5b2118a79d7f29730a25`.

## Optuna

- Study name: `ltr_xgb_optuna_v1`.
- Storage: `sqlite:////home/daryumin/iberdov/diplom/experiments/ltr_xgb_optuna/optuna.db`.
- Sampler: `TPESampler`.
- Sampler seed: `2026`.

## Smoke trial

- Trial number: `0`.
- Status: `COMPLETE`.
- Best iteration: `19`.
- Best boosted rounds: `20`.
- Validation NDCG@10: `0.014951`.
- Validation HR@10: `0.030771`.
- Trial runtime: `83.57` sec.
- XGBoost train time: `2.62` sec.
- Best full-ranking validation time: `41.59` sec.

## Parameters

```json
{
  "colsample_bytree": 0.7396449298400908,
  "eta": 0.022942438908854185,
  "eval_metric": "ndcg@10",
  "gamma": 0.0,
  "max_depth": 4,
  "min_child_weight": 17.671260541443075,
  "nthread": 8,
  "objective": "rank:ndcg",
  "reg_alpha": 0.0009788258835751811,
  "reg_lambda": 86.64678862932995,
  "seed": 42,
  "subsample": 0.5444495108240646,
  "tree_method": "hist",
  "verbosity": 1
}
```

## Full-ranking validation cost

- Users: `23951`.
- Items: `7111`.
- Scores per full validation: `170315561`.
- Batch users: `128`.

## Slurm

- Job: `4271164`.
- Partition: `rocky`.
- Constraint: `type_d`.
- Node: `cn-038`.
- State / exit: `COMPLETED` / `0:0`.
- Elapsed: `00:01:38`.
- AllocTRES: `billing=8,cpu=8,node=1`.
- Batch MaxRSS: `2023540K`.
- Batch MaxVMSize: `5441864K`.

## Caching

- Reused artifact root: `/home/daryumin/iberdov/diplom/experiments/ltr_xgb_baseline/ltr_xgb_002`.
- Forbidden test paths loaded: `[]`.

## Decision

- Pipeline creates/resumes SQLite Optuna study.
- Trial parameters are passed to XGBoost.
- Model selection uses full-ranking validation, not sampled validation metric.
- This smoke trial is not a final result and is not written to `experiments/results.csv`.
