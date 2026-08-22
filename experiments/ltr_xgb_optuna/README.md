# XGBoost LambdaMART Optuna tuning

Цель этапа - проверить, насколько можно улучшить существующий full-ranking
baseline `ltr_xgb_002` только за счет подбора гиперпараметров XGBoost
LambdaMART.

На этом этапе не меняются:

- feature set;
- Protocol B split;
- train candidate generation;
- validation/test semantics;
- negative seed;
- full-ranking evaluation protocol.

## База

Используется только `ltr_xgb_002`.

- Training candidates: `sampled_100`, 1 positive + 100 negatives per query.
- Validation objective: full-ranking over `7111` items.
- Test during tuning: locked, not loaded, not evaluated.
- Objective: validation full-ranking `NDCG@10`.
- Model seed: `42`.
- Optuna sampler seed: `2026`.

`ltr_xgb_001` не используется как tuning base, потому что там была sampled-100
evaluation и абсолютные HR/NDCG не сопоставимы с full-ranking sequential
baselines.

## Файлы

- `config.yaml` - пути, protocol constants, study storage и tuning policy.
- `search_space.yaml` - v1 search space.
- `optuna_search.py` - validation-only Optuna search/smoke runner.
- `run_best.py` - защищенная заготовка будущего final run; test требует явного
  `--allow-test-evaluation`.
- `AUDIT.md` - аудит baseline, search space и test policy.
- `environment.txt` - cluster environment after Optuna install.
- `runs/` - compact JSON/notes для smoke.

## Smoke

Smoke запускает ровно один trial на полном train и full-ranking validation:

```bash
sbatch slurm/ltr_xgb_optuna.sh
```

Ожидаемые compact artifacts:

- `runs/optuna_smoke_001.json`
- `runs/optuna_smoke_001_notes.md`

Smoke не является final result и не записывается в `experiments/results.csv`.

Фактический smoke:

- Slurm job: `4271164`;
- partition/constraint: `rocky/type_d`;
- CPU: `8`;
- status: `COMPLETED`, exit code `0:0`;
- trial number: `0`;
- validation `NDCG@10=0.014951`;
- validation `HR@10=0.030771`;
- best boosted rounds: `20`;
- test evaluation count: `0`.

## Study

- Study name: `ltr_xgb_optuna_v1`.
- Storage: `/home/daryumin/iberdov/diplom/experiments/ltr_xgb_optuna/optuna.db`.
- Storage backend: SQLite.
- Parallel trials: disabled for this stage.

SQLite DB, trial models, cache and rankings are runtime artifacts and ignored by
Git.
