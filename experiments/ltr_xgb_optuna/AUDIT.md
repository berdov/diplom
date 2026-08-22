# Аудит XGBoost LambdaMART Optuna tuning

## Цель этапа

Проверить, сколько качества можно получить из той же табличной постановки
`ltr_xgb_002`, если менять только гиперпараметры XGBoost LambdaMART.

Этот этап не является финальным результатом. Он готовит инфраструктуру и
проверяет ее одним smoke trial на полном train/validation. Test закрыт.

## Исходный baseline

Используется `ltr_xgb_002`, потому что это корректный XGBoost LambdaMART
baseline с full-ranking evaluation.

`ltr_xgb_001` не используется как база: в нем validation/test считались на
sampled-100 candidate sets, из-за чего абсолютные HR/NDCG завышены и не
сопоставимы с sequential full-ranking baselines.

Текущие test-метрики `ltr_xgb_002`:

| metric | value |
| --- | ---: |
| HR@10 / Recall@10 | 0.031397 |
| HR@20 / Recall@20 | 0.055697 |
| HR@50 / Recall@50 | 0.099912 |
| NDCG@10 | 0.015003 |
| NDCG@20 | 0.020946 |
| NDCG@50 | 0.029735 |

Ориентиры из текущего `experiments/results.csv`:

| run | model | HR@10 | NDCG@10 |
| --- | --- | ---: | ---: |
| `mostpop_002` | MostPopular | 0.029519 | 0.016677 |
| `ltr_xgb_002` | XGBoost LambdaMART | 0.031397 | 0.015003 |
| `tim4rec_001` | TiM4Rec | 0.105300 | 0.059800 |
| `ssd4rec_001` | SSD4Rec | 0.103200 | 0.057600 |

## Что остается неизменным

- Dataset: KuaiRand Protocol B.
- Fingerprint: `23951 users / 7111 items / 1134420 interactions`.
- Split: `train=1086518`, `validation=23951`, `test=23951`.
- Training candidate protocol: `sampled_100`.
- Candidate generation: 1 positive + 100 negatives per query.
- Negative seed: `42`.
- Query grouping: one query per user.
- XGBoost objective family: LambdaMART, `rank:ndcg`.
- Evaluation objective for tuning: full-ranking validation over all `7111`
  items.
- Masking: `mask_seen_items=false`, matching `ltr_xgb_002` full-ranking
  protocol.

## Feature set

Feature engineering is frozen. The v1 Optuna stage reuses exactly the
leakage-safe feature list from `ltr_xgb_002`:

1. `user_history_length`
2. `user_unique_items`
3. `user_history_unique_ratio`
4. `user_history_popularity_mean`
5. `user_history_popularity_median`
6. `user_history_popularity_max`
7. `user_history_popularity_last`
8. `user_history_span_days`
9. `user_history_mean_gap_hours`
10. `candidate_seen_before`
11. `item_train_popularity`
12. `log1p_item_train_popularity`
13. `item_popularity_rank`
14. `item_popularity_percentile`
15. `candidate_popularity_to_user_mean`
16. `candidate_popularity_minus_user_mean`
17. `candidate_is_more_popular_than_user_mean`

Нельзя добавлять embeddings, item metadata, user metadata, новые interaction
features или side information. Иначе эффект tuning будет смешан с эффектом
feature engineering.

## Candidate generation и grouping

`ltr_xgb_002` строит train/validation/test sampled candidate tables с размером
group `101`: один positive target и 100 fixed negatives. Для обучения Optuna
использует cached `train_features.parquet` и XGBoost group sizes из этих
candidate groups.

Validation objective не использует sampled validation candidates. Для каждой
validation query строятся features against all `7111` items, затем считается
full-ranking HR/Recall/NDCG. Это повторяет full-ranking evaluation `ltr_xgb_002`,
но только для validation split.

## Validation objective

Optuna objective: full-ranking validation `NDCG@10`.

XGBoost sampled `eval_metric=ndcg@10` может оставаться диагностическим параметром
модели, но не используется для выбора trial или best iteration. Best iteration
выбирается внешним loop:

1. обучить очередной блок boosting rounds;
2. посчитать full-ranking validation;
3. обновить best iteration по validation `NDCG@10`;
4. остановиться по patience на full-ranking validation.

Для smoke configured максимум `20` boosting rounds и full validation каждые `10`
rounds. Для будущего search configured максимум `1000` rounds и early stopping.

## Test policy

Во время tuning test нельзя использовать:

- в objective;
- в pruning;
- для выбора hyperparameters;
- для выбора iteration;
- для ручного выбора trial;
- для сравнения trials.

`optuna_search.py` в search/smoke mode загружает только:

- `train_features.parquet`;
- `validation_queries.parquet`;
- `item_train_popularity.parquet`.

`test_features.parquet` и `test_queries.parquet` не входят в allowed input list.
Smoke artifact должен иметь `test_evaluation_count=0`.

## Search space v1

Диапазоны зафиксированы в `search_space.yaml`.

| parameter | range | reason |
| --- | --- | --- |
| `max_depth` | int 3..10 | control tree capacity without changing feature set |
| `eta` | log 0.005..0.2 | learning-rate / boosting step size |
| `min_child_weight` | log 0.1..20 | regularization for leaf splits |
| `subsample` | 0.5..1.0 | row sampling regularization |
| `colsample_bytree` | 0.5..1.0 | feature sampling regularization |
| `reg_lambda` | log 1e-3..100 | L2 regularization |
| `reg_alpha` | log 1e-4..10 | L1 regularization |
| `gamma` | 0 or log 1e-4..10 | split-loss threshold, with explicit zero option |

`num_boost_round` не подбирается напрямую: iteration выбирается по full-ranking
validation with early stopping.

## LTR-specific XGBoost parameters

По official XGBoost Learning to Rank documentation, LambdaMART supports pair
construction controls such as `lambdarank_pair_method` and
`lambdarank_num_pair_per_sample`. Эти параметры меняют способ построения пар в
ranking loss, поэтому v1 search не включает их автоматически. Их можно добавить
отдельным экспериментом после анализа влияния на методологию.

Sources:

- https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html
- https://xgboost.readthedocs.io/en/latest/parameter.html

## Study storage

- Study name: `ltr_xgb_optuna_v1`.
- Storage: SQLite `/home/daryumin/iberdov/diplom/experiments/ltr_xgb_optuna/optuna.db`.
- Sampler: `TPESampler`.
- Sampler seed: `2026`.
- Parallel trials: disabled.

SQLite DB не коммитится в Git. В Git идут только config/search space/code и
compact exports из `runs/`.

## Первый smoke trial

На этом этапе запущен ровно один full-data trial:

- train: cached full `ltr_xgb_002` sampled-100 training features;
- validation: full-ranking over `7111` items;
- test: not loaded, not evaluated;
- Optuna storage: persistent SQLite;
- result: `runs/optuna_smoke_001.json` и `runs/optuna_smoke_001_notes.md`.

Smoke не записывается в `experiments/results.csv`.

Фактический результат `optuna_smoke_001`:

| field | value |
| --- | ---: |
| Slurm job | 4271164 |
| Slurm elapsed | 00:01:38 |
| trial number | 0 |
| best boosted rounds | 20 |
| validation HR@10 | 0.030771 |
| validation NDCG@10 | 0.014951 |
| full validation time, best eval | 41.59 sec |
| XGBoost train time | 2.62 sec |
| test evaluation count | 0 |

Trial `0` не улучшает `ltr_xgb_002` validation `NDCG@10=0.014972`; это нормально
для одного smoke trial и не является выводом о качестве search space. Smoke
подтверждает только корректность Optuna storage, передачу параметров в XGBoost,
full-ranking validation objective и test lock.

## Future full search

Следующий этап после smoke:

1. запустить примерно `40` sequential trials;
2. оставить `n_jobs=1` до проверки стабильности storage/runtime;
3. использовать тот же validation-only objective;
4. после завершения search зафиксировать best hyperparameters отдельным commit;
5. только затем один раз открыть test через `run_best.py --allow-test-evaluation`.

Главный bottleneck ожидаемо не XGBoost training, а repeated full-ranking
validation over `23951 x 7111` user-item scores.
