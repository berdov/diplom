# XGBoost LambdaMART baseline

Экспериментальный baseline для KuaiRand Protocol B: Learning-to-Rank на XGBoost с objective `rank:ndcg`.

## ltr_xgb_001

`ltr_xgb_001` сохранён как sanity/exploratory experiment, но это не Protocol B full-ranking benchmark.

- `evaluation_protocol`: `B_split_sampled_100_candidates`;
- training candidate protocol: `sampled_100`;
- validation/test candidate protocol: `sampled_100`;
- query candidates: 1 positive + 100 deterministic unseen negatives;
- метрики не сопоставимы с опубликованными SSD4Rec/TiM4Rec, где используется full-ranking evaluation.

Почему абсолютные метрики высокие: при 101 candidate Random HR@10 теоретически около `10 / 101 = 0.099010`; фактический Random HR@10 в `ltr_xgb_001` был около `0.1005` на validation и `0.1002` на test. Это подтверждает, что высокая величина метрик объясняется sampled candidate evaluation, а не качеством модели относительно SOTA.

## ltr_xgb_002

`ltr_xgb_002` исправляет evaluation protocol:

- training candidate protocol остаётся `sampled_100`;
- validation/test evaluation строится full-ranking по item universe Protocol B (`7111` real items);
- sampled negatives для validation/test не используются;
- сохраняется только Top-50 на пользователя, полный массив `23951 x 7111` scores не пишется на диск;
- semantics ориентирована на RecBole sequential `mode: full`: validation context = train history, test context = train + validation interaction, seen history items не маскируются, internal padding item 0 маскируется только в RecBole internal id space.

## Feature design

Общий дизайн:

- query/group: `user_id`;
- один query на пользователя для training;
- training positive: последний item в train history;
- training context: train history без последнего train item;
- validation positive: validation item, context только из train;
- test positive: test item, context из train + validation;
- в `ltr_xgb_001` target item исключался из context, если уже встречался раньше; в `ltr_xgb_002` это сохранено только для train, а validation/test повторяют RecBole sequential semantics и оставляют repeated target в history;
- train negatives: 100 deterministic unseen items из item universe Protocol B;
- seed: `42`;
- raw `user_id` и `item_id` не используются как числовые признаки модели.

Features намеренно простые: длины и разнообразие history, popularity statistics, простая item popularity, ratio/difference относительно пользовательской history popularity. Feedback candidate interaction и validation/test leakage не используются.

Цель baseline: получить корректный end-to-end ranking pipeline и reference lines `Random`/`MostPopular`, а не оптимизировать качество.
