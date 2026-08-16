# XGBoost LambdaMART baseline

Первый экспериментальный baseline для KuaiRand Protocol B: Learning-to-Rank на XGBoost с objective `rank:ndcg`.

Дизайн:

- query/group: `user_id`;
- один query на пользователя для training;
- training positive: последний item в train history;
- training context: train history без последнего train item;
- validation positive: validation item, context только из train;
- test positive: test item, context из train + validation;
- если target item уже встречался в context, он исключается из context для этого query, чтобы target не присутствовал в history;
- negatives: 100 deterministic unseen items из item universe Protocol B;
- seed: `42`;
- raw `user_id` и `item_id` не используются как числовые признаки модели.

Features намеренно простые: длины и разнообразие history, popularity statistics, простая item popularity, ratio/difference относительно пользовательской history popularity. Feedback candidate interaction и validation/test leakage не используются.

Цель baseline: получить корректный end-to-end ranking pipeline и reference lines `Random`/`MostPopular`, а не оптимизировать качество.
