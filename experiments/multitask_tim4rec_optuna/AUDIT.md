# Аудит Optuna search для Multitask TiM4Rec

## Почему tuning нужен

`multitask_tim4rec_001` показал validation NDCG@10 `0.0580` и test NDCG@10
`0.0581`, тогда как базовый `tim4rec_001` имеет validation NDCG@10 `0.0593`
и test NDCG@10 `0.0598`. Auxiliary tasks обучаются, но fixed multitask loss
дал легкий negative transfer в ranking.

## Гипотеза

Negative transfer может быть связан с одинаковыми task weights, слишком
агрессивным `pos_weight` для редких задач, неверным общим `lambda_aux`, разным
масштабом gradients между ranking и auxiliary losses, а также общим learning
rate для backbone и маленьких linear heads.

## Что тюним

- `lambda_aux` для всего auxiliary блока.
- Нормированные task weights: `w_click`, `w_long_view`, `w_like`,
  `w_profile`.
- Imbalance exponents: `alpha_common` для `is_click`/`long_view` и
  `alpha_rare` для `is_like`/`is_profile_enter`.
- `learning_rate`.
- `weight_decay`.
- Существующий `dropout_prob`.
- `head_lr_multiplier`, где `head_lr = learning_rate * head_lr_multiplier`.

## Что не тюним

Не меняются Protocol B, split, identity hash, task set, TiM4Rec backbone,
hidden size, SSD layers, sequence length, head architecture, recommendation
loss, full-ranking evaluation, attention, Flow Matching, MoE, adaptive loss,
gradient surgery, GradNorm и uncertainty weighting.

## Политика test

Optuna search не загружает test dataset и не создает test dataloader. Для
search физически создается отдельный RecBole dataset только из train и
validation parquet-файлов. Study objective, pruning, model selection и logs
используют только full-ranking validation NDCG@10.

`test_evaluation_count = 0`.
