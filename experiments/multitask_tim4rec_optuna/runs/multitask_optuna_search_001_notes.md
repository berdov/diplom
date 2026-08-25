# multitask_optuna_search_001

## Test safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Objective/pruning/best trial selection используют только full-ranking validation NDCG@10.

## Study

- Optuna: `4.9.0`.
- Study: `multitask_tim4rec_optuna_v1`.
- Sampler: `TPESampler(seed=2026)`.
- Pruner: `MedianPruner`.
- State counts: `{'RUNNING': 0, 'COMPLETE': 60, 'PRUNED': 61, 'FAIL': 0, 'WAITING': 0}`.

## Best trial

- Trial: `110`.
- Best epoch: `16`.
- NDCG@10: `0.059900`.
- HR@10: `0.109300`.

## Top 10

| rank | trial | NDCG@10 | HR@10 | best_epoch | lambda_aux | lr | weight_decay | dropout | head_lr_mult |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 110 | 0.059900 | 0.109300 | 16 | 0.131827 | 0.00190754 | 1.9255e-06 | 0.0857 | 0.4585 |
| 2 | 102 | 0.059700 | 0.110900 | 24 | 0.0738637 | 0.00181951 | 7.21595e-06 | 0.0876 | 3.7688 |
| 3 | 55 | 0.059300 | 0.109800 | 16 | 0.0136313 | 0.00224794 | 1.23824e-05 | 0.1263 | 0.2801 |
| 4 | 80 | 0.059200 | 0.107900 | 16 | 0.00958798 | 0.00133141 | 3.53091e-06 | 0.1015 | 2.4083 |
| 5 | 41 | 0.059100 | 0.107800 | 14 | 0.0487263 | 0.00263582 | 3.48211e-07 | 0.0378 | 0.4346 |
| 6 | 32 | 0.059000 | 0.108500 | 16 | 0.0664743 | 0.0021214 | 7.60598e-06 | 0.0802 | 0.4456 |
| 7 | 48 | 0.059000 | 0.109700 | 15 | 0.171922 | 0.00264623 | 2.73024e-06 | 0.0863 | 0.2653 |
| 8 | 84 | 0.059000 | 0.107600 | 6 | 0.0108656 | 0.00136268 | 2.51473e-06 | 0.0194 | 1.2924 |
| 9 | 14 | 0.058900 | 0.107500 | 16 | 0.0643181 | 0.00125544 | 7.03647e-05 | 0.1947 | 0.2780 |
| 10 | 4 | 0.058800 | 0.108100 | 6 | 0.0593696 | 0.00244562 | 3.6205e-06 | 0.0092 | 0.4987 |

## Validation comparison

| model | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TiM4Rec | 0.1086 | 0.1740 | 0.3126 | 0.0593 | 0.0757 | 0.1030 |
| MultitaskTiM4Rec fixed | 0.1061 | 0.1733 | 0.3115 | 0.0580 | 0.0749 | 0.1022 |
| MultitaskTiM4Rec tuned | 0.1093 | 0.1722 | 0.3136 | 0.0599 | 0.0757 | 0.1036 |

## Negative transfer

- Status: `removed`.
- Tuned delta vs TiM4Rec validation NDCG@10: `0.000600`.
- Tuned delta vs fixed validation NDCG@10: `0.001900`.

## Parameter importance

```json
{
  "alpha_common": 0.0455321524934,
  "alpha_rare": 0.03465892187285427,
  "dropout_prob": 0.019336198997458424,
  "head_lr_multiplier": 0.012038294896127129,
  "lambda_aux": 0.029948600430453158,
  "learning_rate": 0.01303809141856768,
  "w_click_raw": 0.011613028385845965,
  "w_like_raw": 0.05559560575509398,
  "w_long_view_raw": 0.0188767962249949,
  "w_profile_raw": 0.009248450917115836,
  "weight_decay": 0.7501138586080884
}
```
