# Optuna search 001

## Study

- Study name: `ltr_xgb_optuna_v1`.
- Storage: `/home/daryumin/iberdov/diplom/experiments/ltr_xgb_optuna/optuna.db`.
- COMPLETE / RUNNING / FAIL / PRUNED: `40` / `0` / `1` / `0`.
- Optuna: `4.9.0`.
- XGBoost: `3.2.0`.
- Sampler: `TPESampler`, seed `2026`.

## Slurm

| JobID | JobName | Partition | State | ExitCode | Elapsed | Timelimit | NodeList | AllocCPUS | MaxRSS |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 4271176 | ltr-xgb-optuna-search | rocky | TIMEOUT | 0:0 | 04:00:19 | 04:00:00 | cn-032 | 8 |  |
| 4271176.batch | batch |  | CANCELLED | 0:15 | 04:00:20 |  | cn-032 | 8 | 2094260K |
| 4271176.extern | extern |  | COMPLETED | 0:0 | 04:00:20 |  | cn-032 | 8 | 1.50M |
| 4271211 | ltr-xgb-optuna-resume1 | rocky | COMPLETED | 0:0 | 03:07:12 | 18:00:00 | cn-038 | 8 |  |
| 4271211.batch | batch |  | COMPLETED | 0:0 | 03:07:12 |  | cn-038 | 8 | 2113876K |
| 4271211.extern | extern |  | COMPLETED | 0:0 | 03:07:12 |  | cn-038 | 8 | 1.50M |

## Best trial

- Trial: `16`.
- Best iteration: `9`.
- Boosted rounds: `10`.
- Validation NDCG@10: `0.018407`.
- Validation HR@10: `0.034278`.
- Validation NDCG@20: `0.023980`.
- Validation HR@20: `0.056616`.
- Validation NDCG@50: `0.033404`.
- Validation HR@50: `0.104296`.

## Baseline improvement

| metric | baseline | best | absolute | relative, % |
| --- | ---: | ---: | ---: | ---: |
| NDCG@10 | 0.014972 | 0.018407 | 0.003435 | 22.95 |
| NDCG@20 | 0.021026 | 0.023980 | 0.002954 | 14.05 |
| NDCG@50 | 0.029576 | 0.033404 | 0.003829 | 12.95 |
| HR@10 | 0.030855 | 0.034278 | 0.003424 | 11.10 |
| HR@20 | 0.055572 | 0.056616 | 0.001044 | 1.88 |
| HR@50 | 0.098827 | 0.104296 | 0.005470 | 5.53 |

## Top 10

| rank | trial | NDCG@10 | HR@10 | NDCG@20 | HR@20 | NDCG@50 | HR@50 | best_iter | params |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 16 | 0.018407 | 0.034278 | 0.023980 | 0.056616 | 0.033404 | 0.104296 | 9 | `colsample_bytree=0.9311835757131137, eta=0.11767873104746542, gamma_is_zero=True, max_depth=7, min_child_weight=2.2399194165687106, reg_alpha=0.004848134217357092, reg_lambda=18.413808630043132, subsample=0.8661195199148233` |
| 2 | 22 | 0.018328 | 0.034445 | 0.024007 | 0.057158 | 0.033666 | 0.106092 | 39 | `colsample_bytree=0.8705884746411255, eta=0.12177173665228995, gamma_is_zero=True, max_depth=7, min_child_weight=1.851884955432125, reg_alpha=6.930248895996336, reg_lambda=0.18317729314260534, subsample=0.8573554056474324` |
| 3 | 23 | 0.018322 | 0.034404 | 0.023987 | 0.057117 | 0.033891 | 0.107469 | 89 | `colsample_bytree=0.8890218347485109, eta=0.10767332915015727, gamma_is_zero=True, max_depth=7, min_child_weight=1.821256397060911, reg_alpha=7.6550131482934205, reg_lambda=0.06991800146912854, subsample=0.8199939390458544` |
| 4 | 25 | 0.018268 | 0.034153 | 0.023887 | 0.056699 | 0.033650 | 0.106175 | 39 | `colsample_bytree=0.876955285642678, eta=0.05629303245123538, gamma_is_zero=True, max_depth=8, min_child_weight=7.247400016604517, reg_alpha=6.864042772090141, reg_lambda=0.023558642906982365, subsample=0.913191381243811` |
| 5 | 13 | 0.018251 | 0.033861 | 0.024046 | 0.057033 | 0.033328 | 0.104088 | 29 | `colsample_bytree=0.9993830562627105, eta=0.10309405520655575, gamma_is_zero=True, max_depth=6, min_child_weight=0.41562150225047534, reg_alpha=0.00010458333187130909, reg_lambda=12.100972665435807, subsample=0.7537134988817518` |
| 6 | 17 | 0.018246 | 0.034195 | 0.023932 | 0.056991 | 0.033645 | 0.106175 | 19 | `colsample_bytree=0.8561995507160409, eta=0.1295593359729472, gamma_is_zero=True, max_depth=7, min_child_weight=1.949954100198625, reg_alpha=1.0217368692559072, reg_lambda=0.1609976352916533, subsample=0.8727173393682572` |
| 7 | 12 | 0.018239 | 0.033944 | 0.024058 | 0.057242 | 0.033463 | 0.105090 | 69 | `colsample_bytree=0.9998103829796021, eta=0.0799044866032073, gamma_is_zero=True, max_depth=6, min_child_weight=4.625080731721623, reg_alpha=0.0001296071653106753, reg_lambda=16.382902929877076, subsample=0.7083375241245049` |
| 8 | 39 | 0.018223 | 0.033861 | 0.023932 | 0.056783 | 0.033580 | 0.105591 | 29 | `colsample_bytree=0.8813095480344095, eta=0.017823488270172688, gamma=0.007926042018172626, gamma_is_zero=False, max_depth=7, min_child_weight=6.411003427752319, reg_alpha=0.0445120063370233, reg_lambda=1.0273856969659378, subsample=0.991457856231952` |
| 9 | 38 | 0.018219 | 0.033861 | 0.023990 | 0.056950 | 0.033666 | 0.105966 | 29 | `colsample_bytree=0.7465017582810075, eta=0.09468963082743377, gamma_is_zero=True, max_depth=6, min_child_weight=3.253077472512388, reg_alpha=0.0155472458460268, reg_lambda=0.07550970211422972, subsample=0.9176581694499208` |
| 10 | 11 | 0.018208 | 0.034028 | 0.024011 | 0.057325 | 0.033314 | 0.104714 | 149 | `colsample_bytree=0.983208772116832, eta=0.07789072254064221, gamma_is_zero=True, max_depth=6, min_child_weight=0.12433937701556733, reg_alpha=0.00012239936531028542, reg_lambda=11.8632767262306, subsample=0.6545484579664068` |

## Parameter importance

- Method: `ped_anova`.

| parameter | importance |
| --- | ---: |
| colsample_bytree | 0.328651 |
| max_depth | 0.230660 |
| subsample | 0.199569 |
| min_child_weight | 0.112136 |
| eta | 0.080507 |
| reg_alpha | 0.041764 |
| reg_lambda | 0.006543 |
| gamma_is_zero | 0.000170 |

## Test safety

- Test evaluation count: `0`.
- Forbidden test paths loaded: `[]`.
- Test metrics отсутствуют в study summary.
- `experiments/results.csv` не обновлялся.

## Decision

- Best trial выбран строго по full-ranking validation `NDCG@10`.
- Final test не запускался.
