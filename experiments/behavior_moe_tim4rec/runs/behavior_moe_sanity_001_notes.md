# Behavior-MoE TiM4Rec sanity 001

## Цель

Проверить 5-epoch trajectory для plain Behavior-MoE без load balancing, Optuna, изменения backbone и доступа к test.

## Архитектура

- Experts: `interest, consumption, positive, shared`.
- Router: `separate learned Linear(hidden, 4) router head per task; softmax(logits / temperature)`.
- Residual: `h_task = h + residual_scale * sum_e p(task,e|h) * expert_e(h)`.
- Load balancing: `False`.
- Residual scale: `0.1` fixed.

## Данные

- Protocol B identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.
- Train rows: `1086518`.
- Validation rows: `23951`.
- Test rows в validation-only benchmark: `0`.

## Обучение

- Epochs: `5`.
- Train batches: `519`.
- Batch size: `2048`.
- Best epoch by NDCG@10: `4`.
- Epoch 5 NDCG@10: `0.054600`.

| epoch | total | ranking | click | long_view | like | profile |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.4832 | 7.2294 | 0.6844 | 0.7084 | 0.4463 | 0.5135 |
| 2 | 6.8099 | 6.5723 | 0.6576 | 0.6817 | 0.4140 | 0.4941 |
| 3 | 6.6492 | 6.4189 | 0.6541 | 0.6786 | 0.3977 | 0.4909 |
| 4 | 6.5505 | 6.3284 | 0.6530 | 0.6774 | 0.3788 | 0.4896 |
| 5 | 6.4786 | 6.2638 | 0.6521 | 0.6767 | 0.3620 | 0.4888 |

## Validation trajectory

| epoch | HR@5 | HR@10 | HR@20 | HR@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0543 | 0.0867 | 0.1347 | 0.2423 | 0.0543 | 0.0867 | 0.1347 | 0.2423 | 0.0378 | 0.0482 | 0.0603 | 0.0815 |
| 2 | 0.0590 | 0.0954 | 0.1540 | 0.2670 | 0.0590 | 0.0954 | 0.1540 | 0.2670 | 0.0408 | 0.0525 | 0.0672 | 0.0895 |
| 3 | 0.0603 | 0.0987 | 0.1601 | 0.2863 | 0.0603 | 0.0987 | 0.1601 | 0.2863 | 0.0422 | 0.0546 | 0.0700 | 0.0949 |
| 4 | 0.0628 | 0.1027 | 0.1646 | 0.2962 | 0.0628 | 0.1027 | 0.1646 | 0.2962 | 0.0434 | 0.0562 | 0.0717 | 0.0976 |
| 5 | 0.0608 | 0.0992 | 0.1657 | 0.2984 | 0.0608 | 0.0992 | 0.1657 | 0.2984 | 0.0423 | 0.0546 | 0.0713 | 0.0974 |

## Auxiliary tasks

| epoch | task | ROC-AUC | PR-AUC | BCE | positive rate |
|---:|---|---:|---:|---:|---:|
| 1 | `is_click` | 0.6704 | 0.6326 | 0.6478 | 0.4826 |
| 1 | `long_view` | 0.6696 | 0.5062 | 0.6149 | 0.3574 |
| 1 | `is_like` | 0.7427 | 0.0653 | 0.1545 | 0.0196 |
| 1 | `is_profile_enter` | 0.6679 | 0.0417 | 0.1808 | 0.0237 |
| 2 | `is_click` | 0.6783 | 0.6419 | 0.6442 | 0.4826 |
| 2 | `long_view` | 0.6774 | 0.5150 | 0.6077 | 0.3574 |
| 2 | `is_like` | 0.7600 | 0.0739 | 0.1689 | 0.0196 |
| 2 | `is_profile_enter` | 0.6841 | 0.0435 | 0.1791 | 0.0237 |
| 3 | `is_click` | 0.6808 | 0.6439 | 0.6414 | 0.4826 |
| 3 | `long_view` | 0.6797 | 0.5175 | 0.6073 | 0.3574 |
| 3 | `is_like` | 0.7770 | 0.0911 | 0.1605 | 0.0196 |
| 3 | `is_profile_enter` | 0.6851 | 0.0447 | 0.1783 | 0.0237 |
| 4 | `is_click` | 0.6825 | 0.6473 | 0.6409 | 0.4826 |
| 4 | `long_view` | 0.6801 | 0.5195 | 0.6062 | 0.3574 |
| 4 | `is_like` | 0.7966 | 0.1213 | 0.1280 | 0.0196 |
| 4 | `is_profile_enter` | 0.6872 | 0.0454 | 0.1722 | 0.0237 |
| 5 | `is_click` | 0.6824 | 0.6471 | 0.6412 | 0.4826 |
| 5 | `long_view` | 0.6788 | 0.5168 | 0.6072 | 0.3574 |
| 5 | `is_like` | 0.8056 | 0.1346 | 0.1446 | 0.0196 |
| 5 | `is_profile_enter` | 0.6902 | 0.0469 | 0.1707 | 0.0237 |

## Routing trajectory

Epoch 1:

| task | interest | consumption | positive | shared |
|---|---:|---:|---:|---:|
| ranking | 0.0411 | 0.2997 | 0.6107 | 0.0485 |
| click | 0.8365 | 0.1260 | 0.0220 | 0.0154 |
| long_view | 0.1046 | 0.3023 | 0.5117 | 0.0815 |
| like | 0.0428 | 0.5583 | 0.0265 | 0.3724 |
| profile | 0.0582 | 0.3175 | 0.4312 | 0.1930 |

Epoch 2:

| task | interest | consumption | positive | shared |
|---|---:|---:|---:|---:|
| ranking | 0.0279 | 0.2190 | 0.7183 | 0.0348 |
| click | 0.4244 | 0.2164 | 0.3393 | 0.0199 |
| long_view | 0.1244 | 0.2838 | 0.4910 | 0.1007 |
| like | 0.0333 | 0.5832 | 0.0303 | 0.3532 |
| profile | 0.0474 | 0.3913 | 0.4040 | 0.1573 |

Epoch 3:

| task | interest | consumption | positive | shared |
|---|---:|---:|---:|---:|
| ranking | 0.0223 | 0.1922 | 0.7611 | 0.0244 |
| click | 0.3252 | 0.1053 | 0.5462 | 0.0234 |
| long_view | 0.1316 | 0.2474 | 0.5490 | 0.0720 |
| like | 0.0378 | 0.6886 | 0.0254 | 0.2482 |
| profile | 0.0225 | 0.4084 | 0.4438 | 0.1254 |

Epoch 4:

| task | interest | consumption | positive | shared |
|---|---:|---:|---:|---:|
| ranking | 0.0164 | 0.1841 | 0.7821 | 0.0175 |
| click | 0.2263 | 0.0842 | 0.6581 | 0.0314 |
| long_view | 0.1334 | 0.2564 | 0.5374 | 0.0728 |
| like | 0.0415 | 0.6786 | 0.0336 | 0.2463 |
| profile | 0.0254 | 0.3804 | 0.4913 | 0.1029 |

Epoch 5:

| task | interest | consumption | positive | shared |
|---|---:|---:|---:|---:|
| ranking | 0.0154 | 0.1610 | 0.8065 | 0.0172 |
| click | 0.1291 | 0.0979 | 0.7344 | 0.0386 |
| long_view | 0.1021 | 0.2073 | 0.6061 | 0.0845 |
| like | 0.0353 | 0.7166 | 0.0294 | 0.2187 |
| profile | 0.0288 | 0.3257 | 0.4453 | 0.2002 |

## Entropy

| epoch | ranking | click | long_view | like | profile |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5281 | 0.2316 | 0.5364 | 0.4448 | 0.6041 |
| 2 | 0.4374 | 0.4856 | 0.5413 | 0.4379 | 0.5655 |
| 3 | 0.3814 | 0.4651 | 0.5303 | 0.4147 | 0.4940 |
| 4 | 0.3431 | 0.4145 | 0.5393 | 0.4510 | 0.5312 |
| 5 | 0.3095 | 0.3822 | 0.5374 | 0.4159 | 0.5599 |

## Expert utilization

| epoch | interest | consumption | positive | shared | collapse | dead expert | shared domination |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 0.2166 | 0.3208 | 0.3204 | 0.1422 | `False` | `False` | `False` |
| 2 | 0.1315 | 0.3387 | 0.3966 | 0.1332 | `False` | `False` | `False` |
| 3 | 0.1079 | 0.3284 | 0.4651 | 0.0987 | `False` | `False` | `False` |
| 4 | 0.0886 | 0.3167 | 0.5005 | 0.0942 | `False` | `False` | `False` |
| 5 | 0.0621 | 0.3017 | 0.5243 | 0.1118 | `False` | `False` | `False` |

## Specialization

- Strongest final pair: `click` vs `like`, L1 `1.597595`.
- Specialization L1 trend: `{'1': 0.9367660166074833, '2': 0.7713187330712875, '3': 0.726647841433684, '4': 0.7264373265206814, '5': 0.6816341572751602}`.
- Semantic matches at epoch 5: `{'click': False, 'long_view': False, 'like': False, 'profile': True}`.

## Gradient diagnostics

| epoch | all experts | all routers | expert max norm | router max norm |
|---:|---|---|---:|---:|
| 1 | `True` | `True` | 0.018038 | 0.118311 |
| 3 | `True` | `True` | 0.022781 | 0.093404 |
| 5 | `True` | `True` | 0.024094 | 0.095815 |

## Comparison with TiM4Rec / Multitask

| run | type | epoch | HR@10 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| `tim4rec_sanity_001` | 5_epoch_sanity_reference | 5 | 0.1002 | 0.0553 | 0.0706 | 0.0958 |
| `multitask_tim4rec_sanity_001` | 5_epoch_sanity_reference | 5 | 0.1011 | 0.0557 | 0.0706 | 0.0959 |
| `behavior_moe_sanity_001` | 5_epoch_sanity | 5 | 0.0992 | 0.0546 | 0.0713 | 0.0974 |
| `multitask_tim4rec_tuned_001` | full_budget_validation_reference | 16 | 0.1069 | 0.0589 | 0.0765 | 0.1046 |

## Cost

- Mean train epoch: `55.067` sec.
- Mean validation: `1.436` sec.
- Peak VRAM: `1956182528` bytes.
- Process MaxRSS: `5030056` KB.
- Slurm batch MaxRSS: `3050872K`.
- Slurm elapsed: `00:09:35`.
- Params overhead: `34580` (`5.82%`).

## Risks

- Collapse: `False`.
- Dead expert: `False`.
- Shared domination: `False`.
- NDCG@10 падение против multitask sanity epoch 5: `-0.001100`.

## Decision

5-epoch sanity завершён, но есть предупреждения; full run лучше не запускать до анализа trajectory.
