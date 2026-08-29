# Длинный скрининг сходимости: 8 MOO families

Отчет основан только на validation-only артефактах `experiments/moo_8families/runs/*_convergence_001.json`. Test split не использовался.

## A. GIT

- Ветка: `exp/moo-8families-benchmark`.
- Start HEAD из задания: `0577dd6ed878ccc059fe13bab5e26ad2092ccfdf`.
- HEAD, на котором обучались cluster jobs и записаны JSON: `40a748b6a4f75a041d82951c61739c7f2d2f78bf`.
- HEAD перед results commit: `40a748b6a4f75a041d82951c61739c7f2d2f78bf`. Итоговый hash results commit указан в `git log` после коммита и в финальном сообщении.
- Commits подготовки:
  - `b15032429d10165bda771f92b83fbd0fdf8930d7 Add MOO convergence screening support`
  - `40a748b6a4f75a041d82951c61739c7f2d2f78bf Pass git metadata to MOO Slurm jobs`
- Results commit содержит компактные JSON/notes, `summary.csv`, `experiments/results.csv`, report и PNG-кривые. Merge в `main` не выполнялся.

## B. ENVIRONMENT

- Python: `3.10.14`, executable `/home/daryumin/iberdov/diplom/envs/tim4rec/bin/python`.
- PyTorch/CUDA: `2.3.0+cu118` / CUDA `11.8`; RecBole `1.2.0`; NumPy `1.26.4`.
- GPU: `NVIDIA A100-SXM4-80GB`, capability `8.0`, `device_count=1`. Все current runs шли на A100 nodes `cn-045`/`cn-046`.
- Dataset: KuaiRand Protocol B, users=23951, items=7111, interactions=1134420, train=1086518, validation=23951, test=23951.
- Fingerprint: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`; validation-only loader: train batches=519, valid batches=6.

## C. JOB TABLE

| method | Slurm job | node | GPU | epochs | Slurm runtime | peak VRAM GB | status | exit |
|---|---:|---|---|---:|---:|---:|---|---:|
| STCH | 4290972 | cn-046 | NVIDIA A100-SXM4-80GB | 95 | 00:48:42 | 2.254 | completed | 0:0 |
| FAMO | 4290973 | cn-046 | NVIDIA A100-SXM4-80GB | 30 | 00:21:22 | 2.254 | completed | 0:0 |
| PCGrad | 4290974 | cn-046 | NVIDIA A100-SXM4-80GB | 40 | 01:12:16 | 2.254 | completed | 0:0 |
| EPO | 4290975 | cn-045 | NVIDIA A100-SXM4-80GB | 30 | 05:04:39 | 2.265 | completed | 0:0 |
| HV-Gradient / GradHV-style | 4290976 | cn-046 | NVIDIA A100-SXM4-80GB | 65 | 01:28:27 | 4.886 | completed | 0:0 |
| PHN-adapter | 4290977 | cn-046 | NVIDIA A100-SXM4-80GB | 75 | 00:40:47 | 2.255 | completed | 0:0 |
| COSMOS-style | 4290978 | cn-046 | NVIDIA A100-SXM4-80GB | 40 | 00:24:55 | 2.256 | completed | 0:0 |
| PaLoRA | 4290979 | cn-046 | NVIDIA A100-SXM4-80GB | 50 | 00:30:49 | 2.260 | completed | 0:0 |

## D. EARLY STOPPING TABLE

| method | best epoch | stop epoch | validation checks | best NDCG@10 | early stopped | stop reason |
|---|---:|---:|---:|---:|---|---|
| STCH | 80 | 95 | 19 | 0.0424 | True | `early_stopping_patience` |
| FAMO | 15 | 30 | 6 | 0.0412 | True | `early_stopping_patience` |
| PCGrad | 25 | 40 | 8 | 0.0444 | True | `early_stopping_patience` |
| EPO | 15 | 30 | 6 | 0.0584 | True | `early_stopping_patience` |
| HV-Gradient / GradHV-style | 50 | 65 | 13 | 0.0486 | True | `early_stopping_patience` |
| PHN-adapter | 60 | 75 | 15 | 0.0423 | True | `early_stopping_patience` |
| COSMOS-style | 25 | 40 | 8 | 0.0453 | True | `early_stopping_patience` |
| PaLoRA | 35 | 50 | 10 | 0.0422 | True | `early_stopping_patience` |

## E. LEARNING CURVES

Primary validation NDCG@10 по фактическим validation checks. После early stopping значения не продлеваются.

| method | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 | 65 | 70 | 75 | 80 | 85 | 90 | 95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STCH | 0.0333 | 0.0386 | 0.0399 | 0.0399 | 0.0396 | 0.0401 | 0.0407 | 0.0404 | 0.0403 | 0.0412 | 0.0413 | 0.0415 | 0.0410 | 0.0417 | 0.0407 | 0.0424 | 0.0411 | 0.0412 | 0.0420 |
| FAMO | 0.0375 | 0.0411 | 0.0412 | 0.0410 | 0.0409 | 0.0411 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PCGrad | 0.0419 | 0.0423 | 0.0432 | 0.0430 | 0.0444 | 0.0437 | 0.0443 | 0.0444 |  |  |  |  |  |  |  |  |  |  |  |
| EPO | 0.0568 | 0.0574 | 0.0584 | 0.0584 | 0.0579 | 0.0573 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| HV-Gradient / GradHV-style | 0.0444 | 0.0458 | 0.0467 | 0.0461 | 0.0468 | 0.0464 | 0.0469 | 0.0478 | 0.0465 | 0.0486 | 0.0466 | 0.0468 | 0.0478 |  |  |  |  |  |  |
| PHN-adapter | 0.0350 | 0.0380 | 0.0392 | 0.0398 | 0.0403 | 0.0406 | 0.0406 | 0.0406 | 0.0410 | 0.0416 | 0.0400 | 0.0423 | 0.0418 | 0.0423 | 0.0422 |  |  |  |  |
| COSMOS-style | 0.0419 | 0.0434 | 0.0449 | 0.0452 | 0.0453 | 0.0450 | 0.0451 | 0.0453 |  |  |  |  |  |  |  |  |  |  |  |
| PaLoRA | 0.0351 | 0.0393 | 0.0385 | 0.0405 | 0.0418 | 0.0406 | 0.0422 | 0.0402 | 0.0420 | 0.0421 |  |  |  |  |  |  |  |  |  |

Кривые сохранены: `figures/convergence_validation_ndcg10.png` и `figures/convergence_train_scalar.png`.

## F. MAIN RANKING TABLE

Control для delta: `multitask_tim4rec_tuned_001`, validation NDCG@10=0.0589. Для HR@10 delta использован единственный registry HR@10=0.1071 из control row; control не переобучался.

| family | method | best | stop | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | dNDCG@10 | dHR@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| loss_balancing | STCH | 80 | 95 | 0.0749 | 0.1163 | 0.2082 | 0.0424 | 0.0528 | 0.0709 | -0.0165 | -0.0322 |
| gradient_weighting | FAMO | 15 | 30 | 0.0719 | 0.1102 | 0.1935 | 0.0412 | 0.0508 | 0.0672 | -0.0177 | -0.0352 |
| gradient_manipulation | PCGrad | 25 | 40 | 0.0790 | 0.1259 | 0.2253 | 0.0444 | 0.0562 | 0.0757 | -0.0145 | -0.0281 |
| finite_preference_set | EPO | 15 | 30 | 0.1078 | 0.1767 | 0.3171 | 0.0584 | 0.0756 | 0.1033 | -0.0005 | 0.0007 |
| finite_no_preference_set | HV-Gradient / GradHV-style | 50 | 65 | 0.0874 | 0.1382 | 0.2440 | 0.0486 | 0.0613 | 0.0820 | -0.0103 | -0.0197 |
| infinite_hypernetwork | PHN-adapter | 60 | 75 | 0.0746 | 0.1155 | 0.2027 | 0.0423 | 0.0526 | 0.0698 | -0.0166 | -0.0325 |
| infinite_preference_conditioned | COSMOS-style | 25 | 40 | 0.0810 | 0.1257 | 0.2252 | 0.0453 | 0.0565 | 0.0761 | -0.0136 | -0.0261 |
| infinite_model_combination | PaLoRA | 35 | 50 | 0.0750 | 0.1159 | 0.2080 | 0.0422 | 0.0525 | 0.0706 | -0.0167 | -0.0321 |

## G. PARETO TABLE

Evaluation reference frozen: `[1.0, 2.0, 2.0, 2.0, 2.0]` для `[1-NDCG@10, click_BCE, long_view_BCE, like_BCE, profile_BCE]`; `invalid_reference_policy=raise` не ослаблялся.

| method | best epoch | ranking NDCG@10 | balanced NDCG@10 | oracle NDCG@10 | eval HV | non-dominated | spread | points | ranking point | oracle point |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| EPO | 15 | 0.0584 | 0.0350 | 0.0584 | 0.3068 | 4 | 0.3014 | 6 | `rank_heavy` | `rank_heavy` |
| HV-Gradient / GradHV-style | 50 | 0.0486 | 0.0486 | 0.0486 | 0.2997 | 2 | 0.2731 | 3 | `2` | `2` |
| PHN-adapter | 60 | 0.0423 | 0.0424 | 0.0424 | 0.2659 | 2 | 0.0030 | 6 | `rank_heavy` | `balanced` |
| COSMOS-style | 25 | 0.0453 | 0.0451 | 0.0454 | 0.2585 | 5 | 0.0190 | 6 | `rank_heavy` | `click_heavy` |
| PaLoRA | 35 | 0.0422 | 0.0421 | 0.0422 | 0.2598 | 1 |  | 6 | `rank_heavy` | `rank_heavy` |

Reference check: все финальные points имеют `status=valid`; минимальные margins до reference положительные, минимум среди MOO families = 0.0038 у GradHV.

## H. EPO TRADE-OFF TABLE

| preference_id | weights | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE | Pareto dominated |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `balanced` | [0.200, 0.200, 0.200, 0.200, 0.200] | 0.0596 | 0.0350 | 0.7369 | 0.6864 | 0.1693 | 0.2129 | NO |
| `rank_heavy` | [0.600, 0.100, 0.100, 0.100, 0.100] | 0.1078 | 0.0584 | 0.6656 | 0.6700 | 0.4863 | 0.5552 | NO |
| `click_heavy` | [0.200, 0.500, 0.100, 0.100, 0.100] | 0.0404 | 0.0232 | 0.7970 | 0.6424 | 0.1770 | 0.2120 | YES |
| `long_heavy` | [0.200, 0.100, 0.500, 0.100, 0.100] | 0.0409 | 0.0229 | 0.6610 | 0.7538 | 0.1988 | 0.1992 | NO |
| `like_heavy` | [0.200, 0.100, 0.100, 0.500, 0.100] | 0.0624 | 0.0317 | 0.6636 | 0.6255 | 0.1336 | 0.1877 | NO |
| `profile_heavy` | [0.200, 0.100, 0.100, 0.100, 0.500] | 0.0386 | 0.0209 | 0.6727 | 0.6336 | 0.1591 | 0.1915 | YES |

Вывод: EPO дает лучший ranking почти на уровне tuned control, но trade-off не монотонный: `rank_heavy` силен в ranking, `like_heavy` лучше по like BCE, часть preference points доминируется.

## I. FAMO WEIGHT DYNAMICS

| epoch | rank | click | long_view | like | profile_enter |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2331 | 0.2200 | 0.2141 | 0.1281 | 0.2048 |
| 5 | 0.2570 | 0.2351 | 0.2435 | 0.1159 | 0.1485 |
| 10 | 0.2866 | 0.2458 | 0.2412 | 0.0949 | 0.1315 |
| 15 | 0.2829 | 0.2534 | 0.2520 | 0.0865 | 0.1251 |
| 20 | 0.2794 | 0.2498 | 0.2598 | 0.0940 | 0.1170 |
| 25 | 0.2802 | 0.2486 | 0.2516 | 0.1006 | 0.1190 |
| 30 | 0.2691 | 0.2492 | 0.2539 | 0.0990 | 0.1288 |

Динамика FAMO не collapsed: rank/click/long_view стабилизировались около 0.25-0.29, like/profile получили меньший вес. Ranking plateau наступил рано: best epoch 15, stop epoch 30.

## J. PCGRAD CURRENT VS HISTORICAL

| run | stage | best epoch | stop | GPU | HR@10 | NDCG@10 | note |
|---|---|---:|---:|---|---:|---:|---|
| historical PCGrad | historical | 9 |  |  | 0.1082 | 0.0586 | старый reference, не используется для family decision |
| current PCGrad | convergence_screening | 25 | 40 | NVIDIA A100-SXM4-80GB | 0.0790 | 0.0444 | сопоставимый current convergence run |

Historical PCGrad был значительно лучше current run (0.0586 vs 0.0444 NDCG@10), поэтому его нельзя считать comparable replacement для текущего benchmark.

## K. GRADHV TRAIN-VS-VALIDATION DYNAMICS

| epoch | train HV | validation HV | non-dominated | ranking NDCG@10 |
|---:|---:|---:|---:|---:|
| 5 | 0.3161 | 0.2897 | 1 | 0.0444 |
| 10 | 0.4818 | 0.2959 | 2 | 0.0458 |
| 20 | 0.6541 | 0.2865 | 2 | 0.0461 |
| 30 | 0.7322 | 0.2883 | 2 | 0.0464 |
| 40 | 0.7581 | 0.2979 | 2 | 0.0478 |
| 50 | 0.8169 | 0.2997 | 2 | 0.0486 |
| 60 | 0.8132 | 0.2875 | 2 | 0.0468 |
| 65 | 0.8849 | 0.2914 | 2 | 0.0478 |

Train HV рос почти монотонно, но validation HV и ranking двигались слабее и шумнее; лучший ranking на 50-й эпохе, после чего early stopping остановил на 65-й.

## L. PHN/COSMOS/PALORA PREFERENCE DYNAMICS

Preference sensitivity здесь посчитана как `max(NDCG@10)-min(NDCG@10)` среди fixed validation preferences на checkpoint.

### PHN-adapter

| epoch | rank-heavy NDCG@10 | balanced NDCG@10 | spread | non-dominated | sensitivity |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.0350 | 0.0349 |  | 1 | 0.0002 |
| 10 | 0.0380 | 0.0379 | 0.0020 | 4 | 0.0002 |
| 15 | 0.0392 | 0.0390 |  | 1 | 0.0002 |
| 20 | 0.0398 | 0.0399 | 0.0017 | 5 | 0.0001 |
| 25 | 0.0403 | 0.0404 | 0.0015 | 5 | 0.0001 |
| 30 | 0.0406 | 0.0405 |  | 1 | 0.0001 |
| 35 | 0.0406 | 0.0404 |  | 1 | 0.0003 |
| 40 | 0.0406 | 0.0404 | 0.0011 | 4 | 0.0002 |
| 45 | 0.0410 | 0.0410 |  | 1 | 0.0000 |
| 50 | 0.0416 | 0.0416 | 0.0018 | 3 | 0.0001 |
| 55 | 0.0400 | 0.0401 | 0.0021 | 4 | 0.0001 |
| 60 | 0.0423 | 0.0424 | 0.0030 | 2 | 0.0001 |
| 65 | 0.0418 | 0.0418 |  | 1 | 0.0000 |
| 70 | 0.0423 | 0.0423 | 0.0022 | 2 | 0.0001 |
| 75 | 0.0422 | 0.0425 | 0.0028 | 2 | 0.0003 |

### COSMOS-style

| epoch | rank-heavy NDCG@10 | balanced NDCG@10 | spread | non-dominated | sensitivity |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.0419 | 0.0419 |  | 1 | 0.0000 |
| 10 | 0.0434 | 0.0430 |  | 1 | 0.0013 |
| 15 | 0.0449 | 0.0450 | 0.0147 | 5 | 0.0001 |
| 20 | 0.0452 | 0.0449 | 0.3578 | 3 | 0.0004 |
| 25 | 0.0453 | 0.0451 | 0.0190 | 5 | 0.0008 |
| 30 | 0.0450 | 0.0447 | 0.1993 | 5 | 0.0015 |
| 35 | 0.0451 | 0.0449 | 0.2551 | 6 | 0.0014 |
| 40 | 0.0453 | 0.0453 | 0.2963 | 6 | 0.0011 |

### PaLoRA

| epoch | rank-heavy NDCG@10 | balanced NDCG@10 | spread | non-dominated | sensitivity |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.0351 | 0.0350 | 0.0006 | 6 | 0.0001 |
| 10 | 0.0393 | 0.0394 | 0.0005 | 5 | 0.0002 |
| 15 | 0.0385 | 0.0388 | 0.0004 | 6 | 0.0003 |
| 20 | 0.0405 | 0.0403 |  | 1 | 0.0002 |
| 25 | 0.0418 | 0.0415 |  | 1 | 0.0003 |
| 30 | 0.0406 | 0.0403 |  | 1 | 0.0004 |
| 35 | 0.0422 | 0.0421 |  | 1 | 0.0001 |
| 40 | 0.0402 | 0.0404 | 0.0013 | 5 | 0.0002 |
| 45 | 0.0420 | 0.0420 |  | 1 | 0.0000 |
| 50 | 0.0421 | 0.0421 | 0.0025 | 2 | 0.0001 |

Вывод: PHN и PaLoRA почти preference-degenerate по ranking (sensitivity около 0.0000-0.0004). COSMOS показывает более заметную preference structure, но ranking sensitivity все равно мала; spread скачет сильнее, чем ranking.

## M. AUXILIARY METRICS

| method | click BCE | long_view BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|
| STCH | 0.7018 | 0.6760 | 0.1421 | 0.2044 |
| FAMO | 0.6444 | 0.6148 | 0.1291 | 0.1750 |
| PCGrad | 0.6685 | 0.6334 | 0.1523 | 0.2074 |
| EPO | 0.6656 | 0.6700 | 0.4863 | 0.5552 |
| HV-Gradient / GradHV-style | 0.6574 | 0.6223 | 0.1441 | 0.2190 |
| PHN-adapter | 0.6636 | 0.6296 | 0.1279 | 0.1705 |
| COSMOS-style | 0.7256 | 0.6845 | 0.1465 | 0.1712 |
| PaLoRA | 0.6621 | 0.6300 | 0.1431 | 0.1913 |

Auxiliary trade-offs не дают единого победителя: EPO выигрывает ranking, но сильно хуже по like/profile BCE на rank-heavy point; FAMO/PHN сильнее по profile/like BCE, но слабее в ranking.

## N. COMPUTE TABLE

| method | GPU | epochs | artifact wall | GPU-hours | peak VRAM GB | params | extra params | models | compute/solution h |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| STCH | NVIDIA A100-SXM4-80GB | 95 | 2872.4 | 0.798 | 2.254 | 593758 | 0 | 1 | 0.798 |
| FAMO | NVIDIA A100-SXM4-80GB | 30 | 1233.0 | 0.343 | 2.254 | 593758 | 0 | 1 | 0.343 |
| PCGrad | NVIDIA A100-SXM4-80GB | 40 | 4295.7 | 1.193 | 2.254 | 593758 | 0 | 1 | 1.193 |
| EPO | NVIDIA A100-SXM4-80GB | 30 | 18209.9 | 5.058 | 2.265 | 3562548 | 0 | 6 | 0.843 |
| HV-Gradient / GradHV-style | NVIDIA A100-SXM4-80GB | 65 | 5264.4 | 1.462 | 4.886 | 1781274 | 0 | 3 | 0.487 |
| PHN-adapter | NVIDIA A100-SXM4-80GB | 75 | 2403.1 | 0.668 | 2.255 | 602462 | 8704 | 1 | 0.668 |
| COSMOS-style | NVIDIA A100-SXM4-80GB | 40 | 1449.9 | 0.403 | 2.256 | 602398 | 8640 | 1 | 0.403 |
| PaLoRA | NVIDIA A100-SXM4-80GB | 50 | 1807.9 | 0.502 | 2.260 | 596318 | 2560 | 1 | 0.502 |

EPO самый дорогой: 6 independently trained preference solutions и около 5.06 artifact GPU-hours. Conditional methods используют одну shared model и дешевле, но пока слабее по ranking/Pareto controllability.

## O. COMPARISON WITH 5-EPOCH SANITY

| method | sanity NDCG@10 | long best NDCG@10 | delta | conclusion changed? |
|---|---:|---:|---:|---|
| STCH | 0.0321 | 0.0424 | 0.0103 | да: долгий run заметно улучшил раннюю оценку |
| FAMO | 0.0377 | 0.0412 | 0.0035 | частично: улучшение есть, но порядок family почти не изменился |
| PCGrad |  | 0.0444 |  | current 5-epoch sanity отсутствует; сравнение с historical вынесено в секцию J |
| EPO | 0.0577 | 0.0584 | 0.0007 | нет: уже на sanity был почти максимум |
| HV-Gradient / GradHV-style | 0.0452 | 0.0486 | 0.0034 | частично: улучшение есть, но порядок family почти не изменился |
| PHN-adapter | 0.0358 | 0.0423 | 0.0065 | да: долгий run заметно улучшил раннюю оценку |
| COSMOS-style | 0.0423 | 0.0453 | 0.0030 | частично: улучшение есть, но порядок family почти не изменился |
| PaLoRA | 0.0352 | 0.0422 | 0.0070 | да: долгий run заметно улучшил раннюю оценку |

Главное изменение после long screening: STCH/PHN/PaLoRA нельзя оценивать по 5 эпохам, они существенно добирают качество позже. EPO остался лидером уже с раннего этапа; GradHV подтвердил второе место, но требует больше compute.

## P. FAMILY CLASSIFICATION

| family | method | class | reason |
|---|---|---|---|
| loss_balancing | STCH | BORDERLINE | простая и дешевая, но ranking заметно ниже control; долгий run улучшает, однако novelty ограничена |
| gradient_weighting | FAMO | DROP | быстрый plateau, слабый ranking; веса стабилизируются без выигрыша для ranking |
| gradient_manipulation | PCGrad | BORDERLINE | current comparable run средний; historical result не репрезентативен для текущего протокола |
| finite_preference_set | EPO | PROMISING | лучший ranking, близко к tuned control, понятная finite preference structure; главный минус compute |
| finite_no_preference_set | HV-Gradient / GradHV-style | PROMISING | второй ranking и неплохой Pareto HV, но validation перенос train HV шумный и VRAM выше |
| infinite_hypernetwork | PHN-adapter | DROP | долго учится, но почти preference-degenerate и ranking ниже COSMOS/PCGrad |
| infinite_preference_conditioned | COSMOS-style | PROMISING | лучший conditional result, умеренный compute, есть признаки preference structure |
| infinite_model_combination | PaLoRA | BORDERLINE | дешевый adapter/combination path и late improvement, но preference sensitivity мала и ranking невысокий |

## Q. TOP CANDIDATES FOR TUNING

| method | why | what to tune later | main risk |
|---|---|---|---|
| EPO | лучший validation ranking, close to control | preference weights/grid, solver/fallback details, compute budget allocation | дорогой и может overfit rank-heavy |
| GradHV | сильный ranking/Pareto baseline без explicit preference vector | reference/training HV temperature or finite solution setup | train HV не всегда переносится в validation |
| COSMOS | лучший conditional family и реальная, хотя слабая, preference variation | conditioning strength, Dirichlet alpha, rank-heavy sampling mix | preference effect может остаться декоративным |
| PCGrad | важен как gradient-manipulation baseline; current run хуже historical, нужен controlled check | projection schedule/mode only in later controlled stage | tuning может просто восстановить historical artifact без novelty |

Controlled tuning сейчас не запускался.

## R. SAFETY

- `test_evaluation_count = 0` во всех 8 current artifacts.
- `test_safety.test_evaluated = false` во всех 8 current artifacts.
- Optuna/trials/manual tuning на этом этапе: `0`.
- Test contamination: `NO`.
- Current jobs launched for final results: `8`; первая metadata-партия была отменена и не использовалась в результатах.

## S. NEXT STEP

Следующий этап должен быть отдельным controlled tuning protocol только для EPO, GradHV, COSMOS и PCGrad: frozen Protocol B, validation-only search, одинаковая A100 hardware policy, заранее заданный search space, no test, отдельные run IDs и отдельный commit. После выбора best validation configs можно планировать финальный locked test run один раз.
