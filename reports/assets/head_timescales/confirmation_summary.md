# Head-timescales confirmation

Status: PASS; new fits 12/12; TEST=0.

| Seed | Variant | Status | NDCG@10 | HR@10 | Best epoch | Epochs | First27 complete | First27 |
|---|---|---|---:|---:|---:|---:|---|---:|
| 2026 | fixed | PASS | 0.0633 | 0.1162 | 27 | 39 | True | 0.062 |
| 2026 | shared_tau | PASS | 0.0627 | 0.1165 | 27 | 39 | True | 0.0617 |
| 2026 | head_tau | PASS | 0.0639 | 0.1191 | 48 | 60 | True | 0.0621 |
| 2027 | fixed | PASS | 0.0625 | 0.1147 | 17 | 29 | True | 0.0625 |
| 2027 | shared_tau | PASS | 0.0628 | 0.1161 | 37 | 49 | True | 0.0624 |
| 2027 | head_tau | PASS | 0.0635 | 0.1179 | 28 | 40 | True | 0.063 |
| 2028 | fixed | PASS | 0.0627 | 0.1169 | 46 | 58 | True | 0.061 |
| 2028 | shared_tau | PASS | 0.0629 | 0.1167 | 29 | 41 | True | 0.0622 |
| 2028 | head_tau | PASS | 0.063 | 0.1175 | 37 | 49 | True | 0.0618 |
| 2029 | fixed | PASS | 0.0641 | 0.1183 | 87 | 99 | True | 0.0615 |
| 2029 | shared_tau | PASS | 0.0628 | 0.1172 | 33 | 45 | True | 0.0623 |
| 2029 | head_tau | PASS | 0.0633 | 0.1159 | 56 | 68 | True | 0.0618 |
| 2030 | fixed | PASS | 0.0624 | 0.115 | 28 | 40 | True | 0.0622 |
| 2030 | shared_tau | PASS | 0.0625 | 0.1159 | 31 | 43 | True | 0.0617 |
| 2030 | head_tau | PASS | 0.062 | 0.1154 | 23 | 35 | True | 0.062 |

## new4_full

| Variant | Available / expected | Seeds | Mean ± sample std |
|---|---:|---|---:|
| fixed | 4/4 | [2027, 2028, 2029, 2030] | 0.062925 ± 0.000793 |
| shared_tau | 4/4 | [2027, 2028, 2029, 2030] | 0.062750 ± 0.000173 |
| head_tau | 4/4 | [2027, 2028, 2029, 2030] | 0.062950 ± 0.000666 |

| Paired contrast | n / expected | Seeds | Δ mean ± std | + / − / 0 | Relative % |
|---|---:|---|---:|---:|---:|
| head_tau-shared_tau | 4/4 | [2027, 2028, 2029, 2030] | 0.000200 ± 0.000529 | 3/1/0 | 0.3187250996016022 |
| head_tau-fixed | 4/4 | [2027, 2028, 2029, 2030] | 0.000025 ± 0.000793 | 2/2/0 | 0.03972983710767153 |
| shared_tau-fixed | 4/4 | [2027, 2028, 2029, 2030] | -0.000175 ± 0.000754 | 3/1/0 | -0.2781088597536896 |

## new4_first27

| Variant | Available / expected | Seeds | Mean ± sample std |
|---|---:|---|---:|
| fixed | 4/4 | [2027, 2028, 2029, 2030] | 0.061800 ± 0.000678 |
| shared_tau | 4/4 | [2027, 2028, 2029, 2030] | 0.062150 ± 0.000311 |
| head_tau | 4/4 | [2027, 2028, 2029, 2030] | 0.062150 ± 0.000574 |

| Paired contrast | n / expected | Seeds | Δ mean ± std | + / − / 0 | Relative % |
|---|---:|---|---:|---:|---:|
| head_tau-shared_tau | 4/4 | [2027, 2028, 2029, 2030] | 0.000000 ± 0.000535 | 2/2/0 | 0.0 |
| head_tau-fixed | 4/4 | [2027, 2028, 2029, 2030] | 0.000350 ± 0.000420 | 3/1/0 | 0.5663430420711935 |
| shared_tau-fixed | 4/4 | [2027, 2028, 2029, 2030] | 0.000350 ± 0.000785 | 2/2/0 | 0.5663430420711935 |

## all5_full

| Variant | Available / expected | Seeds | Mean ± sample std |
|---|---:|---|---:|
| fixed | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.063000 ± 0.000707 |
| shared_tau | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.062740 ± 0.000152 |
| head_tau | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.063140 ± 0.000716 |

| Paired contrast | n / expected | Seeds | Δ mean ± std | + / − / 0 | Relative % |
|---|---:|---|---:|---:|---:|
| head_tau-shared_tau | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.000400 ± 0.000640 | 4/1/0 | 0.6375518010838244 |
| head_tau-fixed | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.000140 ± 0.000733 | 3/2/0 | 0.22222222222223476 |
| shared_tau-fixed | 5/5 | [2026, 2027, 2028, 2029, 2030] | -0.000260 ± 0.000680 | 3/2/0 | -0.412698412698409 |

## all5_first27

| Variant | Available / expected | Seeds | Mean ± sample std |
|---|---:|---|---:|
| fixed | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.061840 ± 0.000594 |
| shared_tau | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.062060 ± 0.000336 |
| head_tau | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.062140 ± 0.000498 |

| Paired contrast | n / expected | Seeds | Δ mean ± std | + / − / 0 | Relative % |
|---|---:|---|---:|---:|---:|
| head_tau-shared_tau | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.000080 ± 0.000497 | 3/2/0 | 0.1289075088624081 |
| head_tau-fixed | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.000300 ± 0.000381 | 4/1/0 | 0.48512289780078266 |
| shared_tau-fixed | 5/5 | [2026, 2027, 2028, 2029, 2030] | 0.000220 ± 0.000740 | 2/3/0 | 0.35575679172057395 |

First27 uses only complete observed windows0–26, with independent pair subsets. Pilot2026 is exploratory. No significance, personalization or head-specialization claim. R denotes global normalization scales. VALID is not compared to published TEST.
