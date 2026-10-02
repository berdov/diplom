# Gap Trap: pilot seed2026

KuaiRand: хронологический leave-one-out, полный каталог. Только VALID; TEST не запускался.

| Variant | Parameters | VALID NDCG@10 | HR@10 | Best epoch | Epochs | First27 NDCG / complete | TRAIN / VALID s | Peak allocated / reserved bytes | Status |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| fixed_replay | 715020 | 0.0633 | 0.1162 | 27 | 39 | 0.062 / True | 934.2671438558027 / 51.98042598110624 | 3024827392 / 4076863488 | PASS |
| gap_trap | 715021 | 0.0626 | 0.1167 | 45 | 57 | 0.0612 / True | 1343.7683879846008 / 27.01334846415557 | 3025238528 / 4076863488 | PASS |

| Contrast | Δ VALID NDCG@10 | Relative % | Status |
|---|---:|---:|---|
| gap_trap - fixed_replay | -0.0006999999999999923 | -1.1058451816745534 | COMPLETE |

Один seed не устанавливает устойчивость. First27 только при полном окне0–26; early stopping до27 не отменяет успешный fit. Время включает JIT/cache эффекты. α — один общий коэффициент поправки Trap; результат не доказывает устойчивый эффект.
