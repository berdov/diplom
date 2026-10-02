# Centered Gap Trap: pilot seed2026

KuaiRand: хронологический leave-one-out, полный каталог. Только VALID; TEST не запускался.

| Variant | Parameters | VALID NDCG@10 | HR@10 | Best epoch | Epochs | First27 NDCG / complete | TRAIN / VALID s | Peak allocated / reserved bytes | Status |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| fixed_replay | 715020 | 0.0633 | 0.1162 | 27 | 39 | 0.062 / True | 968.2067418061197 / 62.95107648242265 | 3024827392 / 4076863488 | PASS |
| centered_gap_trap | 715021 | 0.0635 | 0.1189 | 48 | 60 | 0.0615 / True | 1433.5177418654785 / 31.308843017090112 | 3025238528 / 4076863488 | PASS |

| Contrast | Δ VALID NDCG@10 | Relative % | Status |
|---|---:|---:|---|
| centered_gap_trap - fixed_replay | 0.00020000000000000573 | 0.31595576619274207 | COMPLETE |

Один seed не устанавливает устойчивость. First27 только при полном окне0–26; early stopping до27 не отменяет успешный fit. Время включает JIT/cache эффекты. α — один общий коэффициент поправки Trap; результат не доказывает устойчивый эффект.
