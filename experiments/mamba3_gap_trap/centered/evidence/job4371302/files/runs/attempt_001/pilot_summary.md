# Centered Gap Trap: pilot seed2026

KuaiRand: хронологический leave-one-out, полный каталог. Только VALID; TEST не запускался.

| Variant | Parameters | VALID NDCG@10 | HR@10 | Best epoch | Epochs | First27 NDCG / complete | TRAIN / VALID s | Peak allocated / reserved bytes | Status |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| fixed_replay | 715020 | — | — | — | 0 | — / False | — / — | — / — | NOT_RUN |
| centered_gap_trap | 715021 | — | — | — | 0 | — / False | — / — | — / — | NOT_RUN |

| Contrast | Δ VALID NDCG@10 | Relative % | Status |
|---|---:|---:|---|
| centered_gap_trap - fixed_replay | — | — | NOT_AVAILABLE |

Один seed не устанавливает устойчивость. First27 только при полном окне0–26; early stopping до27 не отменяет успешный fit. Время включает JIT/cache эффекты. α — один общий коэффициент поправки Trap; результат не доказывает устойчивый эффект.
