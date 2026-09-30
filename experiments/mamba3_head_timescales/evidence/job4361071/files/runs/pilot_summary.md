# Head timescales: pilot seed2026

KuaiRand: хронологический leave-one-out, полный каталог. Только VALID; TEST не запускался.

| Variant | Parameters | VALID NDCG@10 | HR@10 | Best epoch | Epochs | First27 complete | TRAIN / VALID s | Peak allocated / reserved bytes | Status |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| fixed | 715020 | — | — | — | 0 | False | — / — | — / — | NOT_RUN |
| shared_tau | 715022 | — | — | — | 0 | False | — / — | — / — | NOT_RUN |
| head_tau | 715024 | — | — | — | 0 | False | — / — | — / — | NOT_RUN |

| Contrast | Δ VALID NDCG@10 | Relative % | Status |
|---|---:|---:|---|
| head_tau - shared_tau | — | — | NOT_AVAILABLE |
| shared_tau - fixed | — | — | NOT_AVAILABLE |
| head_tau - fixed | — | — | NOT_AVAILABLE |

Один seed не устанавливает устойчивость. First27 только при полном окне0–26; early stopping до27 не отменяет успешный fit. Время включает JIT/cache эффекты. R — глобальные параметры, не доказанные периоды интересов.
