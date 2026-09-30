# Head timescales: pilot seed2026

KuaiRand: хронологический leave-one-out, полный каталог. Только VALID; TEST не запускался.

| Variant | Parameters | VALID NDCG@10 | HR@10 | Best epoch | Epochs | First27 complete | TRAIN / VALID s | Peak allocated / reserved bytes | Status |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| fixed | 715020 | 0.0633 | 0.1162 | 27 | 39 | True | 936.4568233509781 / 51.39838225778658 | 3024827392 / 4076863488 | PASS |
| shared_tau | 715022 | 0.0627 | 0.1165 | 27 | 39 | True | 834.5197746772319 / 17.162839455530047 | 3028313088 / 4078960640 | PASS |
| head_tau | 715024 | 0.0639 | 0.1191 | 48 | 60 | True | 1304.0622136719758 / 26.41856360703241 | 3058805760 / 4112515072 | PASS |

| Contrast | Δ VALID NDCG@10 | Relative % | Status |
|---|---:|---:|---|
| head_tau - shared_tau | 0.0011999999999999927 | 1.9138755980861126 | COMPLETE |
| shared_tau - fixed | -0.0005999999999999894 | -0.9478672985781824 | COMPLETE |
| head_tau - fixed | 0.0006000000000000033 | 0.9478672985782044 | COMPLETE |

Один seed не устанавливает устойчивость. First27 только при полном окне0–26; early stopping до27 не отменяет успешный fit. Время включает JIT/cache эффекты. R — глобальные параметры, не доказанные периоды интересов.
