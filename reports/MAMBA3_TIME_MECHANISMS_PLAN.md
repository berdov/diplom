# Mamba3 temporal mechanisms: исторический план

Ниже сохранён план до запусков; его будущие формулировки не описывают текущий статус.
Завершённые VALID и ограничения: [результаты](MAMBA3_TIME_MECHANISMS_RESULTS.md).
Полные equations, pinned upstream audit и numerical contracts:
[experiment README](../experiments/mamba3_time_mechanisms/README.md).

Гипотеза: conditioning native decay и scan/input-weight dynamics одним physical
clock может отличаться от двух независимых learnable mappings одного history gap.
Пять modes: vanilla, decay_only, scan_only, shared, separate. Calibrators=0/1/1/1/2,
каждый shared между двумя layers; additional parameters=0/66/66/66/132.
Все остальные training/model/data settings frozen. Zero-init сохраняет vanilla.

Primary endpoint: full-ranking Protocol B VALID NDCG@10, seed2026, selection
только по VALID. Существующие references: vanilla 0.0584, shared 0.0605.
Новые результаты неизвестны. Shared не переобучается. TEST не используется.

До scientific launch: GPU Suite A (vanilla identity) и Suite B (полный shared RT
forward/loss/gradients), eval/train L50/L64, atol1e-6/rtol1e-5, без fake CPU PASS.
Continuous temporal diagnostics не влияют на loss или selection. Интерпретация
ограничена одним observed seed, без causal или significance claim.
