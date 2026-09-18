# Контекстная временная калибровка Mamba3

Один seed, exploratory. Replay не независимый seed; без significance/CI и сравнения с внешними paper TEST.

| mode | seed | parameters | VALID NDCG@10 | HR@10 | delta replay | best epoch (0-based) | epochs | TRAIN/VALID sec | status |
|---|---|---|---|---|---|---|---|---|---|
| separate_replay | 2026 | 610572 | 0.0633 | 0.1176 | 0.0000 | 51 | 63 | 1091/20 | PASS |
| dense11 | 2026 | 611214 | 0.0614 | 0.1124 | -0.0019 | 15 | 27 | 622/8 | PASS |
| dense12 | 2026 | 611284 | 0.0635 | 0.1177 | 0.0002 | 67 | 79 | 1284/24 | PASS |
| uniform | 2026 | 610968 | 0.0628 | 0.1163 | -0.0005 | 54 | 66 | 1182/20 | PASS |
| routed | 2026 | 611232 | 0.0628 | 0.1164 | -0.0005 | 35 | 47 | 924/16 | PASS |

| mode | первые 27 эпох | полный запуск |
|---|---|---|
| separate_replay | 0.0614 | 0.0633 |
| dense11 | 0.0614 | 0.0614 |
| dense12 | 0.0619 | 0.0635 |
| uniform | 0.0613 | 0.0628 |
| routed | 0.0620 | 0.0628 |
| historical separate2026 | 0.0614 | 0.0633 |

Польза MoE относительно dense controls не показана.
Польза адаптивного выбора экспертов не подтверждена.

TEST=0. Численные различия не доказывают семантические режимы или новизну.

[JSON и hashes](pilot_summary.json)

- [mamba3_context_separate_replay_seed2026_001](mamba3_context_separate_replay_seed2026_001.json), SHA256 `481acff6f71194bf006b094892ac04ccb9b8ea3601de3d4fe50815abc77d41b1`
- [mamba3_context_dense11_seed2026_001](mamba3_context_dense11_seed2026_001.json), SHA256 `2febb220c6965cac8e5461e69370ddce79b69368d1904b74f9988c2ee69c1454`
- [mamba3_context_dense12_seed2026_001](mamba3_context_dense12_seed2026_001.json), SHA256 `df1cc25cc1cb7664a3e6c8696ae0594db23a190cbd2176f9f0cd48aa4db6b38d`
- [mamba3_context_uniform_seed2026_001](mamba3_context_uniform_seed2026_001.json), SHA256 `033d68f28a4ac842d8287182eaa538eeb2a95fd4781f40e04c5cad345eb89d19`
- [mamba3_context_routed_seed2026_001](mamba3_context_routed_seed2026_001.json), SHA256 `c0b554b75ad0e65f93793bae1942daded067a14cf3294771f3596932a5a4ed57`

![VALID NDCG@10](pilot_summary.svg)
