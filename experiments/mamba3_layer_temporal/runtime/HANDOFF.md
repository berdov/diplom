# Layer temporal functions: job4372822

**COMPLETE.** Slurm COMPLETED 0:0, cn-045, 3 октября 2026, 13:51:11–14:30:01 MSK,
00:38:50. Завершены 2/2 fresh fits и одна paired seed2026 пара. TEST=0.
[Snapshot](status_4372822.json), [статус](handoff.json),
[terminal audit](../evidence/job4372822/independent_audit.json).

Execution `30640e2be36b45c6f89e31f91aae83f76f2d2254`, source
`ad71e91b271db416cf05717e03d7429c60d39657216d1cf18275465ed1500e5b`.
381 source files совпали с Git blobs execution commit. CPU 28/28 и no-Git 28/28,
GPU 6/214, synthetic smoke 3 шага каждого варианта: PASS. Унаследованные kernel
checks 45/2342 проверены по SHA. Все 79 эпох / 948 VALID metric cells совпали с
RecBole и process logs; early stopping, best/last-tie selection, first27,
checkpoint metadata и повторно посчитанные SHA/bytes проверены.

Shared control точно воспроизвёл исторический MIMO dual seed2026, включая loss,
все epoch metrics и checkpoint SHA. Common initialization, temporal copies,
RNG, first batch, data, precision и optimizer совпали внутри пары.

Shared/layer-specific NDCG@10: 0.0633 / 0.0625, Δ −0.0008 (−1.2638%).
First27: 0.0620 / 0.0619. HR@10: 0.1162 / 0.1168. Отдельные функции разошлись,
но не улучшили primary metric. Пункт 4 проверен в pilot scope; оставить MIMO dual
fixed-reference с shared temporal functions. Confirmation не рекомендована.

Raw evidence сохранено в [39 компактных файлах](../evidence/job4372822/preservation_manifest.json),
1 991 469 bytes. Веса остались на кластере; сохранены paths, bytes и SHA256.
Терминальный аудит использует только сохранённые записи, без model forward и
загрузки весов. Для float64 diagnostic reductions учтено округление Python/Torch:
140 отличающихся вычислений, максимум 8.88e−16; science metrics и GPU policy не менялись.
Нормы параметров проверены на алгебраическую согласованность, не пересчитаны из весов.

Публикация: после preservation commit добавить RESULTS.md и ровно две строки
реестра (120→122), затем publication commit и merge/push main.
Новые scheduler polls, sbatch, retry, continuation, confirmation и TEST не нужны.
Пункт 5, статья и Overleaf остаются вне выполненной работы. Фонового процесса нет.
