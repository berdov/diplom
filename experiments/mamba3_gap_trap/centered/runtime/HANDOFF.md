# Centered Gap-Trap: attempt002, job4371876

Исправление выполнено, повторный job отправлен по явному запросу после сбоя.
На 2 октября 2026, 23:11:48 MSK: **PENDING (Priority)**. Оценка старта Slurm:
3 октября, 01:13:34 MSK; она может измениться. Node не назначен, elapsed0,
scientific fits0/2, TEST0. GPU gate и smoke ещё не начинались.

- Canonical local: `/Users/berdov/diplom`.
- Canonical cluster: `/home/daryumin/iberdov/diplom`, SSH `hse-karizma`.
- Ветка: `exp/mamba3-gap-trap-centered`.
- Execution commit: `e4e31370e29c2a34c5f0f1071046fccebd77c289`.
- Source hash: `638b836780af7b6f9065546c128cb8ac2451c5d8af3581855029eea515c2ca10`.
- Manifest: `source_manifest_002.json`, 333 файла; plan: `study_plan_002.json`.
- Launcher: `slurm/mamba3_gap_trap_centered_002.sh`.
- Runtime paths: `slurm_logs/attempt_002/` и `runs/attempt_002/`.
- Exact-source CPU/no-Git: оба21/21 PASS, без skips/errors/failures, CUDA false.
- GPU: прежние6cases/158leaves, затем synthetic smoke2x3steps, затем свежие
  fixed_replay и centered_gap_trap seed2026 в раздельных процессах.
- Ресурсы прежние: rocky/proj_1833/type_e, A100x1, CPU4, mem0,6h,no-requeue.

[Scheduler snapshot](status_4371876.json),
[submission evidence](../evidence/job4371876/submission_preservation.json).
Cluster checkout остаётся на executione4e3137. Более поздние commits содержат
только сохранение и post-run tooling. Пока job PENDING/RUNNING, checkout не менять.

## Исправление и прошлая попытка

В `state.transfer_common` адаптер теперь применяется только к target;
source передаётся напрямую в проверенный parent transfer. Исторической модели
достаточно state_dict, поле gap_trap_mode ей не требуется.
Новый CPU-тест использует настоящий ThreeTimeMamba3Rec и оба целевых варианта;
проверяет точность переноса изменённых backbone/calibrator weights и alpha0.
Отрицательные fixtures проверяют missing/extra keys, неверные shape/dtype
и отсутствие частичной записи при отказе. Ещё проверяются gate binding,
attempt002 paths/plan bindings и сохранение preflight RUNNING/FAIL evidence.

Attempt001/job4371302: FAILED1:0 на gate setup, scientific fits0, TEST0.
[Полный разбор](../evidence/job4371302/failure_audit.json),
[исходные22файла и SHA](../evidence/job4371302/preservation_manifest.json).
Они проверены повторно на кластере и не изменились. Старые plan, manifest,
launcher и execution4347ecb сохранены. Формула и scientific protocol не менялись.

## Следующие действия

Проверять только существующий job4371876. Новый sbatch/requeue/continuation
не разрешён: запросом разрешена одна повторная отправка, она уже выполнена.
Максимум2scientific fits суммарно; предыдущая попытка не начала ни одного.
При technical failure сохранить факты и остановиться.

После terminal state сохранить exact sacct, все компактные stage/run/metadata
JSON, logs и locks в `evidence/job4371876/files/` с путями относительно centered.
Создать preservation_manifest.json (`checkout_commit`, `files` path/bytes/SHA,
`checkpoints` relative path→SHA/bytes); SHA checkpoint читать streaming,
weights не загружать. Scheduler terminal сохранить отдельно с job_id/state/
exit_code/raw_sacct/временем. Все файлы на кластере сохраняются без удаления.

Если оба fit PASS, выполнить подготовленный stdlib-аудит:

    python3 -m experiments.mamba3_gap_trap.centered.runtime.audit_saved --job 4371876

Аудитор теперь берёт attempt paths и названия stage files из config. Он ещё
не выполнялся для текущего job: terminal artifacts пока отсутствуют. Проверяет
source/preservation, ownership, gate/smoke, epoch metrics/loss из двух logs,
selection, checkpoint hashes, exact historical fixed replay, pairing, alpha,
BF16 diagnostics, timings/memory, summary и scheduler terminal status.

При полном успешном выполнении опубликовать результат независимо от выигрыша:
raw runs/summary/concise RESULTS.md; anchor #gap-trap-centered-pilot в
reports/MAMBA3_TIME_MECHANISMS_RESULTS.md, краткую строку reports/RESULTS.md,
ровно2scientific rows в registry с сохранением старых bytes. Commit/push/merge
в main разрешены исходным заданием. При неполных fits строки не добавлять.

Registry сейчас110rows, побайтно как Phase A main f533a49038a5a055a19837bb2dfde4518f10e3f4.
One-sided опубликован ранее; старая exp/mamba3-gap-trap остаётся на824ebbe.
Пункт3 закрывается после фактического centered-результата в текущем KuaiRand/VALID
scope. Третьих Trap-функций и автоматической confirmation не будет. TEST,
пункт4/5, статья/Overleaf запрещены. Layer-specific temporal functions — только
рекомендация отдельного будущего задания. Сообщения другим людям не отправлять.
Фонового процесса мониторинга нет.
