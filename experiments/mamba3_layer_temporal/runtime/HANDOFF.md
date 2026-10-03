# Layer temporal functions: job4372822

На **3 октября 2026, 13:49:35 MSK**: **PENDING (Priority)**, node не назначен,
elapsed 00:00:00. Pipeline, gate и smoke ещё не начались. Scientific fits **0/2**,
полных пар **0/1**, TEST = 0. Следующий scheduler poll не раньше **13:59:35 MSK**.
Планировщик показал ориентировочный старт 14:45 MSK; это не гарантия.
[Точный snapshot](status_4372822.json), [структурированный статус](handoff.json).

## Зафиксированный эксперимент

- Canonical local `/Users/berdov/diplom`; cluster `/home/daryumin/iberdov/diplom`, SSH `hse-karizma`.
- Ветка `exp/mamba3-layer-temporal-functions` от main `0afbba24c9402886483d7f3cc2680222264052d9`.
- Execution `30640e2be36b45c6f89e31f91aae83f76f2d2254`.
- Source `ad71e91b271db416cf05717e03d7429c60d39657216d1cf18275465ed1500e5b`, 381 файл.
- [Submission/reservation/login/CPU evidence](../evidence/job4372822/submission_preservation.json).
- CPU 28/28 и no-Git 28/28 PASS, без пропусков/ошибок, CUDA не инициализирована.
- Фактические counts: shared_layers 715020, layer_specific 715152.
- Один job: rocky/proj_1833/type_e, A100 x1, CPU4, mem0, 6 часов, no-requeue.

Контроль буквально делегирует encode_sequence историческому ThreeTimeMamba3Rec.
Treatment сохраняет times для layer0 и добавляет layer1_times=deepcopy(times).
Независимые параметры, одинаковая инициализация, отсутствие дополнительных RNG draws.
Reference 838393 мс, dual decay/scan; write/phase используют один scan tensor в каждом слое.
Kernel, precision, R0 и training settings не менялись. Пункт 5 и статья вне задания.

## Следующие действия

Проверять **только существующий job4372822** не чаще примерно раз в 10 минут.
`python3 experiments/mamba3_layer_temporal/runtime/capture_status.py` выполняет один
явный read-only poll, сохраняет точные squeue/sacct/pipeline snapshots локально.
Никакого фонового процесса нет.

Порядок в allocation:
provenance → GPU gate (6 cases / 214 checks) → smoke (3 Adam steps каждого режима,
batch2048/history50) → shared_layers seed2026 → exact historical replay →
layer_specific seed2026 → summary → terminal evidence.
Старые MIMO kernel checks 45/2342 наследуются по точным SHA.
До scientific fits нового GPU PASS пока нет; CPU surrogate forward не заменяет GPU gate.

Не делать повторный sbatch/requeue, continuation или новые seeds. Даже при technical
failure до TRAIN автоматический retry запрещён. Сохранить logs/traceback/partial
records и сообщить, сколько fits реально началось. При повреждённом result pipeline
counter может отставать: сверять result/lock/process logs, неизвестное не считать нулём.
При неполной паре не делать научного вывода и не подставлять historical число.

Не менять cluster checkout во время job: он остаётся на execution30640e2. После
submission ветка может содержать commits с evidence; вычислительный pipeline проверяет
содержимое source_manifest и сохранённые ownership bindings, не текущее origin HEAD.
Все прежние untracked cluster artifacts сохранены; 12 прошлых result/pair JSON
приняты в Git только после byte-for-byte сверки с опубликованными blobs.

## Terminal audit и публикация

После terminal Slurm state сохранить compact artifacts:
`python3 experiments/mamba3_layer_temporal/runtime/preserve_terminal.py 4372822 30640e2be36b45c6f89e31f91aae83f76f2d2254`.
Скрипт сохраняет raw JSON/log/out/err/lock, plan/source manifest и scheduler;
checkpoint читает только потоково для SHA/bytes. Веса остаются на кластере.
Сам terminal audit ещё предстоит; helper не объявляет результат PASS.

Проверить ownership execution/source/reservation/job у gate/smoke/runs/summary,
все обязательные leaves, 2/2 полных fits, TEST0. Все epochs/12 VALID metrics/loss
сопоставить с обоими логами; last-tie/early-stop/first27, metadata и checkpoint SHA.
Контроль должен точно повторить MIMO dual seed2026, в том числе historical SHA
`cba3da6aa7daf883cf9bcddb1b92540d48c26c62d797f8c9a1b602bcea292db8`.
Внутри пары initial common state, обе temporal copies, RNG, first consumed batch,
config/data/precision/Adam должны совпасть. Разница timing/memory допустима.

Диагностика best checkpoint записана в best_diagnostics.layer_temporal: grid из
study_plan, decay/scan H0/H1 каждого слоя, mean/max abs log-ratio, parameter L2 и
relative L2 с norm(layer0) в знаменателе (null при нуле), доли <.51/>1.99.
Не загружать датасет для дополнительных forwards. Разные curves не доказывают
short/long-term назначение слоёв. First27 отсутствует, если actual_epochs < 27.

Публиковать только после полного PASS независимо от знака эффекта: preservation
commit → RESULTS.md и section #layer-temporal-functions-pilot → ровно 2 новые строки
реестра с прежним byte-identical prefix (сейчас 120, ожидается 122) → publication
commit → merge main/push. Статью/Overleaf не менять. При выигрыше только рекомендовать
confirmation2027–2030, не запускать её. Подготовить короткий текст сообщения в чате,
нижним регистром, не отправлять. Полный A–I ответ по исходному заданию — после результата.
