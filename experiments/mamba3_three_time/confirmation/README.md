# SISO dual/triple: подтверждающая серия

Одна allocation A100, не более 6 часов: проверка источников/данных, один контролируемый TRAIN batch B/C, проверка инициализации, восемь свежих scientific subprocess, сводка. TEST не используется. Низкая VALID-метрика не останавливает серию; техническая ошибка останавливает следующие fits, без retries.

## Зафиксированная процедура

Порядок: dual/triple seed2027, dual/triple seed2028, dual/triple seed2029, dual/triple seed2030. Run IDs перечислены в [study_plan.json](study_plan.json). Seed2026 не обучается и не оценивается повторно.

Модель, kernels, calibrators, dataset/preprocessing, trainer, diagnostics и numerical policy импортируются без изменений из frozen dependencies [первого пилота](../validation_pilot/README.md). Только upstream, SISO chunk64, два слоя и две temporal heads, 610572/610638 параметров, reference838393, bounds[.5,2]. Старые exact-zero FAIL сохраняются; policy v1 не расширяется, MIMO не разрешается.

Настройки читаются из raw `config` пилота, затем весь effective config сверяется с его записью. Разрешены только seed/checkpoint path; режим берётся из соответствующего pilot run. Adam .001, CE, weight_decay=0, batch2048/eval4096, history50, epochs300, eval_step1, stopping_step10, full-ranking VALID NDCG@10. Неизменный trainer сохраняет последний равный максимум, pure state_dict и соответствующие diagnostics. TEST loader не создаётся; reserved dataset удаляется после штатного build.

## Отличия orchestration от validation_pilot/runner.py

- Добавлены явные seed/run/study/path и дополнительные hashes; globals старого runner не меняются, monkeypatch отсутствует.
- Порядок runtime/config -> init_seed/logger -> dataset/build -> TRAIN/VALID loaders -> model -> trainer -> fit сохранён. `prepare()` используется для pre-fit проверки и scientific setup; в trainer/fit вмешательства нет.
- Вместо повторения mathematical attempts используются неизменённые manifests/raw admission: проверяются 277 frozen dependencies, все 28 cases и исторические residuals. Дополнительно проверяются файлы B на execution `6feb832` и C на `6b5618a`, а также пути реально импортированных модулей.
- Отдельный новый one-batch gate: B=`ContextMamba3Rec/separate_replay`, C=`ThreeTimeMamba3Rec/dual`; один и тот же TRAIN batch2048x50, без VALID/TEST loaders. Полный state_dict B строго переносится в C с единственным переименованием `mechanisms.calibrators.* -> times.calibrators.*`. Проверяются все keys/shapes/dtypes/values, включая первые Linear. Это новая контролируемая инициализация, не восстановление исторической B.
- Перед каждым из двух train-mode forward восстанавливаются одинаковые Python/NumPy/CPU/CUDA RNG; dropout не отключается. Один CE backward и один Adam step на модель. Именные проверки encoded/logits/CE, gradients, updates, parameters и optimizer state: atol=1e-6, rtol=1e-5; max/mean absolute, L2 error, reference norm, bitwise equality, failed names сохраняются. Изменять допуски по результату запрещено. PASS одного шага не объясняет окончательно .0633/.0615.
- Gate инициализации не делает forward/fit: повторяет setup для seed2026 и сверяет сохранённые backbone/calibrator/pre-fit RNG hashes; затем проверяет четыре новые пары. В fits сверяются эти hashes и компонентные RNG/DataLoader hashes. First-batch hash берётся неизменным trainer внутри фактического train loop; triple сверяется с dual до первого calculate_loss. Calibrators triple независимы по storage.
- Диагностический процесс завершается до scientific subprocess; его optimizer/модели не передаются в fits. Дополнительная диагностика идёт только в уже существующих VALID passes.

## Артефакты и ограничения

`runs/one_batch_001.json`, `runs/initialization_001.json`; восемь `runs/<run_id>.json`; `slurm_logs/<run_id>/checkpoints/` и logs; `slurm_logs/submission_001.json`, `pipeline_status.json`, exclusive locks. Atomic updates только собственных записей; существующие артефакты блокируют submit. Исходные checkpoint никогда не загружаются.

Внутренний deadline 20700 с оставляет 15 минут до Slurm timeout. Следующий fit не стартует при остатке менее 900 с. Уже начатый fit прерывается по deadline с INCOMPLETE, а не выдаётся за early stopping. Настройки не сокращаются ради времени. На ошибке формируется partial summary и NOT_RUN для оставшихся задач.

`runs/confirmation_summary.json/.md/.svg` заранее определены: новые четыре пары отдельно от всех пяти. Seed2026 помечен exploratory. Историческая separate=.0633 исключена. Для полного горизонта и окна 0-26 показываются все пары, mean/sample std(ddof=1), парные разности, знаки, n_expected/n_available, incomplete. Неполное окно <27 не равно окну27. Все successful rows сохраняются, даже отрицательные; при пропусках парные statistics явно помечены как complete-pair subset, relative gain не вычисляется. Нет статистических заявлений, автоматического выбора модели/seed, смешивания VALID/TEST. Время включает JIT. Сбой SVG не уничтожает числовую сводку.

CPU-проверки: `python -m unittest discover -s experiments/mamba3_three_time/confirmation/tests -v`; `python -m compileall -q experiments/mamba3_three_time/confirmation`; `bash -n slurm/mamba3_three_time_confirmation.sh`. Тест RecBole effective config пропускается локально, если RecBole отсутствует; на кластере обязателен. Установка библиотек не требуется. Login preflight не выполняет model/GPU work. После единственного guarded sbatch сохраняется Job ID; submitter не опрашивает очередь.

## Продолжение attempt003 после job4355052

Parent evidence сохранено побайтно в `evidence/job4355052/`; manifest фиксирует SHA256 47 файлов. Checkpoint dual2027 проверяется потоковым SHA256, не копируется в Git и не загружается в triple. Dual2027 остаётся PASS исходного job; triple2027 остаётся SETUP FAIL, остальные шесть старых записей остаются NOT_RUN.

Причина FAIL: JSON превращает Adam betas tuple в list. `canonical_optimizer_settings()` нормализует копию обеих сторон без округления/потери полей; scalar types, None, False и отсутствующие поля различаются, NaN/Inf и неизвестные типы отклоняются. Реальные param_groups не меняются. Проверки weights/RNG/data/precision/calibrators/первого batch сохранены.

`resume_plan_003.json` заранее выбирает десять источников для пяти пар: exploratory pilot2026, reused dual2027 и семь новых fits. Очередь: triple2027, dual/triple2028, dual/triple2029, dual/triple2030. Старые gates наследуются по SHA; новых diagnostic batches/steps нет. Каждый новый fit проходит прежний setup и проверку сохранённой initialization; cross-attempt dual разрешён только для конкретного parent4355052 по SHA, checkpoint metadata и фактическим paired states.

`resume_lineage_003.json` перечисляет точные before/after hashes инфраструктурных изменений. Старый manifest проверяется против исторических Git blobs на login; runtime проверяет фактические bytes без Git. One-batch, initialization, модель, kernels, trainer, policy, исходный study_plan и aggregation formulas неизменны. AST-проверка prepare допускает только обёртку копии optimizer metadata. Один diagnostic step не объясняет историческую разницу .0633/.0615.

Новые результаты и summary JSON/Markdown: `runs/attempt_003/`; runtime, locks и `submission_003.json`: `slurm_logs/attempt_003/`. Старые summary/locks не переписываются. Счётчики новой попытки отделены от одного предыдущего completed confirmation fit и exploratory pilot. Submitter `resume_submit` выполняет один guarded sbatch; launcher `mamba3_three_time_confirmation_resume.sh` не вызывает Git. После Job ID мониторинг не выполняется. TEST запрещён.

## Инфраструктурный retry 002

Job4354908 (`886ef23`, source `1d524e1e...`) остановился на `git: command not found` до Python: 0 diagnostic batches, 0 optimizer steps, 0 scientific fits. Исходная reservation и логи побайтно сохранены в [evidence/submission_001](evidence/submission_001/failure.json); старый manifest сохранён отдельно. Это не mathematical attempt005. План, one-batch математика, порядок initialization, trainer, aggregation и policy не меняются.

Git теперь используется только в login/freeze/submit: exact HEAD, clean tracked state, опубликованная ветка и исторические Git blobs B/C. [Historical manifest](historical_sources.json) содержит 113 path/commit/SHA256, полученных из `git show`, а не объявленных по текущим файлам. На compute launcher не вызывает Git; `execution_sources()` сравнивает bytes с этим manifest, `identity()` сверяет переданный commit с immutable login evidence и reservation, повторно проверяет source hashes, imported paths и pinned upstream. Это **не проверка Git HEAD на compute**. Транзитивный `three_time.provenance.freeze()` содержит Git, но runtime его не вызывает; дочерние процессы запускаются через Python, не через Git.

Новые записи: `slurm_logs/attempt_002/login_verification.json` и `submission_002.json`. Связь commit/source фиксируется до единственного sbatch, включая parent job, причину retry, hashes policy/плана/historical manifest и доказательство отсутствия 39 scientific/pipeline путей. Новая reservation/lock/result блокирует повтор; общего force нет. Быстрый старт допускает RESERVED без job_id, но все bindings уже обязательны; после появления job_id разрешён только этот allocation.

No-Git regression использует реальные verifiers: блокирует запуск Git, проверяет изменённый source/historical manifest, неверный expected hash, отсутствующую/несогласованную reservation и повторный submit. На кластере `/bin/bash launcher --runtime-preflight-only <temporary-fixture-dir>` запускается с пустым PATH и явным существующим Python. Режим только проверяет provenance/imported paths/pinned runtime и metadata; не вызывает pipeline, CUDA forward, optimizer, полный проход данных или fit, не пишет реальные artifacts. Фиктивный job_id существует только в помеченных временных test fixtures. Все эти проверки обязательны перед retry, пакеты не устанавливаются.
