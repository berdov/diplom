# MIMO base / dual / triple: ограниченный пилот

## Исправление hook: попытка 003

Попытка 002 (job4357075) прошла CPU-preflight и 28 admission cases, затем
остановилась на `prefix_base_L17_P7_x1`: hook вызывал `retain_grad()` внутри
intervention с `no_grad`. Исходное evidence сохранено в `evidence/job4357075/`;
его FAIL и отсутствие несохранённых проверок не пересматриваются.

В `prefix_checks.check_prefix` hook действует только в gradient-фазе: обязательны
grad-enabled, ровно один ожидаемый positional tensor и конечный gradient нужной
формы. Output/gradient клонируются с detach, hook снимается в finally до
interventions. Их forwards остаются под no_grad, лишнего gradient forward нет.
Callback Registry сохраняет stages, gradient checks, снятие hook и каждый
intervention; поздняя ошибка сохраняет partial evidence/traceback и missing keys,
но не даёт PASS. CPU regression использует этот же production helper и весь
prefix case с маленькой Embedding-моделью; обнаруживает в том числе detached
зависимость prefix от suffix. Это не MIMO GPU evidence.

Новые артефакты только в `slurm_logs/attempt_003/` и `runs/attempt_003/`.
Retry разрешён для конкретного hook failure с нулём scientific fits; обе
завершённые ancestor attempts проверяются по сохранённым SHA256. Неизвестная
попытка, новые старые scientific outputs или занятые новые paths блокируют submit.
Новый job заново выполняет все 45 cases/2342 leaves, не переиспользует 28 PASS.
Policy, scientific plan, модель, kernels, fixtures, trainer и допуски неизменны.

## Инфраструктурная попытка 002

Первая попытка 4356310 остановилась на CPU-preflight до admission/smoke/fits:
`CPU preflight initialized CUDA`. Evidence сохранено в `evidence/job4356310/`,
оригинальные locks/reservation/logs/summary остаются на месте. Первый вызов,
инициировавший CUDA, не установлен по прежнему traceback.

Попытка 002 использует `slurm_logs/attempt_002/` и `runs/attempt_002/` для всех
операционных артефактов, checkpoints и результатов, сохраняя logical run_id.
Новая reservation привязана к parent job, commit, source hash и SHA256
preservation manifest. Submit дополнительно сверяет оригиналы failure evidence
и отсутствие parent scientific artifacts. Общего force/retry режима нет.

Только CPU-preflight получает КОПИЮ allocation env с `CUDA_VISIBLE_DEVICES=""`
до запуска нового Python. Launcher `--preflight-only` делает то же перед exec.
Parent, admission, smoke и fits не получают эту маску или выдуманный индекс GPU.
Pipeline сохраняет только наличие/значение маски parent/child, без полного env.
RecBole `_init_device` записывает env из gpu_id, поэтому и проверяемый, и reference
CPU Config имеют use_gpu=False/device=cpu/gpu_id="". Scientific gpu_id не меняется
и не включается в общий список допускаемых научных расхождений.

Preflight сохраняет entry/import/config/model/runtime stages до assertions;
читает только `torch.cuda.is_initialized()`, не current_device/name/CUDA RNG
для диагностики. Проверяет CPU-устройства всех parameters/buffers и сохраняет
ошибку с последним успешным этапом. Guard остаётся также после ВСЕГО runtime
verification. CUDA initialization не приравнивается к model forward/optimizer/fit.
Фактическая доступность A100 проверяется прежним runtime(True) первого admission
case в разрешённой allocation; CPU env tests не являются GPU evidence.

Scientific settings, `study_plan.json`, numerical policy и 45 cases/2342 leaves
первой попытки неизменны. Для текущей попытки пути ниже понимаются относительно
`runs/attempt_003/` и `slurm_logs/attempt_003/`; attempt002 и корневые пути исторические.

KuaiRand: хронологический leave-one-out, полный каталог. Только TRAIN/VALID;
TEST loader/evaluation запрещены. Один exploratory seed2026, не подтверждение
статистической значимости и не сравнение с опубликованным TEST TiM4Rec.

## План до метрик

Один job: rocky / proj_1833 / type_e / A100x1 / CPU4 / mem=0 / 08:00:00 /
no-requeue. Существующий `envs/mamba3`, без установки пакетов. `PYTHONNOUSERSITE=1`
исключает сторонний user-site NumPy, не меняя environment.

Последовательность: CPU runtime preflight, admission, synthetic smoke полного
batch2048/history50, затем свежие subprocess base -> dual -> triple и сводка.
Любой технический FAIL блокирует последующие fits. Низкая VALID-метрика сама
по себе не блокирует. Нет retries, tuning, fallback batch или новых seeds.
Запас на сохранение 600 секунд; новый fit не начинается при остатке <5400 секунд.
Это заранее выбранный минимальный бюджет, не гарантия завершения за 90 минут;
epochs/early stopping по времени не сокращаются. Частичная серия отмечается явно.

`study_plan.json` фиксирует задачи, параметры и точные обязательные leaves каждого
case. Фактические CPU counts: base 714888, dual 715020, triple 715086.
Rank4/chunk8, d_model64/d_state128/expand2/headdim64, ngroups1, layers2,
rope_fraction=.5, bf16 mixer и прежняя outer precision. Kernel-boundary adapter
дополняет L17->24/L50->56, затем crop; история остаётся 50 реальных событий максимум.
Поддержка произвольных native lengths/step/cache/varlen не заявляется.

Base не имеет calibrators, но сохраняет content-dependent внутренний d.
Dual имеет decay и общий autograd tensor write/phase; triple имеет три пути.
Calibrators 1->16->2 SiLU общие для layers, bounds [.5,2], zero-init последнего
Linear; TRAIN reference 838393 ms, float64 gaps, первое событие/padding нейтральны,
реальный zero-gap активен. Temporal heads=2, layers=2, MIMO rank=4 не смешиваются.

Adam .001, CE, weight_decay0, train2048/eval setting4096, max300 epochs,
eval_step1, stopping_step10, последняя равная максимальная округлённая VALID
NDCG@10. Effective config сравнивается с raw опубликованного SISO pilot,
допустимы только mode/is_mimo/rank/chunk/output paths и CPU device при preflight.
Порядок runtime/config -> seed -> data/loaders -> model -> trainer -> fit прежний.
Dataset.build возвращает reserved TEST, который сразу исключается; TEST loader нет.

## Численная политика

`mimo_numeric_acceptance_v1.json` является НОВЫМ решением после diagnostics003/004,
до первых MIMO recommendation metrics. Старые policy/evidence/FAIL не меняются.

- Structural: atol=1e-6, rtol=1e-5, без ослабления.
- Reference output: atol=.003, rtol=.08, relative L2 cap=.08.
- Общий output-independent signed/nonnegative VJP: atol=.006, rtol=.12, cap=.12.
  Cotangent нормирован до bf16 cast, реальные bytes/hash/norm сохраняются;
  reference получает те же квантованные inputs с повышением до fp32.
- Backward residual: float64 rho=||future||/(||prefix||+||future||)
  <=1.1920928955078125e-7 без floor; ZERO_SIGNAL не PASS. Exact-zero legacy
  остаётся отдельным измеренным результатом, не превращается в пройденный check.
- Старый nonlinear loss: finite + прежний mixed atol/rtol обязательны; near-zero
  pure-relative cap и его nonlinear-loss gradients остаются advisory, не PASS
  задним числом. Общий VJP обязателен независимо от loss. Объясняющая граница
  переноса output error в loss не доказывает correctness.

Это инженерный допуск к пилоту, не теорема об эквивалентности. D-only oracle
проверяет skip D; continuous/STE трактовка bf16 не равна производной дискретного
округления. Baseline/reference проверяется раньше triple на каждой семье fixtures;
при FAIL triple остаётся INCONCLUSIVE. Нулевой reference без floor; отсутствующий
математически ожидаемый gradient, NaN/Inf, потерянный/дублированный case или
required leaf блокируют допуск. Tied и triple используют явно разные DT slots,
это не утверждение строго парного сравнения их величин ошибок.

Исторический audit: attempt003 MIMO имел общий FAIL, включая triple L15 D-gradient,
near-zero tied L50 loss и exact future-gradient. Native aligned/adapter и ряд
structural/optimizer/roundtrip checks уже работали. Attempt004 имел статус
DIAGNOSTICS_COMPLETE, не admission: его верхние finite checks недостаточны;
вложенные shared-VJP comparisons были отдельно прочитаны. Chunk16 с rank4
не помещался в shared memory; он не используется. Новый seed314159 имеет
собственный generator, не hardcoded seed2026 под другим названием.

Успех нового gate называется только
`ACCEPTED_FOR_MIMO_PILOT_WITH_DOCUMENTED_NUMERICAL_LIMITATIONS`.
До его результата GPU PASS не заявляется. Допуски после gate не меняются.

## Реализация и происхождение

Переиспользуются неизменённые математические модули `mamba3_three_time`,
frozen temporal core 460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f,
pinned Mamba e9594ce1c732d97440f0332fdc43170a2294dbfa.
Publication commit a462c412ef72dc0fa8af1ec6b48d1eef47ba8e72 отличается от
исторического execution 995c5cde6449ea429c1d80ca6ab276b9791041c0 и нового execution,
который записывает login verification. Старые manifests целиком не перевычисляются.
Собственный `source_manifest.json` включает консервативное замыкание imports,
configs и frozen metadata, но не себя, reservation, будущие результаты или
посторонние publication docs/CSV. Исторические зависимости сверяются с Git blobs
своего execution commit. Реальные imported paths проверяются против manifest.

Новый trainer наследует safe instrumentation `mamba3_context_time.trainer`:
`Trainer.fit`, tie/early-stopping, VALID-only evaluate, first consumed batch hash
и pure state_dict checkpoint callback не переписаны. Изменены только MIMO guard,
collector трёх scales и TRAIN/VALID timing/peak memory. SISO-only guard не вызывается.
Best diagnostics берутся из той же VALID эпохи, без дополнительной evaluation.
Base scales=1/calibrator_count0 валидны. Safe roundtrip только weights_only=True.

## Ownership и запуск

Login проверяет tracked clean, опубликованный exact commit, CPU/effective-config
tests, launcher preflight-only с PATH без Git и ограничения ресурсов. Immutable
login record -> immutable reservation/token -> один `sbatch --parsable`.
Operational submission record отдельно от hash identity. Compute допускает race
до записи job_id, но atomic pipeline.lock связывает allocation/token; child
проверяет тот же контекст. Записи публикуются атомарно без overwrite, последующие
обновления только принадлежащих процессу records. Неоднозначный submit не повторяется.
Git требуется только login freeze/submit; compute использует SHA и pinned package.

`slurm/mamba3_mimo_time.sh --preflight-only` не создаёт dataset/loaders, GPU forward
или fit. CPU tests: `python -B -m unittest discover -s experiments/mamba3_mimo_time/tests -v`.
Отсутствующий PyTorch на Mac означает SKIP зависимых CPU tests, не GPU evidence;
на login они обязательны. Временные тестовые fixtures не попадают в реальные runs.

Артефакты: `runs/admission_001.json`, `runs/smoke_001.json`, три
`runs/mamba3_mimo_{base,dual,triple}_seed2026_001.json`, `runs/pilot_summary.json/.md`.
Runtime/checkpoints/stdout/stderr/reservation находятся в ignored `slurm_logs/`.
Сводка всегда сохраняет base/dual/triple и отрицательные дельты; отсутствующие
метрики обозначает NOT_RUN/INCOMPLETE, не нулями. Первые 27 эпох являются срезом
той же history. SISO dual .0615/triple .0623 показаны только как историческая
другая архитектура одного seed, не подмена comparator и не пятисидовое доказательство.
