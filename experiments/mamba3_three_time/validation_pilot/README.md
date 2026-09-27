# SISO Dual / Triple Validation Pilot

Один exploratory-пилот seed2026: новый dual является основным comparator
для triple. Исторический dual NDCG@10=0.0633 приведён только для контекста;
повтор seed2026 не считается дополнительным независимым seed. Только
KuaiRand Protocol B, TRAIN -> VALID full-ranking. TEST loader не создаётся,
reserved split сразу исключается. MIMO training запрещён.

## Численная политика v1

Это **пересмотр после диагностики attempts001-004**, до новых recommendation
metrics, а не полностью заранее зарегистрированное исследование.
Исходные результаты attempt004 сохранены побайтно в `../evidence/attempt_004/`.
Старые exact-zero FAIL, планы, manifests, locks и `training_authorized=false`
не изменяются. `DIAGNOSTICS_COMPLETE` не означает полного correctness PASS.

Только для SISO вместо абсолютного нуля positional future-gradient как
единственного блокера используется инженерный бюджет:
`rho = ||g_future||_2 / (||g_prefix||_2 + ||g_future||_2) <= 1.1920928955078125e-7`.
Нормы считаются в float64; floor, clipping и округление gradients отсутствуют.
ZERO_SIGNAL не является информативным PASS. Это не теоретическая граница
ошибки kernel и не рекомендация PyTorch использовать такой критерий.
Official/base, frozen separate, local dual/triple проверяются одинаково.
MIMO acceptance и его near-zero policy не меняются.

Structural tolerances `1e-6/1e-5`, все общие gradients, tied/independent slots,
прежний SISO reference profile, finite/forward/padding/cross-user/init/source
проверки сохранены. Математические исходники execution004
`176f72a206eb48ddc4de492b87bc3273b75e1cae` не меняются, backend только
`upstream`; stable_adt/stable_scan/hybrid запрещены. Pinned Mamba:
`e9594ce1c732d97440f0332fdc43170a2294dbfa`.

## Допуск и исполнение

`study_plan.json` и `numeric_acceptance_v1.json` заморожены собственным
`source_manifest.json`, который также покрывает переиспользуемые зависимости.
Admission сначала проверяет 53 конкретных исторических SISO cases, разрешая
только явно разобранный exact-zero failure. Затем выполняет A/B parity,
ненулевое tied recovery, common-cotangent reference/VJP, новые prefix fixtures
seed314159 (L50/P25, L65/P31, multipliers 1/16), padding, length1, zero gaps,
cross-user, state_dict и dual/triple initialization parity.

Единственный успешный допуск:
`ACCEPTED_FOR_SISO_PILOT_WITH_DOCUMENTED_NUMERICAL_RESIDUAL`.
Это не глобальная математическая эквивалентность и не отмена исторических FAIL.
Любая другая required ошибка блокирует оба scientific fits.

Далее отдельный synthetic smoke batch2048/L50 проверяет CE/backward/Adam и
чистый state_dict roundtrip (`weights_only=True`). При OOM нет уменьшения batch.
Каждый scientific fit запускается в свежем процессе, с нуля. Общие backbone,
calibrator initialization, RNG перед fit и первый фактически потреблённый
batch сверяются; дополнительного чтения train loader ради hash нет.

Настройки наследуются из frozen reference и проверяются по effective Config:
Adam .001, weight_decay=0, CE, batch2048, eval batch setting4096, history50,
max epochs300, eval_step1, stopping_step10, VALID NDCG@10. Clipping и AMP
наследуются без изменений. RecBole fit/early stopping/tie semantics сохранены,
checkpoint фиксирует последнюю эпоху с равным лучшим score. Dual610572,
triple610638; две layers, две temporal heads, chunk64, reference838393 ms,
bounds[0.5,2], time functions общие для layers. Write/phase triple независимы.

Diagnostics detached, с отдельным RNG, только на существующих VALID forwards:
три scale distributions по двум heads, log-difference write/phase, saturation;
для dual write=phase. Никаких дополнительных evaluation.

## Защита и артефакты

`submit.py --commit <exact published SHA>` требует чистые tracked files,
текущие CPU tests/login preflight, отсутствие старых pilot outputs и locks.
Атомарная exclusive reservation создаётся до единственного `sbatch --parsable`;
неоднозначный ответ сохраняет reservation и запрещает retry. После Job ID
никакого ожидания или опроса. Launcher: rocky/proj_1833/type_e/A100x1, CPU4,
mem0, 4h, no-requeue. Внутренний дедлайн 3h50 с запасом для partial summary.

- `runs/admission_001.json`: GPU admission, а не recommendation result.
- `runs/smoke_001.json`: synthetic capacity/optimizer evidence.
- `runs/mamba3_three_time_siso_{dual,triple}_seed2026_001.json`: два scientific records.
- `runs/pilot_summary.json` и `.md`: сводка, raw VALID metrics и triple-dual delta.
- `slurm_logs/submission_001.json`, `pipeline.lock`, `pipeline_status.json`: ownership.
- `slurm_logs/<run_id>/checkpoints/`: pure state_dict и отдельная metadata.

При gate failure scientific records имеют NOT_RUN; job завершается non-zero.
При нехватке времени сохраняется INCOMPLETE, epochs не сокращаются.
Только один allocation, максимум два SISO fits, TEST=0. Низкая VALID метрика
не является техническим FAIL и не разрешает дополнительные runs.

CPU: `python -m unittest discover -s experiments/mamba3_three_time/validation_pilot/tests -v`.
Login: `python -m experiments.mamba3_three_time.validation_pilot.preflight`
(без GPU forward, обучения и установки пакетов). CPU algebra не является GPU evidence.
