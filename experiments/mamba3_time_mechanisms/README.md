# Mamba3: раздельные temporal mechanisms

Подготовка реализации и проверок, **без scientific запусков**. Ветка
`exp/mamba3-time-mechanisms` от `de0a137a6f46b5b214ccdc03a5ff1ede93454704`.
Мотивация научрука: нужно ли одному physical inter-event gap одинаково управлять
затуханием памяти и остальными native DT dynamics? Это ablation существующей
[RT-Mamba3](../mamba3_timeaware/README.md), не отдельный новый backbone.

## Сохранённая GPU equivalence

[Evidence JSON](runs/gpu_equivalence_001.json) сохранён побайтно из job **4332447**:
COMPLETED, 0:0, NVIDIA A100-SXM4-80GB. Проверенный implementation commit:
`976eccc82d830929d82df8bbd91d783dda3b0f1a`.
Source fingerprint: `460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f`.
Evidence SHA256: `6484b49cf6d69e2066cc8db6c9c7dbc12bf33a682136682fbf83798e70c9de37`.
Suite A/B: **8/8 PASS**, все numerical max/mean errors=0 при atol=1e-6,
rtol=1e-5. Zero-init identity: **20/20 PASS**. Происхождение JSON не изменено.

VALID-launcher больше не требует git на compute-node: `RUN_COMMIT` проверяется
и экспортируется при submit на login-node. Перед обучением Python проверяет
текущие исходники по `EXPECTED_SOURCE_FINGERPRINT`, фиксированному проверенному
fingerprint и `require_equivalence()`. Triton caches разделены по mode.
Ресурсы, model/config и training protocol не изменены. Результаты будущих
decay_only/scan_only/separate TRAIN→VALID пока неизвестны; TEST запрещён.

## Pinned upstream audit

Проверен `state-spaces/mamba` commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`.

- [SISO forward, L351–408](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py#L351):
  `ADT` преобразуется в cumulative log-decay. Экспоненты управляют переносом
  предыдущего состояния, внутриблочными взаимодействиями и накоплением state.
- [SISO preprocessing, L268–321](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py#L268):
  `DT` с sigmoid(Trap) задаёт gamma, shifted gamma и scale для K/input weights
  и диагонального QK contribution. Поэтому это не исключительно frequency path.
- [Angle accumulation, L83–106](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/angle_dt.py#L83):
  `DT` также умножает pi*tanh(Angles) перед causal cumsum/modulo 2*pi,
  используемыми для rotary Q/K. Название: scan / rotary **и input-weight** dynamics.
- [Autograd wrapper, L186–260](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py#L186):
  отдельный `dADT`; `dDT` складывается из input-weight и angle paths.
  [Backward kernels](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py#L511)
  возвращают эти производные раздельно, не восстанавливая A через ADT/DT.

**Вывод аудита:** API и вычисление поддерживают независимые tensors одинаковой
формы. При положительных s_decay,s_scan можно записать
`A_eff = A * s_decay / s_scan`, тогда `ADT = A_eff * scan_dt`.
Это корректная перепараметризация дискретной recurrence с отрицательным decay,
не утверждение о неизменности исходной continuous-time системы с одним clock.
Знак A сохранён, но effective coefficient уже иной; дополнительный clamp к A_eff
не вводится. Роли путей пересекаются в одном state/output, поэтому их нельзя
называть независимыми физическими подсистемами. Новые kernels не создаются.
GPU forward/backward equivalence подтверждена сохранённым evidence выше;
математическая допустимость сама по себе не заменяет численной проверки.

## Формулы и режимы

`d = softplus(dd_dt + dt_bias)`, `A` вычисляется frozen способом.
`decay_dt = d*s_decay`, `scan_dt = d*s_scan`, `ADT=A*decay_dt`, `DT=scan_dt`.
Порядок floating-point операций: `A*(d*s_decay)`, как в historical RT.

| Mode | real time in ADT | real time in DT | calibrators | Parameters |
|---|---|---|---:|---:|
| vanilla | no: A*d | no: d | 0 | 610440 |
| decay_only | yes: A*(d*s_decay) | no: d | 1 | 610506 |
| scan_only | no: A*d | yes: d*s_scan | 1 | 610506 |
| shared | yes: A*(d*s) | yes, same scale: d*s | 1 | 610506 |
| separate | yes: A*(d*s_decay) | yes, independent: d*s_scan | 2 | 610572 |

[TimeMechanisms](time_mechanisms.py) находится на уровне всей модели, не layer.
Каждый calibrator вычисляется один раз и shared между двумя layers. `shared`
возвращает один tensor и переиспользует один conditioned DT node, сохраняя
порядок накопления autograd как в RT. `separate` содержит ровно два разных
объекта с независимыми parameters, но одинаковым начальным state.

Переиспользуется frozen `TimeCalibrator`: log1p(gap/reference), Linear(1,16),
SiLU, Linear(16,2), exp(log(2)*tanh(raw)). Последний Linear нулевой.
Reference=838393 ms, источник TRAIN, bounds [0.5,2]. Инициализация точно scale=1.
`vanilla` вообще не имеет calibrator. Counts проверяются по реальным parameters
при создании полной модели; CPU tests отдельно считают temporal modules (0/66/132).

[Core forward](time_mamba3.py) меняет только подготовку ADT/DT;
dd_A, A, B/C, Trap, Angles, D, norms, RoPE, chunk size, residual/FFN неизменны.
[Model](model.py) наследует RT scorer/CE и вызывает vanilla constructor,
не создавая лишний historical calibrator. Нет новых embeddings/heads/experts.
Copyright и лицензия upstream: [Apache-2.0](../mamba3_timeaware/LICENSE.upstream).

## Causal input contract

Переиспользуется frozen `history_gaps`: первый valid gap=0, adjacent history
differences clamp_min(0), padding и первый элемент имеют neutral scale=1.
Оба separate calibrators получают один и тот же history gap. Target timestamp
не является аргументом forward; precise float64 history dataset переиспользован.
Scientific config сохраняет Protocol B и max sequence length=50.
Длина 64 используется только в synthetic equivalence.

## Проверки и диагностика

CPU tests проверяют formulas/isolation, zero-init, independence, shared object,
causal gaps, parameter counts temporal modules и неизменность frozen файлов.
Wrapper identity проверяется на CPU с явно обозначенным algebraic scan double;
это **не** замена реальной GPU equivalence.

[GPU script](gpu_equivalence.py), выполнен в job 4332447:
- Suite A: official mixer против нового vanilla path; eval/train, L=50/64,
  output и input-gradient max/mean errors.
- Suite B: полные existing RT и new shared с идентичным state, включая
  **ненулевой** calibrator, histories, dropout RNG; output, CE, embedding-output
  input gradient, все backbone и calibrator gradients. Дополнительно full-wrapper
  zero-init identity всех пяти modes на одном backbone, eval/train, L=50/64.
- atol=1e-6, rtol=1e-5; ожидается близость к exact zero. CUDA unavailable даёт
  SKIP/exit 2, не PASS. Результат machine-readable, без scientific metrics.

[Diagnostics](diagnostics.py) detached и не входят в loss/selection. Каждый VALID
pass: exact per-head mean/std/min/max по active history gaps; approximate
p10/p25/p50/p75/p90 по uniform priority reservoir <=8192 gaps с отдельным seed 2026.
Padding и первый gap исключены, повторения gaps в histories учитываются как
history occurrences. Separate: mean absolute log-difference, pooled log correlation,
fractions decay>scan и scan>decay. При zero variance correlation=null. Gap buckets
не используются; никаких VALID-selected thresholds. Collector не меняет training RNG.

## Подготовленный scientific protocol

[Runner](run.py) разрешает только decay_only, scan_only, separate и создаёт:
`mamba3_decay_only_validation_001`, `mamba3_scan_only_validation_001`,
`mamba3_separate_time_validation_001`. Vanilla/shared scientific reruns запрещены.
Перед обучением обязателен полный GPU PASS evidence с fingerprint текущих
implementation/config файлов и pinned Mamba. Изменение кода инвалидирует evidence.
Существующий result или atomic lock запрещают автоматический повтор.

Scratch backbone, seed2026, Adam .001, CE, batch2048/eval4096, max300,
patience10, every-epoch VALID NDCG@10, full-ranking. Frozen TRAIN stats/manifest
и split проверяются; TEST dataset reservation отбрасывается без loader evaluation.
`TEST=NOT_RUN`, `test_evaluation_count=0`. Checkpoints/logs только в ignored
`slurm_logs/<run_id>`. Сейчас нет ни одного нового scientific JSON.

Подготовлены [equivalence launcher](../../slurm/mamba3_time_mechanisms_equivalence.sh)
и [generic VALID launcher](../../slurm/mamba3_time_mechanisms_validation.sh)
с жёстким mode allowlist: rocky/proj_1833/type_e/1 A100/mem0, существующее envs/mamba3.
Перед отдельно разрешённым submit нужен существующий `slurm_logs` для Slurm output.
Разрешены только три отдельных TRAIN→VALID режима после проверок и записи
submission record. Shared/vanilla не перезапускаются; повтор equivalence не нужен.

## Заранее заданная интерпретация

Canonical references: Vanilla VALID=0.0584, Shared VALID=0.0605. Все новые VALID
неизвестны. Исторический shared TEST=0.0613 уже наблюдался; новые решения
принимаются только по VALID, новый TEST запрещён до отдельного freeze decision.

| Наблюдение на VALID | Ограниченная интерпретация |
|---|---|
| decay_only > scan_only | evidence в пользу conditioning decay path в этой постановке |
| scan_only > decay_only | evidence в пользу conditioning scan/input-weight path |
| shared > оба isolated | evidence в пользу совместного clock |
| separate > shared | independent mappings могут дать дополнительный signal |
| separate примерно shared | необходимости дополнительной параметризации не наблюдается |
| separate < shared | shared clock может быть полезным structural constraint |

«Примерно» не является заранее заданным statistical equivalence margin: не делаем
формальный equivalence claim. Один seed не даёт causal proof/statistical significance.
Никакого post-test tuning. [Краткий Methods/Ablation plan](../../reports/MAMBA3_TIME_MECHANISMS_PLAN.md).
