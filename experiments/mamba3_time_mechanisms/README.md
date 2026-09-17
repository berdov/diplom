# Mamba3: раздельные временные механизмы

Абляция [RT-Mamba3](../mamba3_timeaware/README.md): должен ли один интервал одинаково управлять decay и scan/input-weight dynamics? Три запуска seed 2026 завершены по VALID: decay_only 0.0612, scan_only 0.0611, separate 0.0633 NDCG@10. [Таблицы, графики и ограничения](../../reports/MAMBA3_TIME_MECHANISMS_RESULTS.md); TEST этих вариантов не выполнялся.

## Формулы

`d=softplus(dd_dt+dt_bias)`; `ADT=A*(d*s_decay)`, `DT=d*s_scan`. Порядок операций сохраняет RT. [Модель](model.py) наследует scorer/CE, [forward](time_mamba3.py) меняет только подготовку ADT/DT. Нет новых embeddings, heads или kernels.

| Режим | ADT | DT | Calibrators | Параметры |
|---|---|---|---:|---:|
| vanilla | A*d | d | 0 | 610440 |
| decay_only | A*(d*s_decay) | d | 1 | 610506 |
| scan_only | A*d | d*s_scan | 1 | 610506 |
| shared | A*(d*s) | d*s | 1 | 610506 |
| separate | A*(d*s_decay) | d*s_scan | 2 | 610572 |

[TimeMechanisms](time_mechanisms.py) вычисляется один раз на историю и используется двумя слоями. Shared переиспользует один tensor/DT node, separate имеет два независимых calibrator с одинаковой инициализацией. Calibrator, TRAIN reference и bounds взяты из RT; zero-init даёт scale=1. Оба пути получают одинаковый history gap. [Входной контракт](../../reports/EVALUATION_SETUP.md) описывает формы, padding и точность времени.

## Основание адаптации

Pinned upstream и роли аргументов:
- [SISO decay](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py#L351): ADT задаёт cumulative log-decay.
- [SISO preprocessing](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py#L268): DT влияет на input weights и диагональный QK contribution.
- [Angle accumulation](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/angle_dt.py#L83): DT участвует в rotary Q/K.
- [Autograd wrapper](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py#L186) и [backward](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py#L511): отдельный dADT, dDT суммирует input-weight и angle paths.

При положительных scales можно записать `A_eff=A*s_decay/s_scan`, `ADT=A_eff*scan_dt`. Это перепараметризация дискретной recurrence с сохранением знака A, не неизменная continuous-time система с одним clock. Пути связаны в общем state/output; scan нельзя назвать исключительно frequency path. [Лицензия upstream](../mamba3_timeaware/LICENSE.upstream).

## Проверки и диагностика

Сохранённая [GPU equivalence](runs/gpu_equivalence_001.json): Suite A/B 8/8 PASS, zero-init 20/20 PASS, numerical max/mean errors=0 при atol=1e-6, rtol=1e-5. Suite A сравнивает official mixer и vanilla path; Suite B сравнивает полный RT и shared с ненулевым calibrator (output, loss, input/backbone/calibrator gradients). Cases: eval/train, L=50/64. [Происхождение evidence](../../reports/evidence/input_contract.json); CPU algebraic doubles не заменяют эту GPU-проверку.

[Diagnostics](diagnostics.py) не входят в loss/selection. По active gaps собираются per-head mean/std/min/max; p10/p25/p50/p75/p90 приближены reservoir до 8192 gaps с отдельным seed 2026. Первый gap и padding исключены, повторения в histories учитываются. Для separate есть mean absolute log-difference, pooled log correlation и доли decay>scan/scan>decay; при нулевой дисперсии correlation=null. Collector не меняет training RNG.

## Конфигурация и кластерные entrypoints

[config.py](config.py), [runner](run.py), [HSE VALID launcher](../../slurm/mamba3_time_mechanisms_validation.sh): rocky/proj_1833/type_e, 1 A100, mem=0, существующий envs/mamba3 и ignored slurm_logs. Режимы runner ограничены decay_only/scan_only/separate; `RUN_COMMIT` экспортируется на login-node, исходники проверяются по fingerprint и GPU evidence. Lock или существующий JSON блокируют повтор; Triton caches разделены по mode.

Общий TRAIN/VALID protocol описан в [условиях экспериментов](../../reports/EVALUATION_SETUP.md). Backbone обучается с нуля; TEST loader не оценивается. [Equivalence launcher](../../slurm/mamba3_time_mechanisms_equivalence.sh) сохранён для воспроизводимости, не для автоматического повторного запуска.

[Первоначальный план](../../reports/MAMBA3_TIME_MECHANISMS_PLAN.md) исторический. [Подтверждение shared/separate на пяти seeds](../../reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#confirmation) завершено; decay_only/scan_only остаются односидовыми абляциями. Причинность, статистическая значимость и формальная эквивалентность не установлены.
