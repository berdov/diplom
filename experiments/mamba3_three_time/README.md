# Три внутренних временных пути Mamba3

Реализуется первый пункт плана Дмитрия: decay/write/phase. Шкалы голов,
gap-dependent Trap, функции слоёв и временная память пока не добавлены.
Входные адаптеры и эксперты отложены. Это инструмент для будущего сравнения,
не доказательство новизны, качества MIMO или преимущества трёх путей.

## Вопрос и формулы

Помогут ли три функции одного исторического gap вместо двух? Content-dependent
базовый шаг сохраняется: `d = softplus(dd_dt + dt_bias)`, прежнее `A < 0`.

```text
ADT      = A * (d * s_decay)     -> затухание состояния
DT_write = d * s_write          -> trapezoidal input weights
DT_phase = d * s_phase          -> cumsum(tanh(Angles) * pi * DT_phase) mod 2pi
```

`base`: все scales=1; `dual`: decay и общий scan; `triple`: независимые
decay/write/phase. Один наблюдаемый gap, три отображения, не три измеренных
времени и не доказанные психологические процессы. Общее состояние связывает пути.

Calibrator: `1 -> 16 -> 2`, SiLU, zero-init последнего Linear,
`exp(log(2)*tanh(raw))`, bounds `[0.5,2]`, frozen TRAIN reference `838393 ms`.
Используется прежний безопасный float64 log1p. Первое событие и padding нейтральны;
реальный zero-gap активен. Функции общие для двух layers. Calibrators создаются
после backbone в isolated RNG context; одинаковые копии не разделяют параметры.

## SISO и MIMO

Внешний recommender прежний: D64, state128, expand2, head64, groups1, layers2,
rope_fraction0.5, no-outproj-norm, bf16 mixer. SISO chunk64;
native MIMO rank4/chunk16. Это заранее выбранное различие реализации, не tuning.
SISO counts: base610440, dual610572, triple610638. Фактические MIMO counts
записываются CPU preflight по полной модели с synthetic catalog на 7112 tokens;
равенство SISO/MIMO не предполагается.

Главное будущее сравнение: dual/triple отдельно внутри каждой архитектуры.
Область реализации: dense padded full-sequence forward/backward, без внешних
cached states, step, varlen или возврата final states. Неподдержанные kernel API
явно отклоняются. Это не готовый autoregressive deployment.

Локальные autograd wrappers используют неизменённые pinned kernels. Производная
write возвращается только `DT_write`, phase только `DT_phase`; в dual передаётся
один tensor в два slots и autograd складывает вклады. Q/K/V, ADT, Trap, Angles,
biases, D/Z и MIMO projections сохраняют свои производные. Нет detach, отношения
scales или преобразования через inverse-tanh. [Аудит](evidence/upstream_audit.md),
[upstream hashes](upstream_manifest.json), [лицензия](LICENSE.upstream).

## Проверки и ограничения

[План технических проверок](test_plan.json) фиксирует допуски до GPU-результатов.
CPU: float64 gradcheck ADT/write/phase/Trap/Angles, причинность, прямые
вмешательства, expanded causal sum, границы mod, alias/chain rule, RNG isolation,
counts calibrators, delayed first-layer gradients и запись evidence до assertion.
Отсутствующие CUDA/RecBole/Mamba на Mac обозначаются SKIP, не GPU PASS.

GPU A-I: official base parity; nonzero dual recovery; tied triple и сумма
производных; независимый математический reference; причинность/padding/zero-gap;
границы chunks; synthetic optimizer steps; weights_only state_dict roundtrip;
native upstream support. Инициализация: warm CUDA -> Config -> seed -> CPU model
с isolated calibrators -> CUDA -> отдельные hashes весов и RNG. Каждая строка
сохраняется до именованных checks; при FAIL остаются ключи/формы/dtypes/различия.

Структурная parity: atol1e-6/rtol1e-5. Kernels действительно используют bf16,
даже если часть аргументов fp32. Reference работает на quantized входах обычными
fp32 PyTorch операциями без внутренних bf16 округлений. Отдельные фиксированные
допуски reference сначала проверяются official tied baseline; при его FAIL
triple/reference имеет INCONCLUSIVE, а допуски не расширяются. Даже при PASS
это ограниченная численная проверка, не доказательство полной эквивалентности.

MIMO на A100 может не поддерживаться pinned TileLang kernels. Сначала запускается
официальный baseline, traceback сохраняется отдельно. Нет fallback на SISO,
изменения rank/chunk/env ради PASS. SISO и MIMO выполняются в разных процессах
одной allocation; общий PASS требует все обязательные checks обеих архитектур.

Frozen sources/results/manifests сверяются побайтно с актуальной основой
`ccc4849e6e1ea460a316ab9648f36c54bd832fd4`, старый core hash неизменен.
Исторический `test_frozen_files_byte_stable` сравнивает в том числе документацию
с более ранним `de0a137`; известный FAIL этого старого snapshot не исправляется
отключением provenance guards и не заменяет новую побайтную сверку.

## Запуск

На login: существующий `envs/mamba3`, без установки пакетов; verify/imports/versions
и CPU construction. Pytest на кластере не требуется. Launcher:
`slurm/mamba3_three_time_correctness.sh`, rocky/proj_1833/type_e, A100x1,
CPU4, mem0, 01:30:00, no-requeue. До смены checkout оператор обязан разово проверить
активные jobs и их WorkDir/Command; при занятом checkout переключение запрещено.

```bash
python -m experiments.mamba3_three_time.provenance
python -m experiments.mamba3_three_time.preflight --output experiments/mamba3_three_time/slurm_logs/login_preflight.json
python -m experiments.mamba3_three_time.submit --commit EXACT_PUBLISHED_SHA
```

Submit создаёт durable reservation и вызывает sbatch ровно один раз. При
неоднозначном ответе reservation остаётся; автоматического retry нет. После Job ID
остановиться: не ждать, не опрашивать Slurm и не читать результаты.

Evidence: `runs/siso_correctness_001.json`, `runs/mimo_correctness_001.json`;
сводка `runs/technical_summary.json`; этапы/логи/guards в ignored `slurm_logs/`.
Каждая suite сохраняет реальные ошибки и failed keys, а не подменяет их нулями.
На compute-node git не требуется: commit подтверждается login, файлы Python hashes.

[Будущие шесть режимов](future_plan.json) подготовлены только как план. Следующие
пункты 2-5 перечислены как гипотезы, без кода. После correctness PASS научное
обучение требует отдельного запроса. Сейчас scientific fits=0, TRAIN=0, VALID=0,
TEST=0; разрешены только synthetic forward/backward/optimizer-step tests.
