# Причинный входной attention/time адаптер Mamba3

Ограниченный пилот: помогает ли дополнительное представление исторического времени
на входе уже проверенной Mamba3 `separate`? Код и план фиксируются до метрик;
полезность и новизна не предполагаются. [План](study_plan.json),
[источники и проверка QuITE](evidence/sources.md),
[исходный протокол](../../reports/EVALUATION_SETUP.md).

## Архитектура

`item_embedding -> input_adapter -> прежние input_dropout/input_norm -> separate Mamba3`

Адаптер применяется один раз; длина и порядок сохранённых взаимодействий не меняются.
Сохранены два calibrator, original history gaps, TRAIN reference `838393 ms`,
`ADT = A * (DT_base * s_decay)`, `DT = DT_base * s_scan`, два Mamba layers,
FFN/residual/norm/dropout, output norm, последний valid state, tied item scorer и CE.
Frozen относится к реализации/протоколу: **все параметры обучаются с нуля**,
без загрузки старого checkpoint и без `requires_grad=False`.

| Режим | Новых параметров | Всего | Контроль |
|---|---:|---:|---|
| separate_replay | 0 | 610572 | Identity, проверка нового runner |
| time_add | 1120 | 611692 | Дополнительный соседний исторический gap |
| attention_content | 8192 | 618764 | Причинный attention без времени внутри адаптера |
| attention_time | 8240 | 618812 | Тот же attention с временной добавкой к logits |

Counts проверяются по реальным модулям локально и полным моделям в cluster preflight.
Основное сравнение time/content отличается на **48 параметров**, не exact capacity match.
`time_add` является небольшим практическим контролем, не равным attention по размеру.
`attention_content` не делает всю модель time-unaware: separate сохраняет реальное время.

`time_add`: `tau_i = log1p(gap_i/838393)`, `delta_i = Linear(16,64)(SiLU(Linear(1,16)(tau_i)))`.
Оба Linear имеют bias, последний полностью zero-init. Первый valid event и padding
получают delta=0; настоящий нулевой gap остаётся активным.

Attention: Q/K/V без bias, `64 -> 32`, четыре головы по восемь координат;
`logits_ij,h = dot(Q_i,h,K_j,h)/sqrt(8) + b_ij,h`, softmax по keys.
`delta = Linear(32,64,bias=False)(concat_heads(attention @ V))`, выход zero-init.
Для content `b=0`. Для time `r_ij=log1p(max(t_i-t_j,0)/838393)`;
`b=Linear(8,4,bias=False)(SiLU(Linear(1,8,bias=True)(r)))`, последний слой zero-init.
Дополнительных gates, dropout, FFN, norms и positional embeddings нет.
Вычисление не обходится при нулевых весах: выходная проекция получает gradient,
внутренние проекции начинают обновляться после неё.

## Входы и маски

IDs int64 `[B,L]`, lengths `[B]`, timestamps float64 `[B,L]`, embeddings fp32 `[B,L,64]`;
right padding, длина не меньше 1. Scientific `L<=50`; synthetic gate также проверяет L64.
Разрешённая пара: `valid_i AND valid_j AND j<=i`. Равные timestamps не открывают suffix.
Разности timestamps и безопасный log1p через logaddexp вычисляются в float64 до cast.
Padding санитарно обнуляется до разностей; masked/future пары обнулены до time embedding.
Forbidden keys имеют нулевые weights, а не только нулевые values.
Для padded query logits временно конечны; затем **все** weights/delta этой строки равны 0.
Residual всегда `u + masked_delta`, исходный padding embedding отдельно не меняется.

Нет target item/timestamp, внешнего момента запроса, idle gap, интерполяции или pooled query.
Это event-conditioned queries с сохранением L, а не learned variable/patch queries QuITE.
Временной attention уже использовался в рекомендациях, например TiSASRec; это адаптация
и контроль, не exact reproduction этих работ. Mantis и pretrained weights не используются.
Модуль не создаёт отсутствующие as-of snapshots и не исправляет ограничения KuaiRand-Pure.
Исторический precision adapter и порядок сортировки датасета переиспользуются без изменений;
новая float64 арифметика не означает новой пересортировки или исправления старых runs.
Причинная маска предотвращает чтение suffix, но не доказывает причинные эффекты рекомендаций.
Attention добавляет квадратичную стоимость по L; вся гибридная модель не объявляется линейной.

## Воспроизводимость и границы

Порядок: seed2026 -> dataset -> TRAIN/VALID loaders -> frozen separate -> RNG-isolated adapter
-> trainer -> fit. Hash backbone включает оба calibrator. Сохраняются также hashes adapter,
общих attention weights, pre-fit RNG и фактически потреблённого первого TRAIN batch.
Общий backbone/RNG/batch должны совпасть между четырьмя режимами; исторический опубликованный
context separate replay дополнительно проверяется по config/data/backbone/RNG/batch.
CPU-only и CUDA RNG не смешиваются: адаптер инициализируется CPU generator внутри fork_rng.

Adam .001, CE, прежние weight decay и defaults; batch2048/eval4096, epochs<=300,
VALID каждую эпоху, stopping_step10, full-ranking прежнего каталога 7111 реальных items.
Checkpoint выбирает штатный RecBole fit по VALID NDCG@10, включая last-equal-maximum
при округлении до четырёх знаков. Первые 27 эпох являются срезом истории, не отдельным fit.
Только state_dict и metadata JSON; synthetic roundtrip использует `weights_only=True`.
Reserved TEST split отбрасывается; **TEST loader не создаётся, TEST=NOT_RUN/count=0**.

Диагностика только внутри обычного VALID: delta/u norm, entropy по четырём attention heads
(отдельно >=2 keys), self/past mass, positional distance и time-bias. Старые scales имеют
две temporal heads, общие для двух слоёв. Streaming moments и bounded independent reservoirs
не потребляют training RNG; сохраняется диагностика выбранной лучшей эпохи.

## Проверки и запуск

Локально, в существующей CPU-среде:

```bash
python -m pytest experiments/mamba3_input_time/tests -q
python -m compileall -q experiments/mamba3_input_time
bash -n slurm/mamba3_input_time.sh
git diff --check
python -m experiments.mamba3_input_time.provenance
```

Старый FAIL документационного snapshot относительно `de0a137` не исправляется:
он не означает изменение scientific sources. Здесь проверяются реальные core/confirmation/context
hashes и побайтная стабильность старых tracked файлов относительно `ccc4849`.

До cluster checkout один раз проверить активные jobs и их WorkDir/Command. Активное использование
канонического repo блокирует deploy (`BLOCKED_ACTIVE_CHECKOUT`), без polling/cancel/нового worktree.
Нельзя удалять unrelated untracked или неопубликованные результаты. Разрешён только существующий
checkout `/home/daryumin/iberdov/diplom` и frozen `envs/mamba3`; ничего не устанавливать.
Перед submit: exact pushed commit, source hashes, tracked clean, нет старых artifacts/locks,
login preflight `python -m experiments.mamba3_input_time.preflight` (CPU construction, imports, stat).

```bash
export RUN_COMMIT=<exact-pushed-commit>
export EXPECTED_CORE_HASH=460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f
export EXPECTED_STUDY_HASH=<source_manifest.json-source_hash>
mkdir -p experiments/mamba3_input_time/slurm_logs
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 envs/mamba3/bin/python -m experiments.mamba3_input_time.submit
```

Helper сначала атомарно создаёт durable `slurm_logs/submission_001.json`, затем вызывает ровно
один `sbatch --parsable`. Неясный/ошибочный ответ запрещает retry. После Job ID сразу остановиться,
без squeue/sacct/чтения логов. Read-only проверка позже по отдельному запросу:
`sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList`.

Allocation: rocky/proj_1833/type_e, одна A100, 4 CPU, mem0, 8 часов, no-requeue.
Внутри: preflight -> GPU gates A-F -> четыре отдельных fresh-process TRAIN->VALID -> summary.
Gates: frozen-separate identity output/scores/CE/common+input gradients eval/train L50/L64;
ненулевые masking/causality/time fixtures; content/time reduction; независимый loop-reference;
optimizer updates и safe roundtrip; полный synthetic B2048,L50 forward/backward крупнейшей модели.
Atol/rtol: `1e-6/1e-5`, для loop-reference `1e-5/1e-4`, без ослабления после результата.
OOM/technical fail/budget останавливает оставшиеся NOT_RUN, сохраняет partial report, exit nonzero;
низкая метрика не останавливает фиксированный план. Deadline 7ч40м, новый fit не начинается
при остатке меньше 90 минут. CPU algebra не выдаётся за GPU evidence.

## Артефакты и интерпретация

Ожидаются (не существуют до выполнения):

```text
runs/gpu_checks_001.json
runs/mamba3_input_separate_replay_seed2026_001.json
runs/mamba3_input_time_add_seed2026_001.json
runs/mamba3_input_attention_content_seed2026_001.json
runs/mamba3_input_attention_time_seed2026_001.json
runs/pilot_summary.json
runs/pilot_summary.md
runs/pilot_summary.svg
slurm_logs/<run_id>/checkpoints/best_state_dict.pth
slurm_logs/<run_id>/checkpoints/best_metadata.json
```

Runtime, caches, checkpoints и логи находятся только в новом ignored `slurm_logs/`.
Report читает сохранённые JSON; SVG использует стандартную библиотеку. Ошибка только рисунка
не уничтожает JSON/Markdown; ошибки метрик/provenance остаются fatal, а не plotting warning.
Таблицы содержат full/first27 VALID, replay deltas, epochs, runtime и измеренную peak CUDA memory.
Исторический separate `.0633` показывается отдельно. Replay seed2026 не независимая реплика.
Главный контроль: time vs content; также time vs time_add и все кандидаты vs текущий replay.
Равенство time/content не подтверждает пользу времени; равенство time_add/time не обосновывает
усложнение. Один seed, без significance/устойчивого превосходства, без сравнения VALID с paper TEST.
Никакого автоматического переноса в results.csv, новой основной модели или второй серии.
