# Gap Trap pilot: job4370162

Job завершён 2 октября 2026, 16:14:59 MSK: `COMPLETED 0:0`, 51:22.
GPU gate, smoke и оба fits PASS. Gap Trap уступил fresh fixed replay:
VALID NDCG@10 0.0626 против 0.0633 (−1.106%).
[Результаты и ограничения](RESULTS.md); [сверка](evidence/job4370162/independent_audit.json).

## Зафиксированный запуск

- Branch: `exp/mamba3-gap-trap`.
- Execution commit: `8ee54cf1faee22bb3abad3f31aa9268d77a125d1`.
- Source: `ad9e4591d570a223848c8588dc2e0fed104d6802fccc22e7f2e1c4cc4068ecd0`.
- Checkout: `/home/daryumin/iberdov/diplom`, SSH `hse-karizma`.
- Environment: `./envs/mamba3/bin/python`; пакеты не менялись.
- A100×1, CPU4, mem0, 06:00:00, no-requeue. Максимум 2 scientific fits.
- `fixed_replay` seed2026 → `gap_trap` seed2026, каждый в свежем процессе.
- TEST=NOT_RUN, count=0. Confirmation и пункт 4 не запускать.

Обычный и no-Git CPU preflight: оба PASS, 20 тестов, 0 failures/errors/skips,
CUDA не инициализирована. No-Git проверка не вызвала Git.
GPU gate прошёл 4 случая / 133 обязательных результата;
smoke 2048×50 прошёл по 3 шага на вариант. Старые 45/2342 и 9/228 наследуются
по SHA и lineage. Policy и kernels не менялись.

Пути относительно checkout:

- `experiments/mamba3_gap_trap/slurm_logs/attempt_001/` — reservation,
  submission, pipeline_status, stdout/stderr, checkpoints и metadata.
- `experiments/mamba3_gap_trap/runs/attempt_001/` — inherited evidence,
  targeted_gate, smoke, два result JSON и итоговый pilot_summary.
- `experiments/mamba3_gap_trap/evidence/` — сохранённые preflight и submission
  с SHA256; это снимки, их не перезаписывать при следующем чтении.

## Состояние после завершения

Сохранены 38 raw-файлов с SHA256; числовой отчёт заново сверен с JSON,
логами и metadata. Проверены 96 эпох, 1152 значения метрик, parity common
state/RNG/first batch/optimizer/data, выбор checkpoint и first27.
Checkpoint SHA подтверждены потоковым чтением на кластере; веса не загружались.

Не повторять submit, gate, smoke или runner. Лимит задания исчерпан:
один job, два завершённых fit, TEST0. Дополнительные seeds и пункт 4
не запускать. Этот single-seed pilot не показал улучшения основной метрики.
Alpha лучшего checkpoint равна 0.0005576708354, последней эпохи — 0;
нулевой final alpha не означает идентичности обученных backbone.

## Уже опубликованный пункт 2

Main: `4e5fcd660115318ed8df26849498eabe3a72bb26`; реестр 96 → 108.
Raw job4365206 сохранены, новая таблица и парный график опубликованы.
[Отчёт](https://github.com/berdov/diplom/blob/main/reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#head-timescales-confirmation).
Fixed-reference dual остаётся контролем: head−fixed на новых четырёх seeds
составил +0.040%, 2/4 wins; head−shared +0.319%, 3/4, first27 mean delta=0.
