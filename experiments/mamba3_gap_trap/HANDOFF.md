# Gap Trap pilot: job4370162

Заявка отправлена 2 октября 2026, 11:09:16 MSK. На первом чтении scheduler:
`PENDING (Priority)`, старт не назначен. Это не результат GPU допуска или fit.
Последнее наблюдение и точное время хранятся в `runtime/state.json`.

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
GPU gate ещё должен проверить 4 случая / 133 обязательных результата,
затем smoke 2048×50 по 3 шага на вариант. Старые 45/2342 и 9/228 наследуются
по SHA и lineage. Policy и kernels не менялись.

Пути относительно checkout:

- `experiments/mamba3_gap_trap/slurm_logs/attempt_001/` — reservation,
  submission, pipeline_status, stdout/stderr, checkpoints и metadata.
- `experiments/mamba3_gap_trap/runs/attempt_001/` — inherited evidence,
  targeted_gate, smoke, два result JSON и итоговый pilot_summary.
- `experiments/mamba3_gap_trap/evidence/` — сохранённые preflight и submission
  с SHA256; это снимки, их не перезаписывать при следующем чтении.

## Следующая проверка

Только read-only: `squeue -j 4370162`, `sacct -j 4370162`, `scontrol show job 4370162`,
чтение существующих JSON и логов. Не повторять submit, gate, smoke или runner.
Не менять cluster checkout, пока заявка PENDING/RUNNING. Более поздние commits
в этой ветке могут содержать только документы и сохранённые свидетельства;
они не меняют execution commit заявки.

После terminal status сохранить полученные файлы с SHA256, проверить provenance,
число started/completed fits, gate registry, smoke, TEST0, pairing common state,
RNG/first batch/optimizer/precision/data, непрерывную историю и last-tie checkpoint.
Проверить хеши checkpoint на кластере; весовые файлы не нужны для числового отчёта.
Ничего не переобучать при неполном или отрицательном результате.

Если оба fits завершились, показать VALID NDCG@10, HR@10, best/actual epochs,
first27, TRAIN/VALID seconds, peak memory, absolute/relative delta и сохранённую
диагностику alpha на заранее заданной gap grid. Один seed не доказывает устойчивость.
Нулевой alpha допустим: constrained optimizer мог предпочесть исходный контроль.

## Уже опубликованный пункт 2

Main: `4e5fcd660115318ed8df26849498eabe3a72bb26`; реестр 96 → 108.
Raw job4365206 сохранены, новая таблица и парный график опубликованы.
[Отчёт](https://github.com/berdov/diplom/blob/main/reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#head-timescales-confirmation).
Fixed-reference dual остаётся контролем: head−fixed на новых четырёх seeds
составил +0.040%, 2/4 wins; head−shared +0.319%, 3/4, first27 mean delta=0.
