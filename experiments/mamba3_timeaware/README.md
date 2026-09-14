# RT-Mamba3: time-aware эксперимент

GPU numerical equivalence PASSED: job 4326256, A100, output и input gradients
имеют exact zero error во всех четырёх случаях. [Evidence](runs/gpu_equivalence_001.json).
Научное обучение RT-Mamba3 ещё не запускалось.

## Frozen reference

[Vanilla Mamba3](../mamba3_baseline/README.md): VALID NDCG@10 = **0.0584**,
TEST NDCG@10 = **0.0590**. TEST здесь только historical reference, не основание
для выбора новой модели. Все будущие решения и выбор модели исключительно по VALID.
Результатов RT-Mamba3 нет; доказанная novelty или превосходство не заявляются.

## Единственное архитектурное изменение

Основа: [state-spaces/mamba](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py),
commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`.
[Лицензия upstream](LICENSE.upstream) сохранена рядом с адаптацией.

В [time_mamba3.py](time_mamba3.py), `time_mamba3_forward`, после официального
`softplus(dd_dt + dt_bias)` и перед расчётом ADT:

```text
DT_base = softplus(dd_dt + dt_bias)
DT_real = DT_base * time_scale(delta_t)
ADT_real = A * DT_real
```

`A` остаётся официальной heavy-tail activation с исходным clamp.
Те же `DT_real` и `ADT_real` передаются официальному `mamba3_siso_combined`;
именно `DT_real` участвует и в recurrence, и в rotary accumulation внутри kernel.
Проекции, B/C RMSNorm, trap, angles, gating, D и output projection не изменены.
Поддержан только обычный SISO full-sequence путь без cache/step/packed sequences.
Установленный `mamba_ssm` не редактируется; это локальная адаптация forward.

[Модель](model.py) наследует frozen recommender и переиспользует его embeddings,
два mixer, residual/FFN и tied scoring. Item sequence не меняется.
Модификация native DT применяется одной функцией к каждому mixer;
один shared calibrator вычисляется один раз на последовательность для обоих layers.
Нет MIMO, auxiliary tasks, MTL, MoE, календарных features или TimeEmbedding control.

## Время и формы

API принимает только `history_timestamps [B,L]`, соответствующие `item_seq`;
target timestamp не запрашивается. В RecBole читается только `TIME_FIELD + LIST_SUFFIX`.
Caller обязан передавать историю без target event. Модель проверяет непустое
right-padding и согласованность `item_seq_len` с item mask.

`delta_t[first_valid]=0`, остальные соседние исторические gaps:
`max(t[i]-t[i-1],0)`. Padding исключён до subtraction; нечисловое время у valid
events отклоняется. Расчёт в float64 сохраняет миллисекундную точность обычных
Unix timestamps. Никакие будущие позиции не участвуют в gap текущей позиции.

`tau = log1p(delta_t / time_scale_reference)` вычисляется эквивалентно через
`logaddexp` в float64, чтобы не переполнить отношение при огромных gaps.

```text
tau [B,L,1] -> Linear(1,16) -> SiLU -> Linear(16,H)
log_scale = max_log_scale * tanh(raw)
time_scale = exp(log_scale)                         [B,L,H]
DT_base, A, DT_real, ADT_real                       [B,L,H]
DT_real, ADT_real на входе kernel                  [B,H,L]
Q, K                                              [B,L,1,128]
V, Z                                              [B,L,2,64]
Angles                                            [B,L,2,32]
```

В frozen конфигурации `H = 64 * 2 / 64 = 2`. Ошибочные head shapes отклоняются,
не исправляются broadcasting. Последний Linear инициализирован нулями:
scale точно 1. Первый valid event и padding принудительно получают scale=1
даже после обучения calibrator. Default `max_log_scale=log(2)` ограничивает scale
интервалом `[0.5,2]`: консервативный initial bound, не результат tuning.
При zero-init градиент первого Linear на первом backward нулевой математически;
последний Linear получает ненулевой градиент, после изменения его весов градиент
достигает первого. Это проверено отдельно.

## Конфигурация и будущий запуск

[config.py](config.py) загружает frozen YAML и маленький
[overlay](config_kuairand.yaml), сохраняя Protocol B split/full-ranking и
baseline hyperparameters. Не передавайте overlay отдельно как полный RecBole config.
`time_scale_reference: null` намеренно блокирует создание calibrator до передачи
положительного значения; источник должен быть `TRAIN`, единицы те же, что timestamps.
Статистика по dataset здесь не вычислялась. Позже потребуется TRAIN-only statistic
с provenance; строка `TRAIN` сама по себе не доказывает происхождение числа.

Runner намеренно отсутствует: никакие loaders, preprocessing, VALID/TEST evaluation
или Slurm launchers не добавлены. Для будущего runner нужно сохранить проверку
Protocol B из frozen `run.py`, не вызывая его main и не используя TEST loader.
Новый checkpoint_dir отделён от frozen baseline.

## Проверки

Из корня repo, в CPU-окружении с PyTorch, pytest, PyYAML, einops:

```bash
python -m compileall -q experiments/mamba3_timeaware
python -m pytest experiments/mamba3_timeaware/tests -q
git diff --check
```

Тесты охватывают identity, bounds, огромные gaps, отрицательные/нечисловые inputs,
padding, первый event, prefix causality, DT/ADT consistency, head shapes,
градиенты, config/parser, независимый от CUDA import и статический API модели.
Статическая проверка item input не является end-to-end equivalence recommender.
Импорт полной модели требует официального CUDA stack и RecBole и пропускается,
если этих зависимостей нет; они не подменяются заглушками.

Отдельный [gpu_equivalence.py](gpu_equivalence.py) подготовлен для A100:

```bash
python -m experiments.mamba3_timeaware.gpu_equivalence
```

**На Mac не запускался.** Скрипт проверяет pinned package metadata, создаёт
official vanilla mixer и его независимую копию для time-aware forward, использует
одинаковые веса и входы, scale=1, bf16, train/eval, длины 50 и 64. Сравнивает
output и input gradients, печатает max/mean absolute error, tolerance и PASS/FAIL.
`atol=1e-6`, `rtol=1e-5`: при identity ожидается одинаковый порядок операций,
поэтому большой bf16 tolerance не используется. При расхождении нужно остановиться
и диагностировать, а не расширять tolerance. Нет CPU/fake kernel fallback.
До успешного GPU gate переход к экспериментам не подтверждён.
