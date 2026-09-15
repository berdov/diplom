# RT-Mamba3: time-aware эксперимент

GPU numerical equivalence PASSED: job 4326256, A100, output и input gradients
имеют exact zero error во всех четырёх случаях. [Evidence](runs/gpu_equivalence_001.json).
Первый TRAIN->VALID run и единственный финальный TEST завершены;
`test_evaluation_count=1`. Повторный TEST и post-test tuning запрещены.

## Frozen reference

[Vanilla Mamba3](../mamba3_baseline/README.md): VALID NDCG@10 = **0.0584**,
TEST NDCG@10 = **0.0590**. TEST здесь только historical reference, не основание
для выбора новой модели. Все будущие решения и выбор модели исключительно по VALID.
Единичный VALID результат не доказывает статистически значимое превосходство
или novelty; никаких решений по TEST не принималось.

## Первый VALID результат

- [Smoke JSON](runs/mamba3_timeaware_smoke_001.json): job 4326761,
  cn-044/A100, COMPLETED 0:0, 00:02:18; два optimizer steps, finite gradients,
  calibrator update и checkpoint roundtrip PASS. Smoke-метрики не канонические.
- [Scientific JSON](runs/mamba3_timeaware_validation_001.json): job 4326765,
  cn-045/A100, COMPLETED 0:0, 00:07:22. 27 эпох (0..26), best epoch 15 (с нуля).
- Selection: full-ranking VALID NDCG@10; best **0.0605** против vanilla **0.0584**,
  абсолютная разница **+0.0021**, относительная **+3.60%** по округлённым метрикам.
- Код обоих запусков: `e81129a66c818a570ab589b4e255da08aac22422`,
  610506 параметров. TRAIN reference заранее frozen: 838393 мс, max_log_scale=log(2).

| VALID | @5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|
| HR | 0.0682 | 0.1111 | 0.1800 | 0.3204 |
| Recall | 0.0682 | 0.1111 | 0.1800 | 0.3204 |
| NDCG | 0.0468 | 0.0605 | 0.0778 | 0.1055 |

Best checkpoint остаётся на кластере по `checkpoint_path` из JSON; в Git не включён.
Финальные JSON скопированы без изменения и сверены по SHA256.
После этого checkpoint был зафиксирован для единственного TEST ниже.
Повторные runs и prototypes не запускались. `experiments/results.csv` не изменён.

## Единственный финальный TEST

[Финальный JSON](runs/mamba3_timeaware_final_test_001.json): job **4327348**,
rocky, cn-046, A100-SXM4-80GB, **COMPLETED 0:0**, elapsed **00:04:12**.
Full-ranking Protocol B, каталог 7111 items. Выбран только по VALID checkpoint
epoch 15, VALID NDCG@10 0.0605; нового обучения или повторного VALID не было.

| TEST | @5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|
| HR | 0.0683 | 0.1116 | 0.1764 | 0.3136 |
| Recall | 0.0683 | 0.1116 | 0.1764 | 0.3136 |
| NDCG | 0.0475 | 0.0613 | 0.0776 | 0.1046 |

| Reference | NDCG@10 | RT absolute delta | RT relative delta | HR@10 | RT absolute delta | RT relative delta |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla Mamba3 | 0.0590 | +0.0023 | +3.90% | 0.1062 | +0.0054 | +5.08% |
| TiM4Rec | 0.0598 | +0.0015 | +2.51% | 0.1053 | +0.0063 | +5.98% |

Сравнение по округлённым full-ranking TEST метрикам; статистическая значимость
по одному run не заявляется. Архитектура, конфигурация и checkpoint не менялись.
Checkpoint SHA256 до и после TEST:
`d8960963f8c94baa1229f803eec50fb194e645ef8d209c5fd88b4f6148f9a93d`.
Runner/launcher commit: `8065763be0f9bfceef6f0bbbd8e2ee53b7264b25`.
Pinned Mamba commit подтверждён runtime:
`e9594ce1c732d97440f0332fdc43170a2294dbfa`.

[TEST runner](mamba3_timeaware_final_test.py) загружает state_dict строго,
не создаёт optimizer и вызывает только TEST evaluation один раз.
[Launcher](../../slurm/mamba3_timeaware_final_test.sh) использует frozen environment.
Exclusive lock сохраняется после завершения; JSON публикуется через temporary
file, fsync и rename. Существующий результат блокирует повторный запуск.
Исторические VALID/smoke JSON сохраняют count=0 на момент тех запусков;
финальный TEST JSON фиксирует count=1. Raw logs и checkpoint не включены в Git.
В stderr только pandas FutureWarning из RecBole, без traceback.

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

## Конфигурация и запуск

[config.py](config.py) загружает frozen YAML и маленький
[overlay](config_kuairand.yaml), сохраняя Protocol B split/full-ranking и
baseline hyperparameters. Не передавайте overlay отдельно как полный RecBole config.
`time_scale_reference: 838393.0` мс зафиксирован как median строго положительных
gaps в TRAIN histories. [Статистика](runs/train_time_stats_001.json) получена
[отдельным скриптом](compute_train_time_stats.py) только из SHA-проверенного train.parquet.
Каждый gap учитывается столько раз, сколько встречается в TRAIN-префиксах длиной
до 50; интервал до target не учитывается. VALID/TEST для статистики не читались.

[Runner](run.py) переиспользует frozen verifier и RecBole Trainer: 300 epochs,
early stopping 10, Adam/lr/batch/seed без изменений, full-ranking VALID NDCG@10.
TEST dataset резервируется стандартным split, но удаляется без создания TEST loader.
`test_evaluation_count=0`; JSON и lock запрещают автоматический retry того же run ID.
Checkpoint сохраняется отдельно в игнорируемом slurm_logs, не в frozen baseline.

[Dataset adapter](dataset.py) сохраняет точные миллисекунды в `timestamp_list`:
обычный RecBole FloatTensor теряет младшие биты Unix ms. Исходное float32 поле
сортировки остаётся неизменным; дополнительное точное target поле удаляется после
построения histories. [CPU probe](precision_probe.py) проверяет совпадение item
histories, targets и split с обычным RecBole. Runner дополнительно проверяет
распределение gaps фактических TRAIN histories против frozen stats JSON.

[Launcher](../../slurm/mamba3_timeaware_validation.sh) принимает `smoke` или `train`.
Smoke: два TRAIN optimizer steps, первые 64 VALID histories с full-ranking,
checkpoint save/load; его метрики не являются scientific result.
Полный run был выполнен после smoke PASS. Training runner не выполняет TEST;
единственный финальный TEST выполнен отдельным runner после заморозки checkpoint.

## Проверки

Из корня repo, в CPU-окружении с PyTorch, pytest, PyYAML, einops, NumPy
(для расчёта статистики дополнительно PyArrow):

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

**На Mac не запускался; на A100 PASS (job 4326256).** Скрипт проверяет pinned package metadata, создаёт
official vanilla mixer и его независимую копию для time-aware forward, использует
одинаковые веса и входы, scale=1, bf16, train/eval, длины 50 и 64. Сравнивает
output и input gradients, печатает max/mean absolute error, tolerance и PASS/FAIL.
`atol=1e-6`, `rtol=1e-5`: при identity ожидается одинаковый порядок операций,
поэтому большой bf16 tolerance не используется. При расхождении нужно остановиться
и диагностировать, а не расширять tolerance. Нет CPU/fake kernel fallback.
GPU gate и smoke пройдены; первый scientific TRAIN->VALID завершён.
Единственный frozen TEST также завершён. Работа остановлена: никаких повторов,
новых seeds, ablations, prototypes или post-test tuning.
