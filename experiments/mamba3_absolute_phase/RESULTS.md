# Абсолютная и относительная фаза

**Пилот COMPLETE: три свежих парных обучения seed 2026 и независимый аудит
завершены.** Absolute получил VALID NDCG@10 **0.0635**, relative — **0.0616**,
исходный MIMO dual — **0.0633**. Основной контраст absolute−relative равен
**+0.0019**; преимущество над рабочей моделью — **+0.0002**. Это предварительный
результат одного seed. Условие продолжения выполнено; confirmation находится
на этапе подготовки и пока не отправлена. Оперативное состояние — в
[HANDOFF](runtime/HANDOFF.md).

## Конструкция и протокол

Проверяется общая календарная привязка фазовой поправки относительно отсчёта
от начала доступной истории. В обоих новых вариантах одна FP32-матрица
`W[32,4]` общая для двух слоёв и двух heads. Она добавляет 128 параметров.
Две пары sin/cos имеют заранее выбранные периоды 6 и 24 часа и общий множитель
`1/sqrt(2)`. Remainder и тригонометрия вычисляются в FP64 до приведения
признаков к FP32. Эти периоды не подбирались по VALID.

| Вариант | Часы нового модуля | Параметры |
|---|---|---:|
| baseline_dual | нового модуля нет | 715020 |
| relative_phase | сумма реальных gaps от первого события текущего окна | 715148 |
| absolute_phase | точный timestamp наблюдаемого события относительно Unix epoch | 715148 |

Поправка `delta_raw = tanh(W·phi(c))` прибавляется к raw angles перед штатным
`tanh·pi` и накоплением фазы. `W` инициализируется точными нулями без новых
случайных чисел. Первое событие и padding нейтральны; настоящий zero-gap
остаётся активным. Backbone в обоих вариантах получает прежние реальные
inter-event gaps. Decay/write сохраняют параметризацию и продолжают обучаться;
их веса не заморожены. `DT_write` и `DT_phase` связаны как в dual.
Новый сигнал не входит в embeddings, FFN, scorer или recurrence обходным путём.

MIMO rank4/chunk8, два слоя, две temporal heads, history50 и фиксированный
TRAIN reference `R0=838393 ms` сохранены. Обучение: прежний KuaiRand protocol B,
CE, Adam 0.001, batch 2048, максимум 300 эпох, stopping_step 10, full-ranking
VALID по 7111 items и прежняя обработка seen/repeated targets. Выбор лучшей
эпохи — last-tie по сохранённому VALID NDCG@10. Каждый fit запущен с нуля в
отдельном процессе; common initialization, RNG, optimizer, данные, precision
и первый фактически потреблённый batch проверены на парность. **TEST=0.**

При усечении истории relative origin переносится на начало текущего окна.
Модель не получает timezone пользователя. Периоды признаков не равны точным
периодам вращения состояния: итог зависит от content angles, `DT_phase` и `W`.
Формулы и места подключения — в [DESIGN](DESIGN.md), исходные настройки — в
[сохранённом плане пилота](evidence/job4374917/files/study_plan.json).

<a id="absolute-phase-pilot"></a>
## Результаты пилота

Все значения ниже взяты из сохранённых raw JSON и проверены против process
и RecBole logs. Эпохи нумеруются с нуля. First27 — максимум строго на эпохах
0–26; отсутствие этих 27 эпох не заменяется full-run best.

| Вариант | NDCG@10 | HR@10 | Лучшая эпоха | Всего эпох | First27 complete | Best first27 |
|---|---:|---:|---:|---:|---|---:|
| baseline_dual | 0.0633 | 0.1162 | 27 | 39 | да | 0.0620 |
| relative_phase | 0.0616 | 0.1124 | 11 | 23 | нет | N/A |
| absolute_phase | 0.0635 | 0.1174 | 54 | 66 | да | 0.0620 |

| Контраст | Delta NDCG@10 | Относительно comparator | Delta first27 |
|---|---:|---:|---:|
| **absolute−relative, primary** | **+0.0019** | **+3.0844%** | N/A |
| absolute−baseline | +0.0002 | +0.3160% | 0.0000 |
| relative−baseline | −0.0017 | −2.6856% | N/A |

Разности представлены в точности исходных метрик — четыре знака.
Относительный процент равен `100·(left−right)/right`. First27 использует те же
histories; это не независимое подтверждение и не одинаковый бюджет вычислений.
Relative завершился на 23 эпохах, поэтому оба first27-контраста с ним отсутствуют.

Все 12 метрик выбранных VALID checkpoints:

| Метрика | baseline_dual | relative_phase | absolute_phase |
|---|---:|---:|---:|
| HR@5 | 0.0712 | 0.0689 | 0.0715 |
| HR@10 | 0.1162 | 0.1124 | 0.1174 |
| HR@20 | 0.1865 | 0.1796 | 0.1886 |
| HR@50 | 0.3338 | 0.3236 | 0.3375 |
| Recall@5 | 0.0712 | 0.0689 | 0.0715 |
| Recall@10 | 0.1162 | 0.1124 | 0.1174 |
| Recall@20 | 0.1865 | 0.1796 | 0.1886 |
| Recall@50 | 0.3338 | 0.3236 | 0.3375 |
| NDCG@5 | 0.0488 | 0.0476 | 0.0489 |
| NDCG@10 | 0.0633 | 0.0616 | 0.0635 |
| NDCG@20 | 0.0809 | 0.0784 | 0.0814 |
| NDCG@50 | 0.1100 | 0.1069 | 0.1108 |

| Вариант | TRAIN, с | VALID, с | Peak allocated, bytes | Peak reserved, bytes |
|---|---:|---:|---:|---:|
| baseline_dual | 928.1002 | 51.6827 | 3024827392 | 4076863488 |
| relative_phase | 564.8713 | 18.0784 | 3041423872 | 4250927104 |
| absolute_phase | 1591.3216 | 77.1746 | 3041423872 | 4250927104 |

Пики allocated/reserved составляют 2.8171/3.7969 GiB у baseline и
2.8325/3.9590 GiB у обоих новых вариантов. Это измерения полных запусков
разной длины. VALID включает предусмотренный сбор диагностики; прогрев и
кеши между процессами тоже различаются. Таблица не является чистым benchmark
скорости inference. Absolute обучался 66 эпох вместо 39 у baseline.

## Покрытие TRAIN

[TRAIN-only аудит](evidence/job4374917/files/slurm_logs/pilot/attempt_001/train_coverage.json)
выполнен до submit на 10000 детерминированно выбранных входных окнах.
Статистики не читают targets и не создают DataLoader или модель. Подготовка
прежнего Dataset описывает splits; для подсчётов читаются только TRAIN
item histories, lengths и точные historical timestamps.

- 301109 вхождений валидных событий, из них 291109 активны для поправки.
  Окна пересекаются; это не число уникальных взаимодействий или пользователей.
- Timestamp range: `1649475963278..1650546478532 ms`, то есть
  9 апреля 2022, 03:46:03.278 UTC — 21 апреля 2022, 13:07:58.532 UTC.
  Размах — 12.3902 суток: 49.5609 шестичасовых и 12.3902 суточных периодов.
- Для абсолютных timestamps заняты все 24 фазовых bins каждого периода,
  для valid и active событий.
  Каждый bin встречается на 12–13 UTC-датах. Точные фазы повторяются на
  разных датах: 1340 значений для 6 часов и 368 для 24 часов.
- 27437 соседних пар имеют одинаковый timestamp: 9.425% активных пар.
  Они сохранены активными. Отрицательных adjacent gaps в выборке нет.
- Размах окна: среднее 2.3847 суток, точная медиана выборки 1.7885 суток,
  максимум 11.8819 суток. Средняя длина — 30.1109 события.

Семантические условия были зафиксированы до просмотра качества: конечные
точно представленные целые Unix milliseconds, положительные timestamps,
активные события с непостоянными фазами, диапазон не менее 24 часов и не менее
двух UTC-дат. Порог заполнения bins не подбирался. Покрытие достаточно для
этой конструкции, но не доказывает поведенческие циклы. Распределение по
датам неравномерно: 84.6258% вхождений приходятся на первые четыре UTC-даты.
Идентичность input hashes с прежней фиксированной TRAIN-выборкой подтверждена.

## Технические проверки и чувствительность к сдвигу часов

CPU preflight **100/100 PASS**, отдельный CPU запуск без Git **100/100 PASS**.
JSON точной execution-версии: [CPU](evidence/job4374917/files/slurm_logs/pilot/attempt_001/cpu_preflight_11913ff8fe99aa28f124d667e2ae3aad14465098.json)
и [no-Git CPU](evidence/job4374917/files/slurm_logs/pilot/attempt_001/no_git_preflight_11913ff8fe99aa28f124d667e2ae3aad14465098.json).
Прежние **45 cases / 2342 checks** kernel admission наследуются по проверенным
SHA. Собственный [targeted GPU gate](evidence/job4374917/files/runs/pilot/attempt_001/targeted_gate.json):
**17 cases / 532 required leaves, PASS**. [Smoke](evidence/job4374917/files/runs/pilot/attempt_001/smoke.json):
три режима по три optimizer steps, PASS.

Проверены baseline parity, W0 identity по outputs/loss/common gradients и
первому Adam update, RNG, фактические counts, обе частотные ветви, FP64
reference и gradcheck новой операции, state_dict roundtrip, first/padding
neutrality, active zero-gap, causality/cross-user interventions, отсутствие
целевого timestamp, границы chunk 7/8/9 и длины 17/50/65, большие timestamps,
снятие hooks и измерение counterfactual BF16 cast. Реальные callbacks и
сериализованные required leaves проверены независимо.

Календарная чувствительность проверена на заранее определённой информативной
истории длины 8 с ненулевой `W = 0.8*sin(arange(128)*0.37+0.2)` и synthetic
`dt_bias = inverse_softplus(0.25)`. Это gate fixture, не настройка scientific
fits. При общем сдвиге timestamps items, lengths и реальные gaps неизменны.

| Вариант | +2h: max abs feature change | +2h: max abs correction change | +2h: max abs output change | +24h: max abs output change |
|---|---:|---:|---:|---:|
| baseline_dual | N/A | 0 | 0 | 0 |
| relative_phase | 0 | 0 | 0 | 0 |
| absolute_phase | 1.22285992 | 1.12875327 | 0.30550625 | 0 |

Нулевые output differences здесь побитовые. У absolute различие на +2h
выходит за frozen invariance tolerance; `W` получает ненулевой gradient
(norm `0.0008773113`; у relative `0.0009755500`). При +24h признаки и поправка
повторяются побитово. При W=0 все три модели инвариантны к +2h и +24h.
Входы одного mixer совпадают по всем нефазовым тензорам; общий сигнал
действительно изменяет только raw-angle input. Возможные изменения входов
следующего слоя ожидаемы. Числовые timestamps, features, phase tensors,
outputs и SHA сохранены в gate JSON; независимый аудит перепроверил
88 packed tensors, три shift cases и два native-reference comparisons.

Это показывает, что на информативном fixture абсолютный сигнал не сократился
до одного gap. Это не утверждение о произвольной истории, length1 или
поведенческой периодичности пользователей.

## Диагностика обученной фазы

Таблицы относятся к выбранным лучшим эпохам: 11 у relative, 54 у absolute.
Observed statistics собирались только в существующих VALID forwards:
23951 последняя активная позиция истории, шесть forwards на слой.
Это не статистики всех 724401 активных позиций окон. `W` и поправка общие
для слоёв; content angles и `DT_phase` на их входах различаются.

| Показатель | relative_phase | absolute_phase |
|---|---:|---:|
| Frobenius norm W | 3.997393 | 7.553420 |
| Norm пары W: 6h | 1.261896 | 1.811968 |
| Norm пары W: 24h | 3.792989 | 7.332866 |
| Max abs W | 1.771049 | 2.108739 |
| RMS линейного вклада 6h на равномерной сетке суток | 0.111537 | 0.160157 |
| RMS линейного вклада 24h на равномерной сетке суток | 0.335256 | 0.648140 |
| RMS raw-angle correction на VALID | 0.309911 | 0.500517 |
| Max abs raw-angle correction на VALID | 0.884321 | 0.956014 |
| RMS изменения local native increment, слой 0 | 0.00461920 | 0.01582115 |
| RMS изменения local native increment, слой 1 | 0.03770532 | 0.17831915 |
| Последний доступный gradient norm W на выбранной эпохе | 0.00177664 | 0.00578919 |

Вклад пары — `W_pair·phi_pair` до общего tanh; после tanh вклады не
аддитивны. Local increment рассчитывается как `pi*tanh(raw_angle)*DT_phase`
на сохранённом входе. Это не повторный recurrence forward и не вся
накопленная `Theta`.

На лучших эпохах и на всех сохранённых эпохах доля `abs(delta_raw)>=0.99`
равна нулю как в observed statistics, так и на фиксированных аналитических
сетках. Это относится к ограниченной поправке, а не к насыщению native tanh
для content angles. Все 11937 TRAIN backward callbacks у relative и 34254
у absolute дали конечный ненулевой gradient W; hooks снимались до VALID.
На выбранных эпохах все 128 координат последнего доступного gradient
ненулевые. Final W norms — 4.973961 и 7.681981.

Сетки фиксированы до fit: часы 0–23 для absolute и `0,1,3,6,12,24,48`
для relative. Дополнительные `(content raw angle, DT)` fixtures:
`(-2,0.05), (-0.5,0.25), (0,0.5), (0.5,1), (2,2)`.
Сохранены полная W, полные feature/correction grids, статистики по всем
32 углам и примеры increment differences для первых четырёх координат.
Дополнительных dataset evaluations или model forwards для них не было.

**BF16 ниже — counterfactual приведение raw angles. В реальном MIMO raw
angles остаются FP32.**

| Counterfactual показатель | relative_phase | absolute_phase |
|---|---:|---:|
| Ненулевая поправка исчезла после BF16 cast, слой 0 | 1.299659% | 0.365590% |
| То же, слой 1 | 0.734312% | 0.328274% |
| RMS ошибки local increment от такого cast, слой 0 | 1.35375e−5 | 2.95637e−5 |
| То же, слой 1 | 1.13613e−4 | 3.09357e−4 |

Эти числа не измеряют фактическое downstream округление повёрнутых Q/K.
Норма W, вклад пары 24h и величина поправки не являются feature importance.
Ненулевой суточный сигнал не доказывает личную привычку с периодом 24 часа.

## Выполнение, сохранение и аудит

Job **4374917**, `COMPLETED 0:0`, A100-SXM4-80GB, `cn-043`.
4 октября 2026: **10:56:12–12:13:48 MSK**, elapsed **01:17:36**.
Порядок: `baseline_dual2026 → relative_phase2026 → absolute_phase2026`.
Запрошена одна allocation до 6 часов; три scientific fits начаты и завершены,
unknown starts — 0, retry не использован, TEST — 0.

Pilot execution: `11913ff8fe99aa28f124d667e2ae3aad14465098`.
Pilot source hash: `4c404906e61121f967307e8c3e97a63e9a476711202a4a5baa005ef9899574c6`.
[Сохранённый manifest](evidence/job4374917/files/source_manifest.json): 447 файлов.
Raw evidence и PASS-аудит сохранены commit `f3aad61` до публикации выводов.

[Независимый audit](evidence/job4374917/independent_audit.json) имеет PASS:
проверены **128 эпох / 1536 metric cells**, все три fit, pairing, execution,
reservation/source/runtime, обязательные leaves gate, smoke, обе копии logs,
last-tie/early stopping, first27, лучшие diagnostics, checkpoint metadata и
streaming SHA. Fresh baseline полностью воспроизвёл исторический seed 2026,
включая историю и checkpoint SHA; это не новый независимый seed прежней серии.
SHA256 audit: `19daa0ed41947a10ca327f5c575cff12dba6ac8354cb7433dbeac562f6205399`.
В [реестр](../results.csv) добавлены ровно три индивидуальные строки:
**125 → 128**. Исходные 58287 bytes, включая header и прежние 125 записей,
сохранены побайтно; их SHA256
`30ef3c9c58c90662367fe21de5fcc8d71315d6d475564ec0c6d9b1adb8053c63`.
Aggregate rows не добавлялись.

[Preservation manifest](evidence/job4374917/preservation_manifest.json) связывает
58 компактных файлов объёмом 60688161 bytes; raw bytes не изменены.
Транспортный лимит одного JSON пришлось увеличить с 20 до 100 MB для
сохранения 35.2 MB result. Изменён только вспомогательный transport вне
frozen scientific source; fit не повторялся. [Scheduler evidence](evidence/job4374917/scheduler_terminal.json)
сохранён отдельно. Исходные необработанные данные:
[baseline](evidence/job4374917/files/runs/pilot/attempt_001/mamba3_absolute_phase_baseline_dual_seed2026_001.json),
[relative](evidence/job4374917/files/runs/pilot/attempt_001/mamba3_absolute_phase_relative_phase_seed2026_001.json),
[absolute](evidence/job4374917/files/runs/pilot/attempt_001/mamba3_absolute_phase_absolute_phase_seed2026_001.json),
[summary](evidence/job4374917/files/runs/pilot/attempt_001/pilot_summary.json).

Checkpoints остались на кластере. Общая директория:
`/home/daryumin/iberdov/diplom/experiments/mamba3_absolute_phase/slurm_logs/pilot/attempt_001/`.
Путь каждого файла от неё —
`mamba3_absolute_phase_<mode>_seed2026_001/checkpoints/best_state_dict.pth`.

| Mode | Bytes | SHA256 | Сохранённые metadata |
|---|---:|---|---|
| baseline_dual | 2507332 | `cba3da6aa7daf883cf9bcddb1b92540d48c26c62d797f8c9a1b602bcea292db8` | [metadata](evidence/job4374917/files/slurm_logs/pilot/attempt_001/mamba3_absolute_phase_baseline_dual_seed2026_001/checkpoints/best_metadata.json) |
| relative_phase | 2508161 | `0b46b40c047a500d774b37d611e1dbae6eebdce35d5242b4394e4b6615a764ae` | [metadata](evidence/job4374917/files/slurm_logs/pilot/attempt_001/mamba3_absolute_phase_relative_phase_seed2026_001/checkpoints/best_metadata.json) |
| absolute_phase | 2508161 | `20e6f86e1b075da7ca3c0bebe534c760d16e2881b300c3a7bd323e718a12bbab` | [metadata](evidence/job4374917/files/slurm_logs/pilot/attempt_001/mamba3_absolute_phase_absolute_phase_seed2026_001/checkpoints/best_metadata.json) |

Weights не копировались в Git и не десериализовались для аудита или новой
оценки. SHA и bytes вычислены потоком на кластере при сохранении и связаны
с metadata выбранной эпохи.

Границы независимой проверки:

- Исходные TRAIN histories не сохранены целиком: проверены индексы, hashes,
  identity и арифметика coverage. Multiplicities timestamps и span quantiles
  нельзя заново получить только из компактного JSON.
- Восемь VALID примеров на слой позволяют проверить локальную арифметику,
  но не восстановить все query aggregates. Сохранённые gradient norms не
  заменяют исходные gradient tensors; backward не повторялся.
- W и аналитические grids доступны полностью. Различия порядка редукции,
  FP32 matmul/libm и производных increment calculations записаны отдельно
  в `diagnostic_arithmetic`; это не расхождения научных метрик. Frozen GPU
  tolerances не расширялись. Все метрики и их соответствие logs проверены точно.
- Накопленная Theta и recurrence заново не вычислялись. Сам отдельный no-Git
  CPU JSON не содержит deny-git environment; это proof отдельного invocation,
  а не возможность восстановить среду из одного файла.

## Вывод по пилоту

В рамках одного seed абсолютная привязка лучше относительной при одинаковом
числе параметров и одинаковых двух периодах. Основная delta складывается
преимущественно из ухудшения relative относительно backbone: −0.0017, тогда
как собственный выигрыш absolute составляет +0.0002 — два шага сохранённой
четырёхзначной точности метрики. На первых 27 эпохах absolute равен baseline,
а полный fit дольше. Рабочую модель по одному такому результату не заменяем.

Пилот даёт основание выполнить заранее разрешённую confirmation. Он не
доказывает устойчивость эффекта, статистическую значимость, причинную динамику
интересов или найденные поведенческие циклы. Single-period comparator здесь
отсутствует, поэтому преимущества двух периодов над одним не установлены.
Пользовательская timezone, TEST и внешний holdout не использовались.

<a id="absolute-phase-confirmation"></a>
## Confirmation: разрешена, подготовка

Frozen правило выполнено: три fit технически завершены, pairing и
независимый аудит PASS, `0.0635 >= 0.0633` и `0.0635 >= 0.0616`.
На момент завершения этого отчёта confirmation job ещё не отправлена.
Пилот остаётся отдельным завершённым этапом.

Предусмотрены четыре новые тройки, seeds **2027, 2028, 2029, 2030**;
на каждом seed порядок `baseline_dual → relative_phase → absolute_phase`.
Периоды, bound, sharing, features, precision и обучение остаются прежними.
Основной aggregate должен включать только эти четыре seed: mean ± sample std
(ddof=1), paired deltas, +/−/0 и difference of means на одном наборе seed.
Отдельная таблица всех пяти seed будет exploratory, с явно включённым пилотом.
Новых post-hoc p-values не предусмотрено; sample std не является confidence
interval, а равенство округлённых метрик не означает улучшения или эквивалентности.

До отправки требуется закончить техническую подготовку: отдельная CPU-проверка
confirmation выявила ошибку test fixtures, которые создавали seed 2026 при
активном наборе 2027–2030. [Первичный FAIL](evidence/confirmation_cpu_scope_failure/)
сохранён до результатов пилота. Исправляется только обвязка и её тесты;
новая execution/source identity и повторная CPU-проверка должны быть связаны
с исходной frozen конструкцией. Это не повтор scientific fit и не подбор
метода после просмотра качества. Пилот не перезапускался.

Следующие механизмы описаны в [NEW_PLAN](NEW_PLAN.md). Многомасштабное забывание,
surprise-write, анализ native SSM state и low-rank subspaces в этом этапе
не реализовывались. Статья, Overleaf и draw.io не менялись; сообщения не отправлялись.
