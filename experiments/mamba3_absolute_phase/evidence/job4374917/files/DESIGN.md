# Абсолютная и относительная привязка фазового сигнала

Проверяем, даёт ли общее начало отсчёта времени преимущество перед временем
с начала доступной истории. В обоих вариантах используются одинаковые два
периода и одна и та же матрица параметров. Периоды и формулы ниже — дизайн
этого эксперимента, зафиксированный до просмотра качества.

Основное сравнение: `absolute_phase − relative_phase`. Оба варианта отдельно
сравниваются с `baseline_dual`. Один вариант с одним периодом не запускается,
поэтому преимущество нескольких периодов перед одним здесь не проверяется.

## Исходная математика

Проверены локальные файлы:

- `experiments/mamba3_three_time/model.py:ThreeTimeMamba3Rec.encode_sequence`:
  одинаковые gap-функции общие для двух последовательных слоёв;
- `experiments/mamba3_three_time/mixer.py:three_time_forward`:
  порядок проекций, content-dependent `A` и `DT`, общие углы для heads;
- `experiments/mamba3_three_time/calibrators.py:ThreeTimes.forward,split_dt`:
  в dual один и тот же scan-тензор передаётся в write и phase;
- `experiments/mamba3_three_time/kernels.py:MIMO.forward,MIMO.backward,mimo`:
  native `angle_dt`, MIMO forward/backward и раздельные аргументы DT;
- `experiments/mamba3_three_time/reference.py:intermediates,rotary,recurrence`:
  независимая маломасштабная recurrence на обычных PyTorch-операциях;
- `experiments/mamba3_timeaware/time_inputs.py:history_gaps,TimeCalibrator`:
  точные float64 интервалы, маска и прежнее ограничение scale;
- [условия оценки](../../reports/EVALUATION_SETUP.md): точные исторические
  timestamps, user-wise leave-one-out, history50 и full-ranking VALID.

Установленный на кластере upstream имеет pin
`e9594ce1c732d97440f0332fdc43170a2294dbfa`; это также указано в `direct_url.json`.
Прочитаны `mamba_ssm/modules/mamba3.py:Mamba3.forward,heavy_tail_activation`,
`ops/triton/mamba3/angle_dt.py:angle_dt_fwd_kernel,angle_dt_bwd_kernel`,
`ops/tilelang/mamba3/mamba3_mimo.py:_Mamba3Function.forward,backward`,
`mamba3_mimo_fwd.py:mamba_mimo_forward` и релевантные части
`mamba3_mimo_bwd.py:mamba_mimo_bwd_combined`.

Для входа mixer `u_i` линейная проекция возвращает `z,x,B,C,dd_dt,dd_A,Trap,raw_angle`.
Штатный шаг и коэффициент:

\[
d_i=\operatorname{softplus}(dd\_dt_i+dt\_bias),\qquad
A_i=\min\{-\operatorname{heavy\_tail}(dd\_A_i),-A_{floor}\}.
\]

`heavy_tail(x)` равна `1+x` при `x≥0` и `1/(1−x)` при `x<0`.
Оба значения зависят от содержимого текущего входа mixer. С прежними
калибраторами интервалов получаем:

\[
ADT_i=A_i(d_i s_{decay}(g_i)),\qquad
DT_{write,i}=d_i s_{scan}(g_i),\qquad
DT_{phase,i}=DT_{write,i}.
\]

Калибраторы получают `log1p(g/R0)`, `R0=838393 ms`, и возвращают
`exp(log(2)·tanh(MLP(...)))` с диапазоном `[0.5,2]`. На первом событии и
padding scale равен единице. Их веса продолжают обучаться во всех вариантах.

`angle_dt_fwd_kernel` сначала применяет native приближение `tanh(raw)·π`,
умножает на `DT_phase`, суммирует внутри chunk и добавляет накопленное
состояние. После chunk и на каждом выходе используется modulo `2π`:

\[
\Theta_i=\left[\sum_{j\leq i}\pi\tanh(raw\_angle_j)DT_{phase,j}\right]\bmod2\pi.
\]

Это схема математической операции: выполнение использует FP32,
`tanh_approx`, chunked cumsum и native backward с `sech2_approx`.
Backward суммирует cotangents в обратном порядке и возвращает градиенты
и raw angles, и `DT_phase`. В dual градиенты двух DT-потребителей сходятся
в исходный общий scan-тензор. FP64 reference не заменяет этот путь обучения.

Забывание имеет множитель `exp(ADT_i)`. Это не одна постоянная скорость:
`ADT_i` меняется вместе с событием, его представлением и интервалом.
Запись уже зависит от содержимого. Например, в схематической recurrence
`γ_i=DT_write,i·sigmoid(Trap_i)`, `β_i=DT_write,i·(1−sigmoid(Trap_i))`:

\[
S_i=e^{ADT_i}(S_{i-1}+\beta_i\widetilde K_{i-1}V_{i-1}^{T})
    +\gamma_i\widetilde K_iV_i^T.
\]

`K,V,Trap,d` получаются из проекций текущего содержимого; `K` дополнительно
нормируется, получает bias и вращение. MIMO использует прежние rank-проекции.
Форма записи в chunk-коде с shifted `γ_i+β_{i+1}` эквивалентна этой причинной
recurrence. Поэтому новый календарный вход в write не добавляется, но сама
write-ветвь никогда не была только функцией времени.

SHA256 прочитанных upstream-файлов:

| Файл относительно `mamba_ssm` | SHA256 |
|---|---|
| `modules/mamba3.py` | `930c3dfa04dea8444b1c9ee8b6ac9cbbc7ef492dc5ae1f4c83051ef953eca33c` |
| `ops/triton/mamba3/angle_dt.py` | `7d99f8cff37371cfbfe9d4dd36f17a16135fb07f2c6e14add3ad4dd5c0034e2e` |
| `ops/tilelang/mamba3/mamba3_mimo.py` | `ac691200e3d22ac9820931d83cc2843213ebcc4db156e44a45b2db90c3a0766a` |
| `ops/tilelang/mamba3/mamba3_mimo_fwd.py` | `1db4bc1e2f55be807c59954a82c3a9066913a13e43368c92a767f47c819d2047` |
| `ops/tilelang/mamba3/mamba3_mimo_bwd.py` | `f35e610f6c5c6fdf8dc72dae51df26b95605deb38625f13ae6d8cad9c32c229f` |

## Новый вход только в raw angles

`baseline_dual` напрямую вызывает прежний
`ThreeTimeMamba3Rec.encode_sequence`. Нового параметра или дополнительной
случайной инициализации в нём нет.

`relative_phase` получает `c_i=cumsum(history_gaps)_i`. После усечения начало
отсчёта переносится на первое событие текущего окна. При отсортированной
истории это `τ_i−τ_0`. Прежний clamp отрицательного adjacent gap к нулю
сохранён: для несортированной истории равенство разности времени уже
не обязательно. Этот случай явно проверяется тестом, а не исправляется
скрытой пересортировкой.

`absolute_phase` получает точный исторический `τ_i` в миллисекундах Unix epoch.
Ни batch minimum, ни пользовательское начало отсчёта не вычитаются. Модель
не получает timezone пользователя, поэтому признаки нельзя трактовать
как его локальный час или личный распорядок.

Оба backbone продолжают получать исходные inter-event gaps.
`target timestamp`, `t_score`, следующее событие и user embedding отсутствуют.

Для `P=(21600000,86400000) ms` вычисляем

\[
\phi(c)=\frac{1}{\sqrt2}
[\sin(2\pi(c\bmod P_1)/P_1),\cos(2\pi(c\bmod P_1)/P_1),
 \sin(2\pi(c\bmod P_2)/P_2),\cos(2\pi(c\bmod P_2)/P_2)].
\]

`phase.py:phase_clock,features_from_clock,periodic_features` выполняют
sanitization padding, remainder и sin/cos в float64; только готовые features
переводятся в dtype W. Все valid timestamps обязаны быть конечны.
Unix timestamps не переводятся во float32 до remainder.

`phase.py:PeriodicPhase` создаёт единственную FP32 матрицу `W[A,4]` через
`nn.Parameter(torch.zeros(...))`, без bias, второго gate и расхода RNG.
Один экземпляр модуля используется для обоих слоёв, одна поправка
распространяется на обе heads:

\[
\delta_i=\tanh(W\phi(c_i)),\qquad
raw'_i=raw_i+active_i\delta_i.
\]

`active[0]=False`, `active[i]=valid[i]&valid[i−1]`. Padding и первое событие
нейтральны, настоящий нулевой gap остаётся активным. Функция
`phase_correction` реализует именно эту операцию; отдельный
`reference_correction` вычисляет её scalar FP64 oracle без вызова production
clock, features или linear. CPU gradcheck проверяет production operation.

`mixer.py:project_inputs,phase_forward` сохраняют прежний порядок проекций,
`ADT`, оба DT, Trap, V, B/C до вращений, biases и нормировки. Меняется только
raw-angle input до native `angle_dt`. В следующем слое вход уже может
измениться из-за выхода предыдущего; это ожидаемое следствие phase-ветви.

MIMO raw angles после broadcast и новая поправка имеют FP32 dtype.
`length_adapter.prepare` дополняет хвост до chunk8 нулями без BF16 cast углов.
В TileLang внутренние проекции/вращённые Q,K используют прежнюю mixed
precision. Измерение BF16 cast самой поправки в диагностике — отдельная
гипотетическая потеря точности, а не утверждение, что такой cast есть в
production routing. Kernels и их прежние ограничения не меняются.

`A` извлекается из реального `mixer.num_rope_angles`. При state128 и
rope_fraction0.5 получается 32: это проверяется созданием моделей вместе
с counts `715020 / 715148 / 715148`. Не используется новый DT multiplier,
triple, head/layer-specific параметризация или изменение recurrence.

Периоды относятся к features. Они не задают точный период вращения всего
состояния: итоговая фаза зависит от content angles, `DT_phase`, W и истории.

## Проверка календарной чувствительности

Само наличие timestamp не доказывает эффект: разность линейных фаз может
сократиться до gap. До обучения фиксируется информативная история из восьми
событий, ненулевая `W=0.8·sin(arange(A·4)·0.37+0.2)` и синтетический
`dt_bias=inverse_softplus(0.25)`. Это только gate-fixture, не training settings.

Сдвиг всех valid timestamps на два часа сохраняет gaps и должен сохранять
выход baseline/relative. В absolute проверяются изменение features,
поправки, native фазовых тензоров и конечного выхода. Сдвиг на сутки
возвращает оба периода. Отдельно проверяются обе частотные пары, W-gradient,
общая W, first/padding neutrality, zero-gap, W0 identity и исходные tolerances.
Для length1 влияние фазы не требуется: same-position rotation сокращается.

Observer получает native inputs без изменения их значений. Он не делает
второй forward и не расходует training RNG. Baseline observer-путь используется
только в техническом тесте и отдельно сравнивается с точным default-путём.
Новый независимый recurrence test использует малые synthetic tensors;
scientific runner всегда вызывает native MIMO. Научные результаты нельзя
получать CPU/reference путём.

Дополнительные проверки: output/loss/common gradients и первый общий Adam
update при W0, exact init/RNG, строгий state mapping, roundtrip weights_only,
suffix item/timestamp interventions, cross-user isolation, отсутствие target
time, L7/8/9/17/50/65, finite large timestamps, gradient hooks cleanup.
Прежние structural tolerances и residual policy сохраняются; исторический
exact-zero FAIL не переименовывается в PASS.

## Границы вывода

Coverage считается только по TRAIN histories без model forward. Нужны
диапазон точных timestamps, число покрытых периодов, фазы по bins,
число active событий, повторение фаз в разные даты, timestamp ties и span
историй. Даже хорошее покрытие не доказывает поведенческую периодичность.

VALID callback сохраняет W, вклады обеих frequency-пар, поправки и изменение
`π·tanh(raw)·DT_phase` на заранее заданных grids и доступных существующих
forward tensors. Большая норма W и ненулевой вклад 24h не доказывают важность
времени или привычку с периодом сутки. Отсутствующие измерения отмечаются
`NOT_RECORDED`.

Пилот содержит три fresh fit seed2026. Confirmation разрешён только по
заранее записанному правилу: все три fit завершены и парны, absolute VALID
NDCG@10 не ниже обоих контролей. Равенство округлённых метрик не считается
улучшением. Дальнейшие seeds2027–2030 проверяют ту же frozen конструкцию;
пилот не включается в основной confirmation aggregate. Периоды, амплитуда
и архитектура по результату не меняются. TEST не используется.

Следующие механизмы описаны отдельно в [NEW_PLAN.md](NEW_PLAN.md). Их код
и GPU-эксперименты не входят в этот этап.
