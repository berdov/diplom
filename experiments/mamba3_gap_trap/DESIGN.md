# Gap-dependent Trap: pilot seed2026

Пункт 2 завершён: fixed-reference dual остаётся контролем. Здесь сравниваются
только `fixed_replay` и `gap_trap`, по одному свежему fit seed2026, без TEST.
Общий backbone: MIMO rank4/chunk8, 2 heads, 2 layers, history50.

## A. Штатный Trap

Upstream зафиксирован на
`e9594ce1c732d97440f0332fdc43170a2294dbfa`.
Пути ниже относительно пакета `mamba_ssm`:

- `modules/mamba3.py:176–191`: проекция выдаёт raw Trap logits.
- `ops/tilelang/mamba3/mamba3_mimo.py:279`: контракт Trap — pre-sigmoid `[B,H,L]`.
- `ops/tilelang/mamba3/mamba3_mimo_fwd.py:196–218`:
  `gamma = DT * sigmoid(Trap)`,
  `shifted_gamma[t] = DT[t+1] * sigmoid(-Trap[t+1])`,
  `factor = gamma + shifted_gamma`.
- Там же `:303–321`: factor применяется строго ниже диагонали;
  `:340–346`: диагональ использует только gamma;
  `:436–457`: перенос состояния между chunk.
- `ops/triton/mamba3/mamba3_mimo_utils.py:303–336`: производная Trap,
  включая границу chunk. В docstring на строке 252 ошибочно указан положительный
  знак у shifted sigmoid; исполняемый forward и backward используют отрицательный.

В математике kernel до промежуточных BF16 округлений:

\[
p_t=\sigma(T_t),\quad a_t=e^{ADT_t},\quad d_t=DT_{write,t},
\]
\[
S_t=a_tS_{t-1}+\underbrace{a_td_t(1-p_t)}_{\beta_t}F_{t-1}
                  +\underbrace{d_tp_t}_{\gamma_t}F_t.
\]

Здесь `F` — сумма rank-wise outer products повернутого K и `MIMO_V * V`,
с прежними biases; `S[-1]=F[-1]=0`. Чтение `Trap[t+1]` в kernel не означает
зависимости выхода t от будущего: shifted contribution входит только в более
поздние выходы. Дополнительно сдвигать gap на одну позицию нельзя.

## B. Существующие временные пути

`ThreeTimes('dual')` сохраняется целиком: отдельная обучаемая функция decay,
общая обучаемая функция scan для write и phase; fixed TRAIN reference
`R0=838393 ms`. `ADT=a*(DT*decay)`, `DT_write=DT*scan`,
`DT_phase is DT_write`. Обе функции, их параметры, optimizer и normalization
остаются прежними. Trap не заменяет DT и не управляет коэффициентом переноса
всего накопленного состояния напрямую.

## C. Изолированная поправка Trap

\[
q(g)=g/(g+R_0),\quad 0\le\alpha\le1,\quad
T'_t=T_t+\alpha q(g_t),\quad p'_t=\sigma(T'_t).
\]

На первой позиции и padding `q=0`. Используется прежний `history_gaps`,
только соседние timestamps внутри history. Настоящий zero-gap остаётся valid
event, его поправка равна нулю. Target timestamp не используется.

Один fp32 scalar `gap_trap.alpha` общий для обоих heads и слоёв, init=0.
`fixed_replay`: 715020 параметров; `gap_trap`: 715021. Создание нулевого scalar
не расходует RNG. При init добавка точно нулевая, common state не меняется.
Отношение вычисляется в float64, добавка — fp32; после сложения Trap возвращается
в прежний BF16 dtype до прежнего padding/kernel. Kernel, DT и rotary не меняются.
Маленькая поправка может округлиться в ноль в BF16.

При фиксированных content logit, ADT и DT:

\[
\partial_g p'=p'(1-p')\alpha R_0/(g+R_0)^2\ge0.
\]

Поправка увеличивает gamma и уменьшает beta; отношение new/old умножается
на `exp(alpha*q)`, не более e до BF16 округления. Это односторонний сдвиг к текущему входу.
Нулевой gap нейтрален; никакого обязательного old-предпочтения при коротком gap,
глобальной монотонности предсказаний или устойчивой семантики heads не заявляется.

Для exact zero-init и ненулевого начального градиента используется constrained
scalar с `clamp(0,1)` и проекцией после каждого Adam step. Проекция меняет только
alpha, состояния Adam сохраняются. PyTorch 2.9.1 проводит через clamp производную
1 на границе; это проверяется. Common optimizer settings не меняются;
для нового scalar это projected Adam. Если шаг направлен ниже нуля, alpha
законно остаётся нулевой: это допустимый результат гипотезы, не technical FAIL.

## Сопоставление с Mag-Mamba

Источник: [Mag-Mamba, arXiv:2603.00053v1, §4.3–4.4, Eq.15–21](https://arxiv.org/html/2603.00053v1).
В статье `delta=softplus(W_delta log(1+gap)+b_delta)`, `a=exp(A*delta)`,
`lambda=sigmoid(W_lambda*u)`, `beta=(1-lambda)*delta*a`,
`gamma=lambda*delta`; delta также входит во вращение. В §4.3 log-gap
входит в x, поэтому и lambda может зависеть от gap через u. География и
магнитная составляющая — отдельные ветки, здесь их нет.

Наша поправка сохраняет ADT/DT/phase при фиксированном остальном входе и меняет
только взаимодополняющие доли инъекции. Замена delta в формулах Mag-Mamba
в общем случае этого не воспроизводит: меняются также decay и rotation.
Конкретная ограниченная рациональная добавка с общей alpha не совпадает с
опубликованной параметризацией. Однако `T+alpha*q` можно представить линейным
gate над расширенными признаками `[u,q]`: сама идея gap-conditioned mixing
не новая. Это контролируемая абляция, не заявление нового семейства моделей.
Математическая эквивалентность полного предлагаемого вмешательства Mag-Mamba
не установлена; оснований для STOP по этому критерию не найдено.

## Допуск и бюджет

CPU: identity, monotonicity, masks/zero/extreme gaps, finite и ненулевой
alpha-gradient, roundtrip, projection/boundaries, counts, common state/RNG/config.
Targeted GPU: две parity-проверки относительно прежней модели (выход, loss,
input/common gradients, common Adam step); один набор modulation checks;
одна causality-проверка с alpha>0 (prefix/timestamps, suffix interventions,
cross-user isolation, gradients, finite, hook lifecycle).
Старые 45/2342 и 9/228 наследуются по SHA и lineage, без повторного GPU запуска;
численная policy не меняется. В частности, малый допустимый causality residual
не переименовывается в exact zero.

Далее synthetic smoke 2048×50, 3 шага на вариант; 2 scientific fits в свежих
процессах, одинаковые common params/RNG/first batch/precision/data.
Принудительный выход alpha из нуля проверяется отдельным синтетическим loss;
движение alpha на произвольном CE batch не является обязательным.
Диагностика фиксирована до fit: gaps/R0 = 0,.01,.1,.25,.5,1,2,4,10,100;
alpha, shift, odds ratio и sigmoid доли при content logits −2,0,2.

Один job: A100×1, CPU4, mem0, 6h, no-requeue, максимум 2 fits, TEST=0.
Порядок: provenance → targeted gate → smoke → fixed_replay → gap_trap → summary.
Технический сбой останавливает pipeline; низкая метрика не меняет бюджет.
Confirmation и layer-specific mechanisms в этот запуск не входят.
