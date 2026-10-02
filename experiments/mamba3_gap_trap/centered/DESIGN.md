# Centered Gap-Trap: последний pilot пункта 3

Два свежих fit seed2026: fixed_replay, затем centered_gap_trap. KuaiRand,
TRAIN from scratch, выбор по VALID, TEST=0. Backbone: fixed-reference MIMO
 dual rank4/chunk8, 2 heads, 2 layers, history50. Параметры: 715020 / 715021.

Для активных соседних событий history:

    q_c(g) = (g-R0)/(g+R0), R0=838393 ms
    T' = T + alpha*q_c(g), p'=sigmoid(T'), 0 <= alpha <= 1.

q_c(0)=-1, q_c(R0)=0, предел при бесконечном gap равен+1;
производная 2R0/(g+R0)^2 положительна. Для g>0 это tanh(.5*log(g/R0));
q_c(R0/r)=-q_c(R0*r). Вычисляется рациональная форма, без log(0).
Первое событие и padding имеют q=0. Настоящий активный zero-gap имеет q=-1.
Target timestamp не используется. Один scalar alpha fp32, общий для heads/layers,
init exact0 без расхода RNG. Прежний Adam, прежний LR; после step alpha
проецируется в [0,1], моменты Adam сохраняются. Законное застревание у0
не является технической ошибкой и не разрешает смену гипотезы.

Ratio вычисляется в fp64. Для конечных огромных g результат может округлиться
до+1: только этот верхний endpoint ограничивается nextafter(1,0) в fp64 и
повторно после fp32 cast. Это защита представления открытого верхнего предела,
не новый гиперпараметр. На конечной сетке floating-point монотонность нестрогая.
После сложения fp32 shift с raw Trap сохраняется прежний BF16 kernel dtype.

При фиксированных ADT=a_log, DT_write=d и content logit:

    a=exp(ADT), beta=a*d*(1-p'), gamma=d*p'
    S_t=a*S_(t-1)+beta*F_(t-1)+gamma*F_t.

Short gap<R0 уменьшает p' относительно исходного p и увеличивает предыдущую
инъекцию; long gap>R0 даёт обратный сдвиг. Gate odds умножаются на exp(alpha*q),
крайние множители exp(-alpha),1,exp(alpha), long/short=exp(2alpha).
Это условная интерпретация gate, не монотонность рекомендаций и не полный
коэффициент gamma/beta, в котором также присутствует decay a.

Связь со старой функцией: q_c=2q_plus-1, поэтому alpha*q_c=2alpha*q_plus-alpha.
Это affine recentering: масштаб плюс глобальный Trap-logit bias. Обучаемые
content logits могут поглотить bias; новое множество представимых функций
не заявляется. Проверяется short-old / long-new inductive bias относительно R0,
а не новый класс Mamba или новый gate.

## Upstream и reuse

Pin: e9594ce1c732d97440f0332fdc43170a2294dbfa. Сохраняется
[аудит Trap](../DESIGN.md#a-штатный-trap): modules/mamba3.py:176–191,
ops/tilelang/mamba3/mamba3_mimo.py:279; mamba3_mimo_fwd.py:196–218,
303–321,340–346,436–457; triton/mamba3/mamba3_mimo_utils.py:303–336.
Raw Trap pre-sigmoid, gamma=d*sigmoid(T), shifted contribution использует
sigmoid(-T); переход chunk не меняется. Docstring utils:252 содержит старую
ошибку знака, исполняемый код корректен.

Наследуется encode_sequence и неизменный ../mixer.py, модифицирующий ровно
raw Trap tensor. ADT, write/phase, calibrators, angles и kernels не изменяются.
Изоляция проверяется при одинаковом входе одного mixer: после изменённого
первого слоя вход второго слоя закономерно может измениться.

[Карта reuse](reuse_bindings.json) фиксирует SHA родительских модулей.
FunctionType использует тот же code object и частный globals dict; в нём явно
связываются centered config/callbacks и relative imports. Исходные globals,
sys.modules и parent files не меняются. Это сохраняет frozen lineage без
копии полного runner/pipeline/trainer. Новые части: feature, диагностика,
контроль replay, targeted fixtures и узкая маршрутизация процессов.

## Сопоставление с Mag-Mamba

Повторно проверены [Mag-Mamba v1, §4.3–4.4, Eq.15–21 и B.1](https://arxiv.org/html/2603.00053v1).
Там delta=softplus(W_delta*log(1+g)+b_delta), a=exp(A*delta),
lambda=sigmoid(W_lambda*u), beta=(1-lambda)*delta*a, gamma=lambda*delta;
delta участвует также во вращении. Log-gap входит в x и через u может влиять
на lambda: это не только изменение DT. Буквальной добавки
alpha*(g-R0)/(g+R0) к raw Trap с общей ограниченной zero-init alpha в этих
уравнениях не найдено; критерий STOP по точному совпадению не выполнен.
Gap в Mamba3 и gap-conditioned mixing не объявляются новыми. Это controlled
mechanistic ablation для general sequential recommendation.

## Проверки, сравнение и предел работы

Старые 45/2342, 9/228 и one-sided 4/133 сохраняются через source/evidence SHA;
новые проверки не переименовывают допустимые малые residual в exact zero.
CPU: feature/masks/zero/extremes, gradients, projection, identity, odds,
roundtrip, counts/RNG/config, history-only API, синтетический первый batch,
изолированный reuse/paths и отрицательные fixtures replay contract.
GPU: fixed/centered alpha0 parity output/loss/common gradients/Adam;
centered feature, реальный native forward/backward с raw Trap isolation,
граница chunk на позициях7/8/9 и prefix causality8/9, cross-user и hook lifecycle.
После gate — 2x3 synthetic steps batch2048/history50, без VALID/TEST.

Scientific fits запускаются раздельными свежими процессами. Сохраняются common
weights/calibrators/RNG/DataLoader generator, первый фактически потреблённый
batch, data SHA, Adam/LR/precision/early stopping/max epochs/selection.
До centered fit свежий fixed обязан точно воспроизвести historical checkpoint
SHA, все scientific epoch metrics/train loss/common diagnostics и начальные
pairing fields; timing/memory исключены. Иначе STOP без интерпретации delta.
Прежнее правило: последний tied максимум округлённого VALID NDCG@10,
patience10 (stop после11 ухудшений), max300. First27 — только эпохи0–26.

Диагностика до запуска: g/R0=[0,.01,.1,.25,.5,1,2,4,10,100], T=[-2,0,2].
Сохраняются alpha, q, shift, odds, sigmoid, BF16 shift cast отдельно и эффект
после сложения/cast Trap; доля ненулевых shifts, округлившихся в0, исключает
заведомо нейтральный q=0. Если alpha=0, знаменатель0 и доля null.
История alpha, best/max/final/zero-epochs извлекается аналитически без dataset forward.

Один job rocky/proj_1833/type_e/A100x1/CPU4/mem0/6h/no-requeue, максимум2fits.
Без retry, continuation, confirmation, tuning, TEST, кода/ветки/job пункта4/5
и изменений статьи. После этого pilot пункт3 закрывается в текущем exploratory
KuaiRand/VALID scope. При выигрыше только рекомендация отдельного решения о
paired seeds2027–2030; при отсутствии выигрыша третьего Trap-варианта не будет.
Следующий возможный пункт — layer-specific temporal functions, отдельно.
