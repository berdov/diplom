# Контракт attempt003 до GPU

## SISO: точная область изменения

Pinned `compute_dqkv` и локальный `stable_adt` проверяются AST-сравнением:
kernel, decorators/autotune и wrapper идентичны после единственной замены:

```python
# Было: r=dM_rev_vector, v=dM_vector, c=dM_scalar.
r += (sum(r) + c) + cumsum(v - r) - v
# Стало (точное равенство над вещественными числами):
r = reverse_cumsum(r) + exclusive_cumsum(v) + c
```

Exclusive prefix реализован gather предыдущего индекса inclusive prefix,
с нулём на индексе 0. Casts, masks, tensor strides, return order,
межчанковый carry, forward и остальные reductions не меняются.
Imports локальной копии содержат нужные upstream helpers; тела обеих
функций и decorators проверяются `scan_validation.validate_installed`.
Список autotune configs прежний: stages 1/2/3, warps 2/4/8,
maxnreg None/128/256. Но autotuner и скомпилированное ядро отдельные;
выбранные configs и посторонние derivatives могут различаться.
Именно это ещё нужно измерить, а не объявлять причиной заранее.

Основной кандидат: **stable_adt + официальный angle_dt_bwd**.
Default остаётся upstream. Combined stable_scan только advisory diagnostic.
Фиксированный matched-launch: warps=4, stages=2, maxnreg=128, один элемент
исходного supported списка, выбран до новых измерений. Приватная копия
globals wrapper направляет вызов в исходный JIT или candidate JIT;
upstream globals и autotuner caches не меняются. Launch configs, flags,
target, JIT source hash и доступные compiled metadata записываются реально.

Если replay на одних входах выявит изменение не-ADT outputs, разрешён один
diagnostic_hybrid: official derivatives плюс candidate dADT. Цена: два
полных dqkv на backward layer. Это не выбранный production backend.
Регистрируется повторный original, затем промежуточные dqkv/rotary/DT,
in_proj, residual dropout, norms и positional embedding gradients.
Ненулевые ошибки сохраняются. Float64 алгебра не является GPU evidence.

## MIMO: доказательство допустимости tail

Только rank4/chunk8, native gate L16; длина kernel = ceil(L/8)*8.
Сначала дополняются готовые kernel tensors, затем output обрезается до L.
Q/K: axis1 [B,L,R,G,N]; V/Z axis1 [B,L,H,P]; Angles axis1 [B,L,H,A];
ADT/write/phase/Trap axis2 [B,H,L]. Все tail values равны 0.
Biases, D, rank projections неизменны. Общие dw/dp сохраняют alias.

Pinned `mamba3_mimo_fwd.py` использует
gamma[t]=DT[t]*sigmoid(Trap[t]) и shifted beta[t+1]=DT[t+1]*sigmoid(-Trap[t+1]).
При нулевом tail обе величины равны 0, включая shifted term последнего
реального события. ADT=0 даёт exp(ADT)=1; phase DT=0 не меняет фазу.
Recurrence h[t]=exp(ADT[t])*(h[t-1]+beta[t]*KV[t-1])+gamma[t]*KV[t].
Tail не меняет state, а последний valid output не зависит от будущего tail.
Ненулевые Q/K biases не нарушают это: K_tail*V_tail=0 и gamma=beta=0;
Q_tail может читать прошлый state, но этот output отрезан; Z_tail=0 также
обнуляет gate. Никакого loss/final-state supervision на tail нет.

В pinned forward/backward нет деления или логарифма от DT: write/trap
производные в `bwd_dtrap_ddt_kernel` используют sigmoid и умножение,
phase backward использует tanh/sech и DT. Деления на размер head/RMS eps
не относятся к DT; outproj RMS отключён. Upstream alignment assertion и
nchunks=S//8 остаются на месте. Нулевой tail не заменяется epsilon.

Это алгебраическое обоснование, **не подтверждение GPU backward**.
BF16 states/reductions при другой вычислительной длине проверяются отдельно:
не требуем битового совпадения между длинами, используем неизменённый
reference profile. На одинаковой padded форме действуют прежние structural
допуски. Native L16 и official+adapter L17/L50 являются разными gates.
В MIMO используется только официальный angle-backward.
Base model oracle вызывает неизменённое pinned module.forward через локальную
копию globals: подменяется только kernel boundary на тот же adapter, а не
projection/calibrator code. Upstream module globals не изменяются.

## Evidence и критерии

Registry содержит IDs suite/architecture/backend/length/prefix/multiplier.
Started case сохраняется до вычисления; окончательный row до assertion.
Все required nested checks влияют на итог. Missing/duplicate/INTERRUPTED
не дают PASS. Parent завершает RUNNING progress при падении subprocess.
При Slurm SIGKILL неатомарно незавершённая запись остаётся RUNNING, никогда
не трактуется как финальный PASS. Старые evidence001/002 не исправляются.

Три angle diagnostics разделены: arithmetic pair, фактический GPU reverse
scan с carry против FP64 суммы, ideal elementary-function comparison.
Последний явно advisory с видимыми несовпадениями, не scan correctness.
Original exact-zero публикуется отдельно, без выдачи tiny residual за future
data use. Paired original/candidate/reference используют одни квантованные
входы; tied official calibration выполняется раньше independent triple.
Если baseline calibration не проходит, triple не объявляется confirmed.

Counts frozen: SISO 610440/610572/610638; MIMO 714888/715020/715086.
Никаких scientific fits/TRAIN/VALID/TEST. Только synthetic fixtures.
Каждый architecture subprocess ограничен 40 минутами внутри 90-минутного
Slurm allocation: timeout сохраняется как INTERRUPTED, другая архитектура
получает отдельный subprocess. Повторных запусков нет.
