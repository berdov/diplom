# Память представлений истории: ограниченный пилот

## Вопрос и выбранная конструкция

Пункт плана предлагает явное обращение к прошлым состояниям с учётом реального
времени. Конкретная реализация этого пилота — чтение причинных выходных
представлений encoder внутри уже переданного окна. Native SSM state S_t —
внутренняя матрица recurrence; здесь архивируется causal representation h_t
размера 64 после output_norm. Это разные объекты.

“This pilot retrieves causal encoder representations within the supplied 50-event
window. It does not provide access to events outside that window or maintain
persistent user memory.”

H=[h_0,...,h_(L-1)] [B,L,64] вычисляется один раз прежним
ThreeTimeMamba3Rec.encode_sequence. Новый encode_sequence добавляет readout residual;
прежний forward выбирает последнее valid представление, tied item scorer и CE
сохранены. В архив идут исходные h_j, не residual r_j. Каждый новый forward заново
вычисляет H текущими весами. Межбатчевого cache и отдельной recurrence нет.

## Три контроля

1. no_memory: точный текущий MIMO dual fixed-reference, shared functions,
   715020 параметров, encode_sequence делегирует историческому пути.
2. index_memory: тот же backbone, новый selector использует t-j, 715021 параметр.
3. time_memory: тот же reader, selector использует elapsed time, 715021 параметр.

Index control остаётся time-aware через прежние temporal calibrators backbone:
исторические timestamps там не заменяются. Меняются только часы нового selector.
Seed2026, fresh TRAIN→VALID в этом порядке. Primary — time−index; secondary —
time−no_memory и index−no_memory. Выигрыш только у плохого index control при
проигрыше no_memory не основание заменять backbone.

## Selector

g=history_gaps(timestamps,valid), g_0=0, отрицательные соседние gaps clamp0.
u=cumsum(g), float64. Возраст time=(u_t-u_j)/838393; index=t-j. Общий сдвиг
корректно представимых timestamps не меняет addressing. Target timestamp и
время следующего запроса недоступны. Для каждого собственного query t разрешены
только valid j<t; настоящее zero-age разрешено, current/future запрещены.

K=4, anchors=[1,4,16,32], frozen до coverage. Последовательно по anchors
минимизируется abs(log1p(age)-log1p(anchor)) среди оставшихся позиций. Точные ties
выбирают наиболее поздний j, без epsilon. Выбранные позиции уникальны, повтор
item ID разрешён. После выбора valid indices сортируются по позиции; slot mask
отделён от sentinel -1, gather предварительно переводит sentinel в безопасный
индекс, invalid contribution строго0. K_t=min(4,valid past count) у обоих режимов.
Out-of-range регистрируется относительно доступных оставшихся кандидатов для
данного anchor. Выбор вне диапазона не означает попадание в заданный возраст.

Selector дискретный. Градиенты reader проверяются с fixed indices; ненулевые
вмешательства в прошлое и suffix проверяют addressing непосредственно.
Один future timestamp gradient не является доказательством причинности.

## Reader и инициализация

M_t содержит выбранные h_j. Scores=dot(h_t,h_j)/sqrt(64), masked softmax,
m_t=sum(w*h_j), r_t=h_t+tanh(beta)*m_t. Scores/readout float32;
backbone precision/BF16 kernels прежние. Один fp32 scalar beta, init EXACT0
без RNG draws; Adam0.001, прежние betas/eps и общая группа параметров.
Нет Q/K/V projections, temperature tuning, time bias, новой LayerNorm.
При empty memory readout/weights равны0, softmax полностью -inf строки не считается.
Padding residual нейтрален. Query/values в scientific graph не detach.
Отрицательная lambda=tanh(beta) вычитает readout; она разрешена заранее.

При beta0 выходы и common gradients должны совпасть с no_memory; первый common
Adam update также. После update beta может стать ненулевой, последующие outputs
не обязаны совпадать. Общие initial weights, RNG, first consumed train batch,
данные, Adam и precision совпадают в тройке. Exact no_memory replay ожидается
по всей scientific history и weight checkpoint SHA; mode metadata/output paths
исключены. Расхождение требует расследования, не подгонки или повторного fit.

## Стоимость и coverage

Dense selector: O(B*K*L^2) operations, O(B*L^2) рабочие metadata tensors.
Reader: O(B*L*K*D), gather только [B,L,K,D], не [B,L,L,D]. Вся augmented model
не объявляется линейной по L. Проверяется только L≤50, online lifetime serving нет.

До scientific submit выполняется TRAIN-only аудит без model forward/loaders:
n=min(N,10000), indices=floor(i*(N-1)/(n-1)); только input histories.
Сохраняются hashes, lengths, spans, anchor coverage, ages/lags, out-of-range,
совпадение selected sets. Полное совпадение =>
UNINFORMATIVE_ON_AUDITED_TRAIN_SAMPLE, научный submit блокируется. При частичном
совпадении сохраняется доля без нового post-hoc порога. Anchors не подбираются.

VALID diagnostics собираются в существующих forwards, только last valid query:
beta/lambda, slots, ages/lags, weights, norm ratio, empty/out-of-range, overlap;
streaming counters и ограниченная детерминированная SHA-reservoir без training RNG.
Полные tensors не сохраняются. Веса reader не трактуются как causal importance.

## Предыдущие работы и границы утверждений

[HPMN — Ren et al., SIGIR2019](https://arxiv.org/abs/1905.00758) использует
иерархическую память с периодическими обновлениями для длительных пользовательских
последовательностей. Пересечение — явная память прошлой информации; текущий пилот
не реализует lifelong memory или их иерархическое обновление.

[TiSASRec — Li et al., WSDM2020](https://jiachengli1995.github.io/files/wsdm20.pdf)
учитывает позиции и временные интервалы в sequential self-attention. Пересечение —
использование elapsed time при чтении истории. Здесь фиксированный дискретный
selector четырёх состояний поверх прежнего Mamba encoder, без обучаемых Q/K/V.
Существование этих подходов не мешает контролируемой абляции, но исключает
объявление памяти или временного чтения новой идеей само по себе. Первенство,
доказанные циклы интересов и полная внутренняя память Mamba не заявляются.

## Протокол и остановка

Прежние KuaiRand TRAIN/VALID, full catalog, history50; user-wise chronological
leave-one-out не объявляется global point-in-time replay. TEST evaluations0,
TEST loader не создаётся; технически возвращаемое описание split сразу удаляется.
MIMO rank4/chunk8,2layers/2heads,dual,fixed R0,shared calibrators остаются прежними.
Никаких head_tau, Gap-Trap, triple, layer-specific maps, иных memory variants.

Одна allocation до6часов, максимум3 started scientific fits. Единственный retry
до4часов разрешён только для доказанного infrastructure/wrapper сбоя до любого fit,
после сохранения failure и regression, с прежним scientific design/tolerances.
Numerical gate failure, OOM smoke, quality loss, прерванный fit не разрешают tuning
или refit. Durable reservation до sbatch, неоднозначный submit не повторяется.

После полного audit публикуется любой знак результата, ровно3 registry rows,
исторические строки byte-identical. Confirmation не запускается автоматически;
положительный пилот только мотивирует отдельное multi-seed задание. Ограниченный
отрицательный пилот не опровергает другие способы explicit memory. Статья и
Overleaf вне задания.
