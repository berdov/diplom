# Проверка представителей MOO: замороженный дизайн

Этот этап содержит **адаптации методов**, а не точные воспроизведения опубликованных
экспериментов. Исторический Stage 1 остаётся screening; его код, конфигурация и
метрики защищены `historical_inputs.yaml`. Общие dataset/loss/evaluation/checkpoint
helpers импортируются без изменения старого trainer. Новые циклы находятся здесь.

## Протокол и бюджет

KuaiRand-Pure, Protocol B: 23 951 пользователей, 7 111 items, 1 134 420 взаимодействий;
TRAIN 1 086 518, validation 23 951. Identity hash:
`954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.
Счётчик размера TEST берётся только из метаданных протокола. TEST-файлы не открываются.
У последовательного loader 1 062 567 обучающих примеров после исключения первого
взаимодействия каждого пользователя — это прежнее поведение RecBole.

Seed 2026; для независимых моделей seed=2026+i, как у EPO/GradHV. Batch 2048,
тот же MultitaskTiM4Rec и пять objectives в порядке
`rank, is_click, long_view, is_like, is_profile_enter`.
Adam, learning rate, weight decay, dropout, learning rate голов и pos_weights
наследуются из locked tuned MTL. `lambda_aux` и fixed task weights сохраняются в
метаданных, но дополнительная weighted-sum смесь НЕ добавляется поверх MOO:
методы комбинируют пять нормализованных losses. Это соответствует прежнему MOO.
Loss scales — средние восьми TRAIN diagnostic batches, без validation.

Smoke: 3 обучающих batch после TRAIN calibration; ranking evaluation отсутствует.
Общий data helper создаёт train/valid объекты и проверяет valid loader; TEST loader
не создаётся. Sanity: 5 полных epochs с validation после каждой. Convergence:
максимум 100 epochs, validation каждые 5, остановка не раньше epoch 20 после 3
проверок без строгого улучшения, min_delta=0. Основной checkpoint выбирается по
NDCG@10 заранее определённой рабочей точки. Никакого поиска гиперпараметров.

## Предпочтения и выбор решения

Используются прежние шесть preferences и continuous_eval_grid. Основная точка —
`rank_heavy=[0.6,0.1,0.1,0.1,0.1]`, где r обозначает важность objective.
FERERO обучает соответствующую модель; PHN-HVI оценивается при этом входе h(r).

MosT не получает r в optimizer. На каждой validation из трёх решений выбирается
минимум `sum_j r_j z_ij`, с tie-break по номеру модели, где
`z=(1-NDCG@10, click_BCE, long_BCE, like_BCE, profile_BCE)/(1,2,2,2,2)`.
Знаменатели — зафиксированный reference прежнего evaluation space; TRAIN loss
normalization остаётся отдельной операцией. Никакой нормализации по наблюдаемому
лучшему/худшему challenger или выбора max-NDCG между solutions нет.
Число 2 — scale и HV reference, не clipping BCE; плохие значения сохраняются.

Старый GradHV использовал validation oracle по NDCG для выбора solution. Поэтому
сравнение MosT с исторической цифрой содержит различие selection rule, которое
summary показывает явно. Stage 2 tuning приводится только как дополнительный фон.

## FERERO-adapter

[Статья, NeurIPS 2024](https://arxiv.org/abs/2412.01773),
[официальный код](https://github.com/lisha-chen/FERERO/tree/e350248c4e558173b740bc2ca1e02feb5e448253).
Используется projected dual PGD с simplex для множителей objectives и свободными
множителями равенств. Сохраняются 200 iterations, step 0.1, stopping tolerance 1e-3
из toy solver. Для больших Gram добавлен заранее заданный spectral cap `0.9/L`;
это стабилизация решения того же подзадачного objective, а не fallback scalarizer.

Mapping не вводит performance threshold: `v=(1/r)/||1/r||`, `A=I`,
`B_h[i]=v[i+1]e_0-v[0]e_(i+1)`, `H=B_h F`. Поэтому `H=0` означает только
`r_0 F_0 = r_i F_i`. Это многомерное продолжение loss-ray равенства official toy.
Технически статья относит такие однородные равенства к своей absolute-preference
формализации; здесь нет придуманного ограничения вида BCE≤c. Мы не выдаём их за
изменение cone-only относительного порядка. Gram считается по shared backbone,
как в старом EPO; итоговые коэффициенты действуют и на task heads. Внешний шаг Adam.
Из-за этих адаптаций результат называется FERERO-adapter, exact=false.

Parity сравнивает official solver и наше ядро на одинаковых gradients/losses/B_h
в 2 и 5 измерениях, включая coefficients, direction, scalar и residual. Для
режима активного spectral cap отдельно проверяется устойчивость; совпадение с
нестабильным фиксированным шагом upstream не заявляется.

## MosT-style adaptation

[Статья](https://arxiv.org/abs/2403.04099),
[официальный код](https://github.com/tianyi-lab/MosT/tree/726f20179a9cefd3a0f0d82e7b2bf915d50914a1).
Три независимые модели, пять реальных задач. IPOT и conditional MGDA сохраняют
официальные marginals, маскирование, EMA и обновление transport cost. Клиентские
parameter deltas заменяются градиентами minibatch; outer update — locked Adam.
Gram включает все trainable параметры, в том числе головы. Дополнительного
warmup и curriculum marginals нет. Параметры указаны целиком в config.

В upstream Frank-Wolfe `return` находится внутри первой итерации. Наше ядро
сохраняет этот эффективный один шаг, а также сочетание raw gradients через alpha
после решения на transport-weighted gradients. Проверки выполняют оригинальный
GlobalUpdate через неизменённые AST class nodes, исключая application imports.
Обе ветви (OT и conditional MGDA), направления, coefficients и set objective
сравниваются с официальным кодом. Это проверка ядра; результаты федеративной
статьи и её theoretical guarantees для Adam minibatches не заявляются.

## PHN-HVI-adapter и стоимость exact PHN

[PHN-HVI, AAAI 2023](https://arxiv.org/abs/2212.01130),
[официальный код](https://github.com/longhp1618/MultiSample-Hypernetworks/tree/63e818a8bc780ceb5deeae28d56360d884b8c923),
[исходный PHN](https://github.com/AvivNavon/pareto-hypernetworks/tree/6355fd28a05560806ba88ec898db4e6b46ba5ed9).

Исторический TiM4Rec содержит 593 498 параметров, MTL — 593 758. Генератор
`5→64→593758` содержал бы **38 594 654** параметра; FP32 weights+gradients+два Adam
moments занимают около **589 MiB**, без generated weights и activations. Восемь
сгенерированных наборов добавляют около 18 MiB только weights. Exact PHN технически
возможен на A100 80 GB; утверждения о невозможности по памяти здесь нет.
Однако генератор увеличивает параметрический бюджет примерно в 65 раз и требует
отдельной проверки functional forward через пользовательские CUDA kernels.
В этом screening выбран тот же параметрически экономный adapter, что у старого
PHN, чтобы исследовать эффект multi-sample HVI при общей архитектуре. Это осознанный
family-level контроль, не доказательство непригодности full PHN. Exact sanity
не запускается как необязательный четвёртый эксперимент.

FiLM generator `5→64→128`: **8704** параметра, 128 generated values, total 602 462;
та же нулевая инициализация последнего слоя и scale=0.1. Joint HVI учитывает восемь
Dirichlet(.2) inputs на batch, cosine coefficient=.001: defaults Jura. HVI действует
на вектор batch-mean objectives, как в Jura. Multi_MNIST имеет также per-example
вариант; его агрегация здесь не используется. Нет дополнительного 30-epoch cosine
warmup Jura и decay внешнего optimizer. Это явно отмеченные адаптации.

В official Jura sampled vector обозначает направление loss ray; в нашем benchmark
r обозначает важность задачи. Для сохранения смысла rank_heavy hypernetwork
получает прежний r, а cosine использует нормированный `1/r`. Например, большой
вес ranking требует меньшего ranking loss на луче. Это явная перепараметризация
с иным распределением loss rays, а не exact reproduction. HVI geometry не меняется.

Сохраняются multifront normalized HV gradients, duplicate handling и правило
`reference=max([1.5]*5,1.1*max_current_batch_loss)`; правило заморожено до данных.
Знак cosine соответствует official code. Наш геометрический solver использует
inclusion-exclusion для малого множества из восьми точек в пяти измерениях.
Parity включает dominated/duplicate/частично совпадающие точки, HV, surrogate,
backward и взаимодействие нескольких samples.

Два forward прохода с восстановлением CPU/CUDA RNG дают тот же joint gradient при
одном живом activation graph. Synthetic dropout test проверяет gradients и конечное
состояние RNG. Preference buffer копируется перед hypernetwork для корректного
autograd; старый adapter-файл не изменён. После smoke требуется ненулевая
чувствительность representation и ranking scores к preference.

## Provenance, лицензии и gates

SHA и SHA256 отдельных reference-файлов закреплены в `provenance.yaml`. У FERERO и
MosT на проверенных commits repository license не обнаружена; код не vendored.
Реализации математических ядер независимы. Parity загружает official files во
внешний cache и сверяет hash до исполнения. У PHN-HVI и исходного PHN — MIT.
Для старого HVI reference применяется только NumPy compatibility alias `np.bool`;
его математический код не изменяется. Недоступный reference — ошибка, не skipped test.

`verification.json` привязан к digest кода/config/dependency locks. Изменение
исходников закрывает gate. Каждый следующий stage требует completed JSON предыдущего
stage с passed checks и тем же digest. NaN/Inf, нулевой backbone gradient, отсутствие
обновления голов, одинаковые MosT models, постоянные FERERO coefficients или нулевая
PHN sensitivity закрывают gate. Отдельно проверяется TEST=0.

Runtime audit блокирует открытие TEST-файлов (включая symlink), запись в shared
processed/validation-only data и создание любого loader кроме train/valid.
Ограничение механизма: это Python audit hook и guarded factory, не OS sandbox для
произвольного нативного кода. Используемые loaders читают через Python API.

Slurm использует отдельный checkout, опубликованный SHA, rocky/type_e/A100,
4 CPU и 24h. Requeue отключён: неполный run сохраняется и не перезаписывается.
Автоматического resume нет; после preemption потребуется отдельное решение.
Submit имеет atomic lock, не создаёт дубликаты при неопределённом ответе scheduler.
Checkpoint binaries и runtime cache исключены из Git. Исторические jobs не меняются.
