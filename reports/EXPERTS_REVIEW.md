# Эксперты поверх временной Mamba3: обзор и границы новизны

Дата поиска: **17.09.2026**. Основа: `origin/main` на `bfac35ac1cff1aa80103df709f11ad5c69b61391`.

## Вывод

**Четыре MLP после итогового `h` являются стандартным MoE-контролем, а не установленным новым вкладом.** Уже существуют Mamba+MoE для рекомендаций, смеси временных SSM и маршрутизация параметров перед единственной рекуррентностью. Перенос на Mamba3, добавление gap в router и слова «временные эксперты» сами по себе новизну не устанавливают. Ближайшие пересечения: [HM2Rec](https://ojs.aaai.org/index.php/AAAI/article/view/38567), [Swimba](https://arxiv.org/abs/2603.06938), [STM3](https://arxiv.org/abs/2508.12247), [MoM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae2dd45fb3d7ce1dce45e5de4bdd9ba0-Abstract-Conference.html), [KVAE](https://papers.nips.cc/paper_files/paper/2017/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html).

Обоснован один **условный исследовательский вопрос**: полезна ли зависимость именно поправок physical-gap к decay/scan от доступного item-контекста, сверх существующей селективности Mamba и увеличения числа параметров? Ни необходимость экспертов, ни прирост, ни оригинальность такой реализации пока не доказаны. Ниже один baseline и один кандидат, не утверждённая архитектура.

## Наша постановка и предыдущие попытки

[Контракт входов](EVALUATION_SETUP.md): single-task next-item CE, full-ranking по 7111 items; история до 50 событий, обучаемые item embeddings `[B,L,64]`, точные исторические gaps. Это не 64 видеоатрибута. Нет rich user/video features, `t_score`, интервала до запроса или target timestamp во входе encoder. Итоговый `h:[B,64]`, tied scorer `h E_item^T`.

[Separate](../experiments/mamba3_time_mechanisms/time_mechanisms.py) вычисляет две функции `tau=log1p(gap/838393)`: `1→16→2`, SiLU, затем `exp(log(2)*tanh(raw))`. Выходы `[B,L,2]` относятся к **двум heads**, а не слоям; обе функции переиспользуются двумя layers. Первый event и padding дают scale=1; действительный нулевой gap активен. [Mixer](../experiments/mamba3_time_mechanisms/time_mamba3.py):

```text
ADT = A * (DT_base * s_decay)
DT  = DT_base * s_scan
```

`A<0`, `DT_base>0`; DT влияет на input weighting и rotary/phase. Базовые `A/DT_base` уже зависят от содержимого, хотя дополнительные calibrators видят только gap. Обычная Mamba selectivity не равна выбору независимых экспертов. [Подтверждающая серия](MAMBA3_TIME_MECHANISMS_RESULTS.md#confirmation) поддерживает сравнение shared/separate в своей постановке, но не доказывает потребность в MoE; новый constant-gap array здесь не проверялся.

История проверена по коду и сохранённым JSON, не по названиям launcher:

| Что найдено | Что действительно подтверждено |
|---|---|
| Behavior-MoE TiM4Rec, ref `7ab0165` | Четыре MLP после `h`, отдельные soft gates для пяти задач; smoke и 5-epoch sanity, VALID NDCG@10=.0562, TEST=0. Это MTL, не Mamba3. |
| Structured Behavior-MoE | Сохранён только smoke с масками допустимых task/expert связей; не завершённый ranking benchmark. |
| PLE/CGC-style на том же ref | Код one-level baseline есть; ожидаемых PLE run JSON в проверенных refs нет. Выполнение не подтверждено. |
| EPO-MoE | В исторической summary все m0/m2/m4/m8 `missing`; инфраструктура удалена коммитом `bb4fcea`. Завершённое обучение не подтверждено. |
| Proto-Mamba3 | [Завершённый run](../experiments/mamba3_prototypes/README.md): смесь постоянных обучаемых векторов, **не экспертов-функций**; не доказательство проверки MoE. |

В текущих 63 строках `results.csv` нет MoE/PLE/expert run IDs. Чистое рабочее дерево и доступные refs не содержали незавершённого Mamba3-experts расширения; это не утверждение об отсутствии файлов на кластере. Исторические refs, пути, статусы и ограничения перечислены в [evidence](evidence/experts_sources.json).

## Сравнение механизмов

`F` = Methods/формулы прочитаны в полном тексте указанной версии; не воспроизведение результатов. `P` = частичный доступ, `A` = аннотация. Полные авторы, версии, формулы, абляции, код и неизвестные поля: [источники](evidence/experts_sources.json), [BibTeX](evidence/experts_refs.bib). Отобраны 12 работ; версии одной работы не считаются отдельными публикациями.

| Работа / статус | Эксперт и место | Router / гранулярность | Физическое время и отличие от separate |
|---|---|---|---|
| [MMoE, KDD 2018](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) A | Общие expert submodels, отдельные task gates | По примеру и задаче; детали реализации не проверены | MTL; gap-управление SSM не установлено |
| [MoSE, KDD 2020](https://storage.googleapis.com/gweb-research2023-media/pubtools/5631.pdf) F | LSTM-bottom → LSTM-experts → task towers | Dense softmax **всей истории**, один gate на задачу/пример | Временные ряды активностей; не event-wise gap router |
| [PLE, RecSys 2020](https://doi.org/10.1145/3383313.3412236) P | Shared/task-specific extraction, много уровней | Полные формулы gate недоступны | Соседняя MTL-постановка, не доказанный temporal аналог |
| [HM2Rec, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38567) F | MLP-MoE вместо FFN в RPMamba; graph/VGAE | Token embedding; softmax, в настройках top-2 | RoPE позиции; physical gaps в уравнениях router не заданы |
| [FAME, WSDM 2025](https://arxiv.org/html/2411.01457v2) F | Expert query projections внутри attention heads SASRec | Dense смесь expert outputs, per-token/per-head | Item history; не временная дискретизация SSM |
| [MoE-Mamba, ME-FoMo 2024 workshop](https://arxiv.org/html/2401.04081v2) F | Mamba-1 чередуется с FFN-MoE; также inner-projection варианты | Token-wise Switch top-1 | Языковая модель; не physical-gap calibration |
| [MoM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae2dd45fb3d7ce1dce45e5de4bdd9ba0-Abstract-Conference.html) F | Несколько recurrent memories + shared memory | Token-wise top-k, обновляются выбранные states | Отдельные траектории, не смесь scale одной траектории |
| [STM3, KDD 2026](https://arxiv.org/html/2508.12247v3) F | Multiscale Mamba experts; bias дискретизации | Node embedding, top-1 + shared expert | Масштабы регулярных рядов/свёрток, не интервалы событий пользователя |
| [KVAE, NIPS 2017](https://papers.nips.cc/paper/2017/file/7b7a53e239400a13bd6be6c91c4f6c4e-Paper.pdf) F | Смесь матриц A/B/C внутри LGSSM | LSTM прошлых кодов, dense по шагам | Моделирование видео; переключение динамики давно известно |
| [Swimba, arXiv 2026](https://arxiv.org/abs/2603.06938) F | Mamba-2 expert in-projections перед одним SSM | Token/layer routing, эксперимент top-1 из четырёх | Очень близкий parameter-space MoE; не наш двухканальный gap calibrator |
| [TriSSR, Neurocomputing 2026](https://doi.org/10.1016/j.neucom.2026.133573) P | Bi-Mamba2 + frequency + Time Fourier experts | Адаптивное объединение; точный gate неизвестен | Реальные интервалы заявлены; полный Methods недоступен |
| [More Experts, Worse Dynamics, arXiv 2026](https://arxiv.org/html/2608.21840v1) F | Dense смесь spectral transition operators | Текущий input и magnitude state | Отрицательный synthetic результат; не теорема о неуспехе Mamba3 |

## Ближайшие совпадения и пробелы

**HM2Rec:** pp.15406–15409, Eq.16–17 и Table3 подтверждают MLP expert routing и сравнение с обычным FFN. Но формулы суммируют всех экспертов, настройки говорят top-2; порядок truncation/renormalization без кода неизвестен. Релиз Mamba не зафиксирован, приписывать Mamba3 нельзя. Авторский код не найден, что не доказывает его отсутствия.

**Swimba:** §3.2, Eq.7–9 смешивают injection/readout при общем A; §4 переносит expert projections в Mamba-2. Значит «эксперты до scan, одна recurrence» уже не новое отличие. Наша возможная граница уже: bounded physical-gap поправки отдельно к decay и scan/input weighting. Это проверяемая специализация известного подхода, не доказанная новая категория.

**STM3:** Eq.8–11 связывают multiscale experts с дискретизацией. Код содержит softplus и gated shared branch, которые нельзя восстановить буквально из сокращённых формул; лицензия репозитория не определена. Не переносим теоретические гарантии статьи на нашу параметризацию.

**MoM и KVAE:** первая маршрутизирует память, вторая интерполирует параметры динамики. Несколько независимых memories, mixture-of-outputs и mixture-of-parameters не эквивалентны. В частности, `exp(sum(pi*ADT_e))` обычно не равно `sum(pi*exp(ADT_e))`.

Открыты существенные пробелы: полный Methods TriSSR и [TMTRec](https://doi.org/10.1109/ECIS69634.2026.11604427) (метаданные IEEE подтверждены, текст недоступен), полный PLE/MMoE, точное соответствие HM2Rec коду. Свежий отрицательный GMS-препринт мотивирует контроли, но его synthetic setup и supervised routing warmup не переносятся на наш CE. До закрытия этих пробелов нельзя заявлять полную оригинальность.

## Стандартный baseline: выходной dense MoE

Проверяет, помогает ли условное преобразование уже сжатой истории. Не проверяет изменение внутренних DT/ADT и не восстанавливает гарантированно утраченную информацию.

```text
h: [B,64]; E_e(h)=Linear(64,64)→GELU→Linear(64,64), e=1..4
pi=softmax(W_g h+b_g): [B,4]
h'=h+sum_e pi_e E_e(h): [B,64]
scores=h' E_item^T: [B,7112]
```

Фиксированный residual factor=1, без нового loss/dropout; последние linear экспертов можно zero-init для identity. Все четыре эксперта вычисляются: добавлено **33540 total/active parameters**, всего с separate **644112**; примерно 33024 дополнительных MAC на историю, scorer неизменен. Это проектный расчёт, не измеренный runtime.

Контроли: separate; residual dense `64→260→64` с 33604 параметрами (+64 к MoE, +0.19% блока); те же эксперты с uniform weights; обучаемый input-independent gate. Dense-контроль приблизительно, не точно, равен по ёмкости. Сопоставить фактическое время. Никаких заранее названных «interest/consumption» экспертов или принудительного diversity. Если dense/uniform дают тот же эффект, вклад routing не подтверждён. Результат не следует называть воспроизведением HM2Rec: там token-level FFN и другие компоненты.

## Единственная гипотеза: контекстные функции исторического gap

**Проблема.** Сейчас дополнительная поправка для одинакового gap одинакова для всех items. Возможно, нужны разные поправки в разных контекстах. Но content-aware `DT_base/A` уже могут это компенсировать: поэтому это гипотеза об избыточности/полезности отдельной факторизации, не обнаруженный дефект separate.

**Подключение.** Перед двумя mixers, один раз по исходному item embedding `u:[B,L,64]` и `tau:[B,L,1]`; общие для layers веса. Router видит локальное содержимое текущего исторического item, не всю историю. Четыре пары независимых MLP `1→16→2`; каждая пара вычисляет log-scales decay/scan, не отдельную SSM. Один router выбирает пару функций:

```text
pi = softmax(Linear(65,4)(concat(u,tau)))          [B,L,4]
q_e,p = log(2)*tanh(MLP_e,p(tau))                [B,L,2]
ell_p = sum_e pi_e*q_e,p                         [B,L,2]
s_p = exp(ell_p), p in {decay,scan}              [B,L,2]
ADT = A*(DT_base*s_decay); DT = DT_base*s_scan
```

Здесь `p` означает путь, последняя размерность означает heads. Decay/scan функции разные, gate общий: это явно ограниченная смесь пар, а не два независимо маршрутизируемых банка. Items/gaps только из доступного префикса; нет target, user embedding, внешнего `t_score`, idle gap, новых признаков, MTL, prototypes или смены CE. Нельзя подавать финальный `h` назад для маршрутизации ранних событий: он содержит более поздний контекст.

**Чем отличается в рассмотренных работах.** От выходного baseline и HM2Rec: непосредственное изменение ADT/DT. От Swimba: ограниченные log-scale функции physical gap вместо полной expert in-projection. От STM3: per-event gap/content, а не node-wise выбор целой multiscale SSM. От KVAE: не полные матрицы перехода; от MoM: одна state trajectory. Эти различия не гарантируют новизну и не снимают вопроса о TriSSR/TMTRec.

**Совместимость.** Выпуклая смесь `q∈[-log2,log2]` оставляет `s∈[0.5,2]`; знак A и положительность DT сохраняются. `exp(ADT)` остаётся затухающим множителем, но это не доказательство устойчивости всего nonlinear backbone. Softmax/SiLU/tanh/exp дифференцируемы. First/padding принудительно дают 1. Zero-init последних linear даёт vanilla temporal identity; один эксперт или одинаковые пары восстанавливают семейство separate. Router получает нулевой gradient на самом первом шаге при всех `q=0`; это ожидаемо, а не доказательство collapse. Не следует копировать все expert hidden weights идентично.

Scales вычисляются до scan и имеют прежние формы, поэтому **интерфейс** существующего `mamba3_siso_combined` не требует нового kernel. Это проектная совместимость, не GPU evidence: позже понадобятся проверки output/gradients, masking и нейтрального режима. Router, зависящий от текущего recurrent state, сюда не входит: для него нельзя обещать прежний параллельный scan.

**Бюджет.** Восемь calibrators по 66 параметров + router `(65+1)*4=264`: 792 вместо 132 у separate; всего **611232 total/active**, прирост **660**. Все эксперты активны. Около 644 MAC на event до смешивания против 96 у separate, плюс нелинейности; storage expert outputs `[B,L,4,2,2]`. Число backbone scans прежнее; sparse speedup не обещается.

**Опровержимые контроли.** Separate; direct dense calibrator `[u,tau]→11→4` (774 параметра) и ширина 12 (844) по обе стороны бюджета 792; uniform/input-independent router; content-only router, где убран только `tau` из gate, но истинные gaps сохранены у экспертов; gap-only router. Bounds, маски, scorer и loss одинаковы. Dense-контроли выводят четыре bounded scales, это не точное parameter matching, разницы должны быть указаны.

Если direct dense не хуже, польза экспертной факторизации не показана. Если content-only не хуже, temporal routing не подтверждено. Если uniform не хуже, условный выбор не нужен. Диагностика usage/entropy и scale-кривых описывает результат, но не служит loss для навязывания специализации. Будущий протокол должен фиксировать бюджет, paired seeds и VALID selection до запусков; новых TEST для выбора архитектуры не нужно.

## Следующий шаг

**Согласовать с Дмитрием один контрольный эксперимент: separate + выходной MoE против сопоставимого dense residual и uniform mixture, без статуса нового метода.** Внутренний кандидат оставить условным до оценки необходимости routing и закрытия ближайших источников. Сейчас реализации, запусков и изменений научных артефактов нет; main не изменяется.

### Последующее решение, 18.09.2026

Рекомендация выше сохранена как история решения. В новом задании выбран [внутренний context-time pilot](../experiments/mamba3_context_time/README.md): separate replay, dense11/dense12 и uniform/routed calibrators, чтобы прямо проверить контекстную временную калибровку, а не преобразование final h. Это наша фиксированная exploratory постановка, не согласование формул Дмитрием. Новизна не доказана; full Methods TriSSR/TMTRec и соответствие HM2Rec авторскому коду остаются пробелами. На момент выбора results pending; выходной MoE в этом задании не реализуется.

Проверка завершена: [результаты внутренней калибровки на seed2026](MAMBA3_TIME_MECHANISMS_RESULTS.md#context-time-pilot). Routed не улучшил итоговый VALID NDCG@10 относительно separate replay и сравнялся с uniform с доступным округлением; экспертный вариант пока не включён в основную модель. Это вывод о проверенной постановке, не о бесполезности экспертных механизмов вообще.
