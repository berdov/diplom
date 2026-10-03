# Временные функции отдельно по слоям

Один парный пилот на seed 2026: `shared_layers` и `layer_specific`.
Меняется только совместное использование параметров temporal calibrators между
двумя слоями Mamba-3. Контроль — MIMO dual fixed-reference, R0 = 838393 мс,
rank4/chunk8, 2 temporal heads, history50/padded56. TEST = 0.

Контроль сохраняет исходный constructor, state_dict и encode path. Treatment
оставляет `times` для layer0 и добавляет `layer1_times = deepcopy(times)` без
random draws. Два набора равны при старте, storage независим. В каждом наборе
есть decay и scan; write/phase используют один и тот же scan tensor. Оба слоя
получают один physical gap, вычисленный только по timestamps истории.
Параметров 715020 / 715152. Reference, формула calibrator, ядра и оптимизатор
не меняются. Никаких Gap-Trap, learned R, triple или других новых факторов.

До научных запусков: CPU проверки и targeted GPU gate из 6 случаев, включая
паритет control с исторической реализацией, исходный и нетривиальный tied-state
паритет, сумму temporal gradients, routing/interventions, masks/zero-gap,
roundtrip и causality на границах 8/9. Допуски из прежней политики не меняются.
Затем по 3 synthetic Adam steps для обоих вариантов при batch2048/history50.
Старый kernel gate 45/2342 наследуется по точным SHA.

Один job, A100 x1, rocky/proj_1833/type_e, CPU4, mem0, 6 часов, no-requeue.
Порядок: gate → smoke → fresh shared → exact historical replay → fresh specific.
Максимум 2 scientific fits; без retry, continuation или автоматического confirmation.
При техническом сбое сохраняются partial evidence и traceback, результат не
подменяется историческим числом. Метрика не является основанием менять план.

Primary — VALID NDCG@10, HR вторична. First27 только при полном окне 0–26.
Исторический контроль должен совпасть по всем метрикам, loss, выбранной эпохе,
checkpoint SHA, initial state, RNG и первому фактическому batch. Timing/memory
исключены из exact replay. Actual epochs не интерпретируются как speed benchmark.

Диагностика на заранее заданной сетке g/R0 из study_plan: decay/scan, оба heads,
оба layers; mean/max abs log-ratio и L2/relative L2 соответствующих параметров.
Знаменатель relative L2 — norm(layer0), null при нуле. Near bounds: <.51 и >1.99.
Оценка calibrators не запускает dataset forward. Сеточные доли не равны долям
событий. Расхождение функций не доказывает short-term/long-term роль слоёв.

Публикация после gate/smoke PASS, 2/2 полных fits, replay PASS и terminal audit.
Отрицательный результат публикуется так же, как положительный. Положительный
пилот позволяет только рекомендовать новые seeds отдельным решением. Статья,
Overleaf и пункт 5 остаются вне этой работы.
