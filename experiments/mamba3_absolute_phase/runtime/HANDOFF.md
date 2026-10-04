# Абсолютная фаза: подготовка пилота

Phase: CPU_PREFLIGHT. GPU jobs ещё не отправлялись, scientific fits=0, TEST=0.
Canonical local `/Users/berdov/diplom`; cluster `/home/daryumin/iberdov/diplom`,
SSH `hse-karizma`. Branch `exp/mamba3-absolute-phase`, base main `de2cddf271bd4c5ccaa4693552f005799a171526`.

Запрос: `/Users/berdov/.codex/attachments/867dda07-c9ee-42e7-a7ff-e83250722c3e/Вставленный текст.txt`.
Выполняется только новый phase этап; старые пять исследований не повторяются.
Baseline MIMOdualfixed, relative/absolute clocks, две периодические пары6h/24h,
единственная shared W[32,4] exactzero, correction tanh(Wphi) до native angle_dt.
Пилот3 fresh fitsseed2026 baseline→relative→absolute. Confirmation12fitsseeds2027–2030
разрешено без вопроса только после complete/paired/auditPASS и absolute>=обоихcomparators.
Сначала прочитать state, plan и reservations; не дублировать отправки.

Candidate commit `33c13bb923f93852ed6e054ee449e1a7f98aa1a5` отправлен и развернут;
исходники пока до финального freeze/submit. CPU preflight этого кандидата выполняется
в существующем clusterenv с пустой CUDA_VISIBLE_DEVICES. LocalvenvTorchотсутствует;
новое окружение не создаётся. Дополнительные submissionregressions ещё добавляются.

Ограничения: всего3submit/15startedfits/18hrequestedGPU максимум. Pilot6h,
conditionalconfirmation8h, один доказанный pre-fitwrapperretry4h. После начатого
прерванногоfit — INCOMPLETE, без автоматического refit. Scheduler polling послеJobID
не чаще600s; очередь в сессии не дольше4h. НовыхTEST/modelreplayвнеплана нет.
Статья, Overleaf, draw.io неизменны, сообщения не отправляются. Имена в репо не писать.

Следующий шаг: закончить CPU/noGit, TRAINcoverage, проверитьfinalmanifest,
зарезервировать и отправить ровно один pilot job. Полные результаты независимо
проверить и опубликовать; если frozenruleразрешает, выполнить confirmation.
Финал11пунктов по запросу и короткий lowercase текст сообщения, не отправлять.
