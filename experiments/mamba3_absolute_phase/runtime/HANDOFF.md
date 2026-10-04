# Абсолютная фаза: подготовка пилота

Phase: SUBMITTED_PENDING. **2026-10-04 10:51:49 MSK:** PENDING (Priority), node UNKNOWN, elapsed 00:00:00. Scientific fits started UNKNOWN, completed UNKNOWN/3. GPU gate UNKNOWN; smoke UNKNOWN. Next explicit poll no earlier than 2026-10-04 11:01:49 MSK (2026-10-04T08:01:49.007566+00:00). No background process.

Запрос: `/Users/berdov/.codex/attachments/867dda07-c9ee-42e7-a7ff-e83250722c3e/Вставленный текст.txt`.
Выполняется только новый phase этап; старые пять исследований не повторяются.
Baseline MIMOdualfixed, relative/absolute clocks, две периодические пары6h/24h,
единственная shared W[32,4] exactzero, correction tanh(Wphi) до native angle_dt.
Пилот3 fresh fitsseed2026 baseline→relative→absolute. Confirmation12fitsseeds2027–2030
разрешено без вопроса только после complete/paired/auditPASS и absolute>=обоихcomparators.
Сначала прочитать state, plan и reservations; не дублировать отправки.

Точный execution11913ff8fe99aa28f124d667e2ae3aad14465098 опубликован и развернут.
TRAINcoverage:10000windows,301109validoccurrences,291109active;9–21апреля2022UTC,
49.56шестичасовых/12.39суточныхperiods,все24binsoccupied,9.425%zero-gapactive.
ДополнительныхstageB/C/E/Fнет. Native targetedgate17cases/532leaves затемsmoke3×3,
после них3freshfits2026. Sourcefreezeзавершён, никакихправокscienceпослеsubmit.
Runtime/audit_saved.py готовит независимыйagent; этиинструментывнеманифеста.

Ограничения: всего3submit/15startedfits/18hrequestedGPU максимум. Pilot6h,
conditionalconfirmation8h, один доказанный pre-fitwrapperretry4h. После начатого
прерванногоfit — INCOMPLETE, без автоматического refit. Scheduler polling послеJobID
не чаще600s; очередь в сессии не дольше4h. НовыхTEST/modelreplayвнеплана нет.
Статья, Overleaf, draw.io неизменны, сообщения не отправляются. Имена в репо не писать.

Следующий шаг: наблюдать существующийjob4374917 черезstage-aware
`runtime/capture_status.py` не чаще600секунд. После terminal выполнить
`runtime/preserve_terminal.py`, независимый `runtime/audit_saved.py`,
сохранитьaudit+preservationcommit. Полныйpilotопубликовать независимоотзнака.
Еслиfrozenruleразрешает, выполнитьconditionalconfirmation, затемполныйаудит/публикацию.
Строго ниодного дополнительногоsubmitрадиочереди/метрики. Веткупушитьможно;
clustercheckoutоставлятьexecution11913ff доterminal. Итог11пунктовпозапросу.

## Проверка будущей confirmation-обвязки

Pilot CPU100+100 PASS. Read-only запуск stdlibtests с ABS_PHASE_STAGE=confirmation
выявил12 ошибочных testmethods: fixtures создаютseed2026, но оставляют SEEDS2027–2030.
Это проблема областификстур, не научногоrunner; productionstage/seedпередаютсяправильно.
Текущийjob/checkoutНЕменять. Еслиpilotразрешитconfirmation, послеterminalисправить
толькоtestfixtures, сохранитьстарыйexecution/source и явно доказатьнеизменность
научныхфайловприпривязкеновоготестовогоexecution. Не копироватьpilotPASSсновымstamp.
Еслиfrozenruleнепройдёт, confirmationне готовить/не запускать.
