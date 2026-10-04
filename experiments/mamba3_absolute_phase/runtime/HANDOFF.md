# Абсолютная фаза: подготовка пилота

Phase: AWAITING_TERMINAL_AUDIT. **2026-10-04 12:14:24 MSK:** COMPLETED, node cn-043, elapsed 01:17:36. Scientific fits started 3, completed 3/3. GPU gate PASS; smoke PASS. Terminal evidence awaits preservation and independent audit. No background process.

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
Независимый runtime/audit_saved.py готов: 12 stdlib regressions PASS.
Он проверяет saved bytes без Torch, model replay и загрузки весов; вне manifest.

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

Фактический confirmation CPU preflight на execution11913ff: 100 tests,
29 failing subcases и 2 errors в тех же 12 fixture methods. Исходный FAIL JSON
и stdout/stderr сохранены в evidence/confirmation_cpu_scope_failure с SHA.
Это CPU-проверка обвязки: confirmation job не отправлялся, fits0.

## Подготовка публикации

Независимый reviewer проверил runtime/audit_saved.py (12 stdlib tests PASS)
и runtime/append_registry.py. Второй инструмент по умолчанию выполняет dry-run;
--write разрешён только после сохранения terminal evidence и полного audit PASS.
Он сохраняет исходные 125 строк / 58287 байт и добавляет отдельные fits.
Реестр не изменён. RESULTS.md содержит проверенные вводные и явный pending.
Полный terminal audit ещё не выполнялся. При изменении manifest locator для
confirmation согласованно обновить preserve_terminal, audit_saved и append_registry.
