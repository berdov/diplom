# Обучаемые временные масштабы: подтверждение

Четыре заранее выбранных seed2027–2030, по три новых запуска:
fixed → shared_tau → head_tau. Основной контраст head_tau−shared_tau;
head_tau−fixed и shared_tau−fixed сохраняются независимо от знака.
Пилот2026 опубликован отдельно и не переобучается. TEST=0.

`study_plan.json` фиксирует порядок, бюджет, настройки и правила сводки.
`source_index.json` задаёт единственный физический источник каждого logical run.
Никакого поиска «последнего успешного» результата. Все модели начинают с нуля;
settings копируются из соответствующего успешного JSON пилота, меняются
только seed и checkpoint_dir. Модель, calibrators, trainer и kernels прежние.
Trainer по-прежнему читает diagnostic_grid старого неизменного плана;
новый план проверяет его точное равенство, dependency входит в manifest.

До submit: CPU integration tests и15 construction-only setup replays,
включая seed2026 без forward/fit/evaluation. Canonical Adam проверяется после
настоящего JSON roundtrip. CUDA RNG/loader/first batch не подменяются CPU replay:
их парность проверяется в allocation до fit и на первом обычном train batch.
CPU mask действует только на subprocess, GPU children сохраняют Slurm mask.
Runtime content checks не вызывают Git; Git/publication проверяются на login.

Gate9/228 и smoke job4362620 наследуются по SHA и полной проверке leaves.
Kernel admission45/2342 относится к job4358147. Повторных GPU checks нет.

Основная allocation: rocky/proj_1833/type_e, A100×1, CPU4, mem0,8h,no-requeue.
За10 минут до Slurm walltime действует внутренний deadline; новый fit не
начинается, если осталось менее90 минут. Epochs300/stopping_step10 неизменны.
Каждый fit запускается отдельным процессом; scientific_fit_started сохраняется
до trainer.fit. Новые настройки по метрике не выбираются.

Максимум12 начатых fits и2 submissions,8h+не более4h. Единственное продолжение
допустимо только после terminal первого job, полного preservation и regression
конкретного сбоя orchestration до fit либо deadline-before-start. Любой начатый
незавершённый fit, численная ошибка или неизвестный start-state блокирует его.
`continuation_authorization.json` с точными SHA хранится в runtime; второй
source_index сохраняет PASS первого job и назначает новые paths только
достоверно не начатым runs. Третьего submit нет; старые files не перезаписываются.

Reservation и submission intent создаются до sbatch. Неоднозначный SSH/sbatch
ответ требует восстановления факта отправки, не повторной команды.
`runtime/state.json` и `runtime/handoff.md` — единственная служебная точка
возобновления, вне frozen inputs. Секреты и полный environment туда не пишутся.
Мониторинг не чаще600 секунд; ожидание ограничено16 часами от первой отправки
и4 часами суммарной очереди. Работа не обещается после окончания сессии.

Pipeline: runtime/inherited admission →12 subprocess fits → JSON/Markdown.
При ошибке сохраняются исходный traceback, все частичные records и NOT_RUN.
Картинка и TeX производны и не управляют сохранностью научных результатов.
После terminal — отдельный read-only audit histories/logs/metadata/streamed
checkpoint SHA без десериализации checkpoint и повторного inference.

Сводка разделяет четыре новые тройки и пять с exploratory pilot. Sample std
с ddof1, все парные разности и знаки, relative gain по одинаковым seed sets.
First27 требует реальные полные эпохи0–26; каждая пара имеет своё пересечение,
короткий третий run её не исключает. Это тот же history slice, не независимая
репликация и не строго равный GPU-бюджет. Нет новых p-values, TEST, SOTA или
автоматического выбора backbone. R — глобальный масштаб нормировки; выводы
о персональных периодах и специализации голов по нему не делаются.
