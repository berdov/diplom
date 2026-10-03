# Time-addressed memory pilot: attempt002, job4373393

Phase: AUDITED_AWAITING_PUBLICATION. Job4373393 COMPLETED0:0, cn-044,01:31:29,
2026-10-03 19:17:54–20:49:23MSK. All3fits complete:39/42/76epochs; TEST0.
Independent audit PASS:157epochs,1884metriccells,exact control replay+checkpointSHA,
CPU85+85,GPU11/429,smoke3×3,412publishedsourceblobs,ownership/runtime/checkpoints verified.
No new forwards or weightloading.55rawfiles preservedSHA;8canonicalJSON copies verified.
NDCG@10 no/index/time=.0633/.0626/.0634;first27=.0620/.0611/.0613.
Primary time−index+.0008;time−no+.0001,first27−.0007. Weak single-seed signal;
no automatic confirmation or replacement of baseline. Reports/registry publication remains.
[Exact snapshot](status_4373393.json), [state](state.json).

## Рабочие каталоги и версия

- Local: `/Users/berdov/diplom`.
- Cluster: `/home/daryumin/iberdov/diplom`, SSH `hse-karizma`.
- Ветка: `exp/mamba3-time-addressed-memory`.
- Main перед пилотом: `7ba5e45ed110914b354c89d4377836910b55408c`.
- Execution attempt002: `53bcc76752d0d7fd06c8ba0866709a64cd978165`.
- Source: `ad4565d171b78c83831ef84c139a4912af2022a04fdb39fdfd5b4a858461f5e6`, 412 файлов, `source_manifest_002.json`.

На кластере остаётся точный execution commit, tracked clean. Не менять исходники,
пока job их использует. Остальные активные jobs работают в других каталогах и не затрагивались.
Исходное задание для восстановления контекста:
`/Users/berdov/.codex/attachments/4de68a7a-1c0e-4e3e-89d7-b6b8a1d03f62/Вставленный текст.txt`.

## Конструкция и бюджет

Три свежих TRAIN→VALID запуска seed2026: no_memory → index_memory → time_memory.
Параметры: 715020 / 715021 / 715021. Основа — прежний MIMO dual, fixed R0=838393ms,
shared temporal functions, rank4/chunk8, два слоя и две головы, история50.
Архивируются исходные причинные представления H после output_norm, а не SSM-матрицы S_t.
K=4, anchors=[1,4,16,32], только j<t; greedy nearest в log1p, при ties последний j,
уникальные позиции, сортировка выбранных slots. FP32 reader dot/sqrt64, lambda=tanh(beta),
один fp32 beta с начальным точным нулём. Нет новых Q/K/V, нормализации и persistent cache.
Primary: time−index; оба сравнения с fresh no_memory — secondary.

Job4373393: одна A100, CPU4, mem0, rocky/proj_1833/type_e, no-requeue, **до4 часов**.
**Использованы оба submit. Третья отправка и автоматический повтор fit запрещены.**
Всего допускаются три начатых scientific fit; TEST=0. Нельзя добавлять seeds, confirmation,
другую память, продолжение обучения, длинную историю или правки статьи. Сообщения не отправлять.

## Первый job и исправление

Job4373262 использовал execution55d812bf55b1dffbbab6a7b0da86e616a6227b8b,
source4e26434f8bf084d6acd562e6859c24699013510d8f2ffcaaf2653435d3698e8d, 410 файлов.
В18:30:29MSK он отменён только в состоянии PENDING после воспроизведения ошибки обвязки.
Start=None, elapsed00:00:00, node=None assigned; allocation, gate, smoke и fit не начинались.
Причина отмены — ошибка создания progress, не длительность очереди и не численный результат.
13 компактных файлов сохранены с SHA до исправления, preservation commit3b2832b.
[Проверка отмены до запуска](../evidence/job4373262/pre_fit_cancellation_audit.json).

Первый update пытался записать ещё не созданный progress.json. Отдельная ошибка report.write
теряла неизвестный старт при lock без результата. Исправлены четыре файла обвязки,
добавлены18 regressions; 406 прежних файлов не изменились. AST функции обучения,
основного pipeline и проверки научных результатов совпадают с первой версией.
[Основание технического retry](retry_review.json), [ошибка progress](progress_initialization_repro.json),
[ошибка unknown counter](report_unknown_start_repro.json), [regression evidence](wrapper_regression.json).

## Проверки перед последней отправкой

Exact execution: CPU85/85 и no-Git85/85 PASS, без errors/failures/skips, CUDA не инициализирована.
TRAIN-only coverage:10000 фиксированных входных историй. Входы и вся диагностика точно совпали
с attempt001; отличаются только шесть полей версии/attempt/времени выполнения.
Targets, loaders, модели, model forwards и CUDA в coverage не используются.
Разные selected sets —89.02%, минимум четыре прошлых события —91.27%, пустая память —2.11%.
Средний span206036133.1549ms, максимум1026592333ms; достижимы anchors1/4/16/32R0
в97.21/96.31/94.12/91.08% историй. Anchors не менялись.
Coverage SHA: `d19a9ebdef16c6cfa9c535148096b0fd9b4cee5e818de335a3dbdbbaa21ec40c`.

На GPU ожидаются11 cases/429 leaves, затем smoke3×3, затем три fit.
Унаследованы MIMO kernel evidence45 cases/2342 checks по SHA.
В negative_evidence ожидаются отрицательные raw checks намеренно leaking fixtures.
Движение beta на случайном smoke batch не является дополнительным критерием PASS.

## Следующие действия

Только один компактный read-only poll с интервалом не менее600 секунд:
`python3 experiments/mamba3_time_memory/runtime/capture_status.py 4373393 53bcc76752d0d7fd06c8ba0866709a64cd978165 --attempt 002`

Helper читает компактные progress, не histories или weights, и обновляет state/HANDOFF.
Не запускать отдельные частые запросы scheduler. Если очередь превысит4 часа или сессия
закончится, сохранить реальное состояние. Не отменять и не переотправлять job ради очереди.
Не обещать работу в фоне после окончания сессии.

После установленного terminal status:
`python3 experiments/mamba3_time_memory/runtime/preserve_terminal.py 4373393 53bcc76752d0d7fd06c8ba0866709a64cd978165 --attempt 002`

Helper считает SHA/bytes весов потоком на кластере, не загружает их и не десериализует;
сохраняет также retry_review. Затем независимый stdlib audit:
`PYTHONNOUSERSITE=1 python3 -B -m experiments.mamba3_time_memory.runtime.audit_saved experiments/mamba3_time_memory/evidence/job4373393 --execution 53bcc76752d0d7fd06c8ba0866709a64cd978165 --cpu-tests 85`

Проверить каждую metric cell против логов, last-tie/early stopping/first27, metadata и SHA
checkpoint, точный historical replay контроля, pairing, source/runtime/coverage ownership, TEST0.
Quantiles приближённые по reservoir256; опубликованы только восемь примеров. Полностью
восстановить агрегаты или quantiles по этим восьми примерам нельзя.

После audit PASS и3/3 completed: preservation commit, затем полный RESULTS.md в пакете,
раздел reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#time-addressed-memory-pilot,
краткая запись reports/RESULTS.md и ровно три новые строки registry122→125.
Прежние56821 байт CSV и header должны сохраниться точно; исходный SHA:
`073435a00d73646b4ed2c442a2538cd60338090a0bf00ae18902ca8a9181e79e`.
Завершённый отрицательный результат тоже публикуется. После проверки diff разрешён обычный
merge/push main. Вывод ограничить readout причинных представлений текущего окна50 событий.
Финал A–K по заданию, включая короткий русский текст в нижнем регистре с обращением на«вы»
и полными GitHub URL отдельными строками; сообщение не отправлять.
При incomplete сохранить evidence и partial report в ветке, не объявлять пункт полностью
завершённым. При любом исходе третьего submit нет.
