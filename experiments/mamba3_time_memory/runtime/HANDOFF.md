# Пилот временной адресации памяти

Phase: PUBLICATION_REVIEW. Все эксперименты и аудит завершены. Отчёты подготовлены,
реестр дополнен с 122 до 125 строк; осталось закончить проверку публикации,
создать commit и выполнить разрешённые обычные merge/push main.

## Результат

Job **4373393**, attempt002: **COMPLETED 0:0**, cn-044, A100-SXM4-80GB,
**01:31:29**, 3 октября 2026, 19:17:54–20:49:23 MSK.
Scientific fits: **3 начаты, 3 завершены**, неизвестных стартов 0; TEST=0.
NDCG@10 no/index/time: **0.0633 / 0.0626 / 0.0634**;
first27: **0.0620 / 0.0611 / 0.0613**; эпох **39 / 42 / 76**.
Time−index +0.0008; time−no_memory только +0.0001, first27 −0.0007.
Это слабый сигнал на одном seed. На VALID 89.18% time anchor choices лежат
вне диапазона оставшихся кандидатов. Рабочая модель сохраняется.

Проверено чтение причинных h_j внутри текущего окна 50 событий, не полная
SSM-матрица и не persistent memory. K=4, anchors=[1,4,16,32], R0=838393ms,
lambda=tanh(beta), beta первоначально 0; параметры 715020/715021/715021.

## Каталоги и версии

- Local: `/Users/berdov/diplom`.
- Cluster: `/home/daryumin/iberdov/diplom`, SSH `hse-karizma`.
- Ветка: `exp/mamba3-time-addressed-memory`.
- Main перед пилотом: `7ba5e45ed110914b354c89d4377836910b55408c`.
- Execution: `53bcc76752d0d7fd06c8ba0866709a64cd978165`.
- Source: `ad4565d171b78c83831ef84c139a4912af2022a04fdb39fdfd5b4a858461f5e6`, 412 файлов.
- Preservation commit: `a7aecd5129e6eb03f02d0ad11b0c41cec0c0d4b0`.

## Evidence и проверки

CPU85/85 и no-Git85/85 PASS, GPU11/429 PASS, smoke3 режима × 3 шага PASS.
MIMO kernel evidence45 cases/2342 checks унаследовано по SHA.
Независимый terminal audit PASS:157эпох,1884metric cells в двух логах,
pairing, last-tie/early stopping, metadata и streaming SHA весов,
точный replay исторического no_memory2026. Это не новый независимый seed.
55 raw файлов сохранены с SHA, 8 canonical JSON копий byte-identical.
Веса остаются на кластере; новых forwards и десериализации для отчётов не было.

Первый job4373262 отменён в PENDING, до allocation и любого fit, после
локального воспроизведения ошибок progress/unknown-start accounting.
13 compact artifacts сохранены до исправления; preservation commit3b2832b.
Изменены только четыре wrapper-файла, добавлены18 regressions;
406 прежних исходников и проверенные научные функции не менялись.
[Retry review](retry_review.json), [parent audit](../evidence/job4373262/pre_fit_cancellation_audit.json).

TRAIN coverage10000: selected sets различаются89.02%, ≥4past91.27%, empty2.11%.
Source и диагностика между attempts совпали; менялись только execution metadata.
Quantiles приблизительные по reservoir256 запросов; опубликованные8 примеров
не позволяют заново восстановить все агрегаты.

[Полный отчёт](../RESULTS.md) · [Raw summary](../runs/attempt_002/pilot_summary.json) ·
[Preservation](../evidence/job4373393/preservation_manifest.json) ·
[Аудит](../evidence/job4373393/independent_audit.json) · [State](state.json).

## Бюджет и следующий безопасный шаг

**Submissions2/2, scientific fits3/3. Новых submit, fits и polling не требуется.**
Не выполнять TEST, новые seeds, confirmation, continuation, новые model forwards,
загрузку весов или расширение истории. Статья и Overleaf не меняются.
Сообщения другим людям не отправляются.

После публикации остановиться. Возможная будущая confirmation этой же
конструкции требует отдельного решения; текущий пилот её не запускает.
При восстановлении контекста читать state и публикационные commits,
не начинать с нового submit. Старые raw logs и terminal snapshots не исправлять.
