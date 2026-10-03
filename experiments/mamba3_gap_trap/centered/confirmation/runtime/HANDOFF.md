# Centered Gap-Trap confirmation: завершена

Job **4372023** завершился **3 октября 2026, 10:36:41 MSK**, `COMPLETED 0:0`,
node `cn-044`, elapsed **02:48:56**. Все **8/8** запусков и **4/4** пары завершены
в одном allocation. Continuation и повторных запусков не было, TEST = 0.

[Статус](handoff.json), [scheduler](../evidence/job4372023/scheduler_terminal.json),
[полный аудит](../evidence/job4372023/independent_audit.json),
[preservation manifest](../evidence/job4372023/preservation_manifest.json).

Проверены 407 эпох, 4884 значения VALID-метрик, обе копии логов, train loss,
last-tie/early stopping, 8 checkpoint SHA, начальные состояния/RNG/первые batch.
Все 4 fresh fixed точно воспроизводят исторические runs. Веса остаются на кластере,
в Git сохраняются только пути, размеры и SHA; при аудите веса не загружались.

Execution: `23468e74389aed841aab0ac16b370c5ce376e328`.
Source: `ec0ab4281cb7fdba01e0f8465d7c144895894038eeed2517aaa633ebc157bef5`, 358 файлов.
CPU 38/38 и no-Git 38/38 PASS; GPU 6/158 и smoke унаследованы по SHA.
Cluster checkout остаётся на execution; теперь удалённую ветку можно продвинуть
до сохранённых evidence и публикации. Дополнительных запусков не требуется.

На новых seeds centered имеет 2 выигрыша и 2 проигрыша, paired Δ −0.000025.
Устойчивое преимущество не подтвердилось; MIMO dual fixed-reference остаётся основой.
First27 доступен для трёх пар: centered2030 остановился после 23 эпох.
Пункт 3 завершён. Пункт 4 и статья не менялись.

Полный результат: [отчёт](../RESULTS.md),
[SVG и TeX](../../../../../reports/assets/gap_trap_centered_confirmation/),
[проверка реестра](publication_audit.json). Реестр вырос с 112 до 120 строк,
старые байты сохранены. Preservation commit:
`ec7560f40d0977f5f6290af70b2390fa09aa1b0d`.
Publication commit включает полный результат и изменение реестра; merge отражён в истории main.
Фонового мониторинга нет, новых jobs не подавать.
