# Centered Gap-Trap confirmation: job4372023

На **3 октября 2026, 01:06:53 MSK**: **PENDING (Priority)**,
node не назначен, elapsed00:00:00. Pipeline ещё не начался.
Scientific fits0/8, complete pairs0/4, TEST0.
Следующий scheduler poll — не раньше01:16:53MSK; интервал минимум10мин.
[Exact snapshot](status_4372023.json), [структурированный статус](handoff.json).

## Зафиксированный запуск

- Canonical local: `/Users/berdov/diplom`.
- Canonical cluster: `/home/daryumin/iberdov/diplom`, SSH `hse-karizma`.
- Ветка: `exp/mamba3-gap-trap-centered-confirmation`.
- Execution: `23468e74389aed841aab0ac16b370c5ce376e328`.
- Source hash: `ec0ab4281cb7fdba01e0f8465d7c144895894038eeed2517aaa633ebc157bef5`,358файлов.
- Cluster checkout остаётся на execution. Не checkout/pull во время allocation.
- [Submission, reservation, login и оба CPU preflight](../evidence/job4372023/submission_preservation.json).
- CPU38/38 и no-Git38/38 PASS,0failures/errors/skips, CUDAfalse.
- GPU6cases/158checks и smoke наследуются от job4371876 по точным SHA;
  прежние one-sided4/133, head9/228, MIMO45/2342 сохранены в lineage.
- Модель/modulation/trainer/initial-state опубликованного centered pilot не менялись.
- Resources: rocky/proj_1833/type_e, A100x1, CPU4, mem0,8h,no-requeue.

## Оставшийся порядок

2027: fresh fixed_replay → centered_gap_trap.
2028: fresh fixed_replay → centered_gap_trap.
2029: fresh fixed_replay → centered_gap_trap.
2030: fresh fixed_replay → centered_gap_trap.

Каждый процесс начинает TRAIN с нуля. Fixed обязан точно повторить historical
MIMO dual того же seed до старта centered. После пары сохраняется pair_seed*.json.
Не запускать seed2026, новые варианты, TEST или пункт4. Плохие метрики не являются
основанием прерывать оставшиеся заранее выбранные пары.

## Мониторинг и отказ

Проверять только существующий job4372023: squeue/sacct и
`slurm_logs/attempt_001/pipeline_status.json`, не чаще примерно10мин.
У fit results один physical path: `runs/mamba3_gaptrap_centered_confirm_*_seed*_001.json`;
logs/checkpoints находятся в `slurm_logs/fits/<run_id>/`.
Не подавать повторный sbatch, не requeue. Reservation001 уже использована.

При technical FAIL сохранить stage/error/traceback/logs/partial results,
действительный started count и неизвестные starts, если evidence неполное.
Никакого автоматического resubmit. При повреждённой записи summary остаётся
INCOMPLETE, непроверенные метрики не агрегируются; raw не исправляется.

Штатный deadline guard (остаток<90мин до внутреннего deadline,10мин margin)
между fits возвращает PAUSED_DEADLINE, exit0. Только после terminalCOMPLETED0:0,
полного preservation и проверки completed prefix/never-started suffix допустим
ровно один continuation002. Same execution/source/config; максимум8starts суммарно.
При started-incomplete/unknown/OOM/replay mismatch continuation запрещён.

Continuation admission ещё НЕ создавался. Его нужно сформировать по terminal
архиву с exact scheduler snapshot, remaining_tasks и preserved_files path→SHA
для login/reservation/submission/pipeline/terminal/summary, всех completed
results/metadata/locks, process logs и pair JSON. `continuation.validate` проверяет
эти bindings, затем `submit --attempt002` резервирует только suffix.
CPU/no-Git proof для002 — тот же exact execution, отдельные operational paths.
До решения о continuation удалённая confirmation branch сохраняется на exact
execution23468e7: этого требует frozen login_verify. Handoff/evidence сохранены
отдельным локальным commit в той же ветке; пока не push этот commit. Cluster
checkout и origin tracking ref остаются на execution. Если понадобится002,
сначала выполнить его admission/preflight/submit на том же execution; после
последней submission либо полного завершения001 можно push сохранённые evidence.
Это не требует изменения source или force-push. Сейчас continuation не требуется.

## Terminal audit и публикация

После завершения сохранить sacct и все compact JSON/log/md/out/err/lock файлы
из `runs/` и `slurm_logs/`, исключая caches/TensorBoard/checkpoint weights.
Checkpoint читать только streaming SHA+bytes, веса оставить на кластере.
Preservation manifest должен содержать execution/source, source paths/bytes/SHA,
checkpoint paths/bytes/SHA, scheduler. Ничего на кластере не удалять.

Проверить8results/metadata, все epoch metrics/train loss по RecBole log и process
stderr, early stopping/last-tie, checkpointSHA, TEST0 и4exacthistoricalfixed.
Сверить каждую пару и pair artifact. Alpha best/max/final/zeros/trajectory,
аналитическую grid и BF16 пересчитать без model/dataset forward.
Для exp/sigmoid допустим только прежний1ULP libm tolerance с журналом расхождений;
SHA/метрики/BF16 проверять строго. Existing centered/runtime/audit_saved.py
может служить основой, но он рассчитан на2fits и не является готовым аудитом8fits.

Primary — только новые4pairs, mean±sample std(ddof1), paired delta/signs и
relative difference of means. All5сpilot2026 отдельно, first27 отдельный paired
available subset. Показать best/actual epochs; не приписывать эффект extra epochs
причинно. Alpha — global scalar strength, не пользовательская шкала времени.

Публикация только8/8fits+4/4pairs+terminalauditPASS+TEST0. Иначе evidence branch-only.
После COMPLETE: concise section #gap-trap-centered-confirmation в основном отчёте,
короткая строка reports/RESULTS.md, ровно8rows в registry с byte-identical прежним
prefix. Сейчас112rows; ожидается120, но пересчитать фактически. Создать
reports/assets/gap_trap_centered_confirmation/paired_delta.svg и valid_table.tex.
В main merge разрешён только полный результат. Main сейчасe8f8f1e, прежние
pilot branches не менялись. Статья/Overleaf и пункт4 остаются вне задания.

По результатам дать нейтральное решение о fixed/centered provisional backbone;
нового Trap tuning не будет. Сообщение руководителю подготовить в чате, не отправлять.
Фонового мониторинга нет.
