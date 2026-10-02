# Centered Gap-Trap: job4371302 завершился до обучения

**INCOMPLETE.** Slurm: FAILED, exit1:0, cn-043, 2 октября 2026,
21:36:50–21:40:20 MSK, elapsed3:30. Scientific fits0/2, TEST0.
Первая проверка gate остановилась до forward/backward. Smoke не запускался,
историй обучения и checkpoint нет. Научного результата centered-пилота нет;
пункт3 остаётся незавершённым.

Причина: `centered/state.py:view()` безусловно читает `gap_trap_mode`.
В GPU parity источник — исторический `ThreeTimeMamba3Rec`, у которого этого
поля нет. `transfer_common(old,new)` падает до создания histories и
`model_measure`. Это ошибка адаптера проверки, не отрицательный результат
гипотезы. CPU-тесты проверяли перенос между двумя экземплярами новой модели
и не покрывали historical→new. Их прежние PASS сохранены без изменения.

- Canonical local: `/Users/berdov/diplom`.
- Canonical cluster: `/home/daryumin/iberdov/diplom`, SSH `hse-karizma`.
- Ветка: `exp/mamba3-gap-trap-centered`.
- Execution commit: `4347ecbd3bfa98b075dbf0dd8f227bfff3590e50`.
- Source hash: `8593838aedf9abd976db8a38367de3697716be1ae6661426c3acb52d3ab5d2fa`.
- Exact-source CPU/no-Git: по16 PASS, zero skips/errors/failures, CUDA false.
- GPU: первый case начал setup, ноль numerical checks выполнено; остальные
  пять cases и оба fit NOT_RUN. OOM не обнаружен.
- Cluster checkout остаётся на execution4347ecb, tracked clean.
- Один разрешённый job использован. Retry/continuation/requeue не отправлялись.

[Terminal scheduler](../evidence/job4371302/scheduler_terminal.json),
[22 сохранённых файла и SHA](../evidence/job4371302/preservation_manifest.json),
[аудит сбоя](../evidence/job4371302/failure_audit.json).
Сохранено139542bytes; все hashes и306 execution blobs проверены. Audit PASS
означает согласованность сохранённых фактов, а не успешность эксперимента.
[Первоначальный PENDING snapshot](status_4371302.json) сохранён как история.
`audit_saved.py` рассчитан только на два успешных fit и для этого сбоя не запускался.

## Опубликованное ранее

One-sided pilot в main: `f533a49038a5a055a19837bb2dfde4518f10e3f4`,
publication `824ebbedcbcc8816a99a5f3c4df61429bc12accd`. Registry110rows
побайтно совпадает с этим main. Новых scientific rows не добавлено.
Старая ветка `exp/mamba3-gap-trap` остаётся на824ebbe.
[Результат one-sided](../../../../reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#gap-trap-one-sided-pilot).

При deploy семь совпадающих untracked parent artifacts были перемещены с
manifest в `.codex_deploy_preservation/centered_phase_a_20261002/` внутри
canonical cluster repo. Файлы и checkpoint/caches не удалялись.

## Остановка

По заданию при gate FAIL scientific fits не запускаются, новый job/retry
запрещён. Выполнен terminal read-only audit, код execution не менялся.
Исправление должно учитывать отсутствие специального mode у исторической
модели и покрывать historical→new отдельным CPU regression fixture; оно здесь
не применялось. Продолжение требует отдельного задания, разрешающего новый
контролируемый запуск после исправления. Существующий job повторять нельзя.

Нет оснований интерпретировать delta, выбирать модель, объявлять пункт3
закрытым или рекомендовать confirmation по centered-результату. TEST, пункт4/5,
статья и Overleaf не затронуты. Итоговое сообщение с научными выводами не
составлено и никому не отправлено. Фоновый мониторинг не работает.
