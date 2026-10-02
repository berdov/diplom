# Centered Gap-Trap: job4371302 ожидает ресурс

На 2 октября 2026, 19:35:55 MSK: **PENDING (Priority)**. Оценка старта Slurm:
3 октября, 06:28 MSK; она может измениться. Node не назначен, elapsed0,
scientific fits0/2, TEST0. GPU gate и smoke ещё не начинались. Это не результат
centered-пилота; пункт3 пока нельзя закрыть по фактическим метрикам.

- Canonical local: `/Users/berdov/diplom`.
- Canonical cluster: `/home/daryumin/iberdov/diplom`, SSH `hse-karizma`.
- Ветка: `exp/mamba3-gap-trap-centered`.
- Execution commit: `4347ecbd3bfa98b075dbf0dd8f227bfff3590e50`.
- Source hash: `8593838aedf9abd976db8a38367de3697716be1ae6661426c3acb52d3ab5d2fa`.
- Источник: 306 SHA, включая все278 неизменённых parent dependencies.
- Один job4371302 уже отправлен; reservation/submission сохранены.
- Exact-source CPU и no-Git: оба PASS, по16 tests, zero skips/errors/failures,
  CUDA initialized=false. Предварительные CPU-проверки старых revisions остаются
  на кластере; scientific fits они не запускали.
- GPU plan: 6cases/158required leaves, затем synthetic smoke2x3steps,
  затем fixed_replay и centered_gap_trap seed2026 в свежих процессах.
- A100x1, rocky/proj_1833/type_e, CPU4, mem0, 6h, no-requeue.

[Снимок scheduler и checkout](status_4371302.json),
[сохранение submission evidence](../evidence/job4371302/submission_preservation.json).

## Уже опубликованное

One-sided pilot опубликован в main: `f533a49038a5a055a19837bb2dfde4518f10e3f4`,
publication `824ebbedcbcc8816a99a5f3c4df61429bc12accd`. Registry108→110;
прежние49702bytes сохранены точным префиксом. После Phase A registry не менялся.
Старая ветка `exp/mamba3-gap-trap` остаётся на824ebbe.
[Краткий результат](../../../../reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#gap-trap-one-sided-pilot).

При deploy семь совпадающих untracked parent artifacts перемещены с manifest в
`.codex_deploy_preservation/centered_phase_a_20261002/` внутри canonical cluster
repo. Ничего не удалялось. Папки caches/checkpoints и другие untracked сохранены.

## Следующая проверка

1. Только read-only status существующего job4371302: squeue/sacct и stage JSON.
   **Не отправлять sbatch, continuation, requeue или повторный fit.** Не менять
   cluster checkout, пока job PENDING/RUNNING. Он закреплён на execution4347ecb;
   более поздние branch commits содержат только preservation/handoff tooling.
2. При terminal state сохранить точный `sacct` и stderr/stdout/process logs,
   все компактные JSON/Markdown/locks, metadata и history под
   `evidence/job4371302/files/` с исходными путями относительно centered.
   Сохранить `preservation_manifest.json`: `checkout_commit`, `files` с
   `path/bytes/sha256`, `checkpoints` с исходным relative path→SHA/bytes.
   Checkpoint SHA читать streaming, weights не десериализовать. Файлы не удалять.
3. Сохранить `scheduler_terminal.json` с `job_id`, `state`, `exit_code`, точным
   raw sacct и временем наблюдения. Для full PASS нужны COMPLETED и0:0.
4. Только если оба fit завершились: `python3 -m
   experiments.mamba3_gap_trap.centered.runtime.audit_saved --job 4371302`.
   Это подготовленный stdlib-аудитор terminal artifacts, ещё не выполнялся:
   PENDING artifacts для него недостаточны. Он проверяет source/preservation,
   ownership, GPU leaves, smoke, все epoch metrics/loss из двух logs, selection,
   checkpoint hashes/metadata, exact historical fixed replay, pairing, alpha,
   analytic/BF16 diagnostics, timings/memory и summary. Ошибки аудита расследовать
   read-only; запуск новых fits не разрешён.
5. При technical failure или неполной серии сохранить факты и остановиться.
   При full PASS сохранить raw results в `runs/attempt_001/`, summary и краткий
   RESULTS.md; append ровно2 scientific rows в registry (перед этим110rows,
   старые bytes должны сохраниться). Добавить concise anchor
   `#gap-trap-centered-pilot` в reports/MAMBA3_TIME_MECHANISMS_RESULTS.md и строку
   reports/RESULTS.md, commit/push/merge в main. Не менять результаты прошлого.
6. После полного результата закрыть пункт3 в текущем KuaiRand/VALID exploratory
   scope. Primary — NDCG@10; HR/first27/epochs/alpha показывать честно.
   При положительной delta только рекомендация отдельного решения о paired
   seeds2027–2030. Иначе никаких третьих Trap-функций.

TEST запрещён. Нет tuning/дополнительных seeds/пункта4/5/статьи/Overleaf.
Layer-specific temporal functions — только рекомендация будущего отдельного
эксперимента. Итоговое сообщение для обсуждения готовится после фактического
результата, только текстом пользователю; никому не отправлять.
Фоновый мониторинг отдельным процессом не запускался.
