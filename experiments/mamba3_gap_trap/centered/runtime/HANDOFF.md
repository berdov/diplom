# Centered Gap-Trap: завершено

Job4371876, attempt002: **COMPLETED 0:0**, cn-043, 00:50:21.
Старт 2 октября 2026, 23:26:27 MSK; конец 3 октября, 00:16:48 MSK.
Оба scientific fits завершены, TEST0; [terminal Slurm](../evidence/job4371876/scheduler_terminal.json).
Первоначальный [PENDING snapshot](status_4371876.json) сохранён как история.

Fixed/centered: VALID NDCG@10 **0.0633/0.0635**, first27 **0.0620/0.0615**,
HR@10 **0.1162/0.1189**, best epoch27/48 (с нуля), всего39/60эпох.
Небольшой положительный pilot одного seed; устойчивость не установлена.
Пункт3 закрыт в текущем exploratory KuaiRand/VALID scope.

- Canonical local: `/Users/berdov/diplom`.
- Canonical cluster: `/home/daryumin/iberdov/diplom`, SSH `hse-karizma`.
- Ветка: `exp/mamba3-gap-trap-centered`.
- Execution: `e4e31370e29c2a34c5f0f1071046fccebd77c289`.
- Source hash: `638b836780af7b6f9065546c128cb8ac2451c5d8af3581855029eea515c2ca10`.
- `source_manifest_002.json`: 333 файла; `study_plan_002.json`.
- CPU/no-Git21/21, GPU6cases/158checks, smoke6steps: PASS.
- [Независимый аудит](../evidence/job4371876/independent_audit.json): PASS.
  Сверены все99эпох/1188метрик, paired initialization, исторический fixed и SHA.
- [Preservation](../evidence/job4371876/preservation_manifest.json):35исходныхфайлов,
  оба checkpoint SHA прочитаны streaming. Weights при аудите не загружались.
- Canonical результаты: `runs/attempt_002/`, [RESULTS](../RESULTS.md),
  [диагностика](../diagnostics.json), [publication audit](publication_audit.json).
- Registry110→112: добавлены только два фактических fit, прежние bytes сохранены.

Attempt001/job4371302 сохранён отдельно: FAILED на setup gate, 0forward/0fits,
TEST0. Причина — применение centered adapter к историческому source;
исправление ограничено `state.transfer_common`, covered реальной исторической моделью
и отрицательными CPU fixtures. [Разбор](../evidence/job4371302/failure_audit.json).
Старые artifacts не менялись; научная формула и протокол прежние.

Работа по этому заданию закончена. Не отправлять sbatch/requeue/continuation,
не запускать новые seeds/Trap-варианты/TEST. Следующий пункт — layer-specific
temporal functions — только рекомендация отдельного задания. Парную confirmation
2027–2030 можно рассматривать отдельным решением; сейчас она не запускалась.
Статья/Overleaf не менялись. Сообщения другим людям не отправлялись.
Cluster checkout оставлен на execution commit, исторические файлы не удалялись.
Фонового мониторинга нет.
