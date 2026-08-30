# Отчёты KuaiRand benchmark

Этот каталог разделяет уровни evidence для дипломного benchmark.

| Файл | Смысл |
| --- | --- |
| [paper_results_kuairand_protocol_b.md](paper_results_kuairand_protocol_b.md) | Published literature benchmark: paper TEST results, не наши запуски. |
| [paper_results_provenance.csv](paper_results_provenance.csv) | Machine-readable provenance для каждого paper metric. |
| [protocol_comparability.md](protocol_comparability.md) | Почему Protocol B strongly comparable с SSD4Rec/TiM4Rec, но не exact byte-level. |
| [paper_vs_reproduction.md](paper_vs_reproduction.md) | Сравнение published SSD4Rec/TiM4Rec rows с нашими canonical TEST reproductions. |
| [our_experiments_protocol_b.md](our_experiments_protocol_b.md) | Наши actual runs: canonical TEST, validation-only screening, historical/exploratory/sanity. |
| [project_status_summary.md](project_status_summary.md) | Master summary всей истории project experiments по уровням evidence. |
| [SUPERVISOR_SUMMARY.md](SUPERVISOR_SUMMARY.md) | Короткая страница для обсуждения с руководителем. |
| [pcgrad_discrepancy_audit.md](pcgrad_discrepancy_audit.md) | Forensic audit разрыва между historical и current PCGrad. |
| [tuning_design.md](tuning_design.md) | Frozen design controlled validation-only tuning для EPO, GradHV, COSMOS и PCGrad. |

Правило чтения: paper results и our experiments не объединяются в одну scientific ranking table. MOO convergence в этой ветке является validation-only family screening, поэтому он не сравнивается напрямую с published TEST rows.

## Политика артефактов

Compact summaries, CSV, Markdown provenance and report plots можно хранить в Git. Raw run directories, partial JSON, checkpoint/model weights, per-step jsonl и Slurm logs должны оставаться на кластере или игнорироваться локально.

Tracked inventory перед этим documentation stage: 307 files, 18,615,506 bytes (17.753 MiB). Крупнейшие tracked artifacts — compact EDA CSVs и compact MOO run JSON summaries; tracked Slurm logs/checkpoints/partial JSON во время audit не найдены.

Raw cluster artifact locations, на которые ссылаются текущие reports: `/home/daryumin/iberdov/diplom_exp_moo_8families/experiments/moo_8families/*_convergence_001/` и baseline stage directories, записанные внутри run JSON files.
