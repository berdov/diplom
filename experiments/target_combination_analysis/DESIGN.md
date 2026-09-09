> Retry 002: первая smoke-попытка была PREEMPTED, array отменён до запуска.
> Все новые IDs заканчиваются на _002; новый изолированный checkout:
> `/home/daryumin/iberdov/diplom_exp_target_combinations_002`.
> Smoke, array и CPU-summary переведены в rocky (PreemptMode=OFF,
> PriorityTier=10 против quick Tier=5/REQUEUE). GPU A100, type_e, CPU×4,
> 3h, concurrency 4 и все scientific hyperparameters сохранены.
> Старый checkout и его результаты остаются неизменными.
> Ниже — исходный дизайн с историей выбора quick partition.

# Full 2^4 auxiliary-target screening

База: `16ba99b69929a711137f616df28e18e7f92b76c1` (актуальный origin/main при старте).
Ветка: `exp/target-combination-screening`. Отдельные local worktree и cluster checkout.

Primary next-item включён всегда. Четыре фактора в bitmask-порядке:
A=is_click, B=long_view, C=is_like, D=is_profile_enter. Manifest combinations.yaml
задаёт 16 строк в порядке размера subset, затем combinations(A,B,C,D), индексы 0–15.
Ни одна историческая строка не заменяет новый scientific run.

Для непустого S: `L_rank + 0.13182740780834337 * sum(BCE_i / |S|)`.
Для пустого S: `L_rank`. BCE включает зафиксированный target-specific effective
pos_weight из best_params.yaml: click 1.0573239205700589, long 1.286659439842131,
like 9.092198541195886, profile 7.598613882204391. Tuned task weights не используются.
Среднее сохраняет сумму scalar weights = 1; оно не гарантирует одинаковых реальных
норм градиента для разных subsets. Matched effects описывают добавление target
вместе с предусмотренной перенормировкой остальных active weights.

Seed=2026, max_epochs=80, patience=5, min_delta=0, validation каждый epoch,
TRAIN batch=2048, VALID batch=4096. Adam lr=0.0019075370668084298,
head lr=0.0008746259318071252, weight decay=1.925502656735127e-6,
dropout=0.08565706719893162. Все значения проверяются против source of truth.
Нет per-cell tuning, Optuna studies, multiseed или post-hoc budget changes.

## Повторное использование Stage3

Импортируются неизменённые data/config/loader helpers, train_one_epoch,
optimizer_for_model, gradient_diagnostic и aggregate_gradient_records.
Loss weights передаются через уже существующий аргумент; старые loss modes не меняются.
В model всегда существуют четыре heads; optimizer включает только active heads.
Инициализация heads и два stochastic forward (rank, затем auxiliaries) сохранены
точно как в Stage3, включая различающиеся dropout masks. Это обеспечивает
совместимость primary и singles. Gradient diagnostic восстанавливает RNG; его
cosines exploratory и не идентичны каждому реальному stochastic training gradient.

Новый runner добавляет запись best checkpoint по validation NDCG@10, provenance,
atomic reservation, finite-gradient checks, smoke gating, timestamps и memory.
Старый Stage3 выбирал best metrics, но не сохранял соответствующий checkpoint
в run_ablation; исторические JSON и выводы не переинтерпретируются.

## Данные и TEST

Fingerprint `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.
23 951 users, 7 111 items, TRAIN 1 086 518 rows / 1 062 567 sequential examples,
VALID 23 951 examples. Хронологический Protocol B, MAX_ITEM_LIST_LENGTH=50.
TEST count известен лишь из metadata; TEST dataset/loader/metrics не используются.
Runtime guards: Python file audit, запрет shared writes, train/valid-only factory,
проверки config/summary/hashes/source IDs. Native Polars reads в CPU audit ограничены
явно заданными TRAIN/VALID paths; Python audit hook сам по себе не покрывает любой
нативный файловый syscall. Данные не пересоздаются и не копируются.

Перед submit data_audit полностью сравнивает готовые .inter rows с TRAIN/VALID
parquet: IDs, labels и точный train prefix истории; source_row текущего target
исключён. Повторный item_id в истории допустим: запрещается текущая interaction,
а не все более ранние взаимодействия с тем же item. Validation history состоит
только из TRAIN. Existing target_audit запускается в sibling artifacts один раз.
Model input fields только item_id_list, item_length, timestamp_list.

Full-sort evaluator сохранён; padding item исключается, seen items не маскируются.
Runner hard-fails, если sequential loader неожиданно возвращает history_index.
HR/Recall consistency проверяется на @5/10/20/50; MRR не добавляется.

## Pipeline

`slurm/submit_target_combinations.sh` требует TC_REPO и проверенный TC_PYTHON.
Перед submission обязательны source-bound local verification, data audit и environment
certificate. Все scientific runs используют один published exact SHA.

1. All-four smoke: 2 train batches, 1 epoch, gradient diagnostics, полная VALID,
   optimizer/head updates, checkpoint и JSON. Не входит в scientific table.
2. Array 0–15%4, afterok smoke, kill-on-invalid-dep=yes; внутри каждого task ещё
   проверяются smoke scientific gates и exact SHA.
3. CPU summary, afterany array: полный список 16 expected IDs, invalid/failed/missing
   не включаются в ranking. При SIGKILL результат может отсутствовать; sacct сохраняется
   рядом со списком missing, без превращения отсутствующих metrics в нули.

GPU specification одинаков для smoke и всех cells: gpu-ef-quick, type_e,
A100×1, CPU×4, mem=0, 3h, no-requeue. Исторические jobs: 8m55s–20m15s;
3h даёт существенный запас и сохраняет Stage3 resource class. Concurrency 4
ограничивает нагрузку; очередь не является основанием для изменения hyperparameters.
CPU summary: cpu-e-quick, 1 CPU, mem=0, 15m. Slurm объявляет RealMemory=1 MB
для этих узлов, поэтому запрос 2 GiB отклоняется; mem=0 соответствует partition default. Кэши отдельные для каждой job.
Другие MOO/MoE jobs и их checkouts не изменяются.

## Summary и интерпретация

Primary ranking — validation NDCG@10. CSV ranked + отдельный canonical CSV.
JSON содержит все 8 matched deltas каждого target и все 4 background interactions
каждой пары. Partial factorial вычисляется лишь для доступных matched backgrounds,
с явным count/expected. Best by n_aux, relative deltas vs primary, top-gap,
mean по cardinality и exploratory conflict/delta Pearson также сохраняются.

One seed: никаких p-values/significance/causal claims. Gap <=0.0001 помечается
near-tie (одна единица исторического округления), а не доказанным winner.
Historical primary/singles сверяются отдельно; |delta NDCG10|>0.001 фиксированно
помечается requires_investigation, interpretation_blocked_by_regression=true.
Этот порог — инженерный сигнал для расследования, не статистический тест.
All-four Stage3 не equivalent: использовал tuned weights.

Outputs: runs/*.json; artifacts/<run_id>/best_validation.pth и progress.json;
logs/slurm/target-combo-*; summary.json; target_combinations.csv;
target_combinations_canonical.csv; reports/TARGET_COMBINATION_ANALYSIS.md.
Canonical README/reports/results.csv не изменяются. Merge не выполняется.
