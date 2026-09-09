> Retry 002 отправлен: smoke **4315279**, array **4315280** (0–15%4,
> afterok:4315279), summary **4315281** (afterany:4315280).
> Все jobs в rocky, PreemptMode=OFF; GPU спецификация A100/type_e, CPU×4,
> 3h и все scientific parameters сохранены. Code SHA:
> `599bcdb6e50dceea73233169b8834eaceef083a8`, published до submit.
> Новый checkout `/home/daryumin/iberdov/diplom_exp_target_combinations_002`.
> 23/23 tests, полный повторный TRAIN/VALID audit и environment checks passed.
> Старый checkout и attempt 001 не изменены. Evidence: deployment/retry_002/.
> Ниже сохранён исторический audit первой попытки.

# Audit и submission — 9 сентября 2026

Scientific target-combination screening has been submitted.

## Repo и SHA

- Base/current origin/main при старте: `16ba99b69929a711137f616df28e18e7f92b76c1`.
- Local worktree: `/Users/berdov/diplom-target-combinations`.
- Branch: `exp/target-combination-screening`.
- Initial implementation: `b8fbbce4fc38428ee2ada3148f0926830f0ae3dd`.
- **Immutable scientific/job SHA: `a69759ba8dec523be37bcc654b3fe3eef215ee5f`.**
- Код committed/pushed до submit. Cluster checkout clean перед отправкой.
- После submit отдельный commit сохраняет evidence и исправляет только CPU memory
  specification submit script для будущих запусков. Он не меняет научный код
  отправленных jobs. Кластерный checkout НЕ обновлён: все scientific cells и summary
  остаются на одном a69759b. Для SHA итогового evidence commit: `git log -1 --format=%H`.
- Original `~/diplom` остался на ветке challengers, clean. Main не merged.
- Полный implementation file list: deployment/implementation_files.txt.
  Добавлены только sibling experiment, два Slurm scripts, новый report skeleton и ignores.

## Pipeline (снимок 18:18 МСК)

| Этап | Job ID | State | Dependency | Resource |
|---|---|---|---|---|
| All-four smoke, 2 train batches + full validation | 4314482 | PENDING / Priority | нет | gpu-ef-quick, A100×1, CPU×4, mem=0, 3h |
| Все 16 scientific cells | 4314483_[0-15%4] | PENDING / Dependency | afterok:4314482 | то же, concurrency 4 |
| CPU summary | 4314485 | PENDING / Dependency | afterany:4314483_* | cpu-e-quick, CPU×1, mem=0, 15m |

GPU constraint запрошен type_e; submit policy partition преобразовал Features
в `type_e|type_f` (это видно в scontrol). Запрос `gres/gpu:a100:1` сохранён, спецификация
единая для всех cells. No requeue. Array при неуспешном smoke отменяется через
kill-on-invalid-dep=yes; runtime дополнительно проверяет completed smoke scientific gate.

Первая CPU submission с mem=2G отклонена ДО создания job: узлы объявляют RealMemory=1 MB.
Отправлен только отсутствующий summary с mem=0; smoke/array не дублировались.
Обе попытки сохранены в deployment/pipeline.json. В future submit script исправлен mem=0.

Cluster: `/home/daryumin/iberdov/diplom_exp_target_combinations`.
Новый checkout создан отдельно, stripe OST7 только для новой директории.
Другие jobs (4313103, 4313930, 4313931, 4313933), их код и outputs не изменены.
Старые EPO+MoE jobs также не трогались.

## Scientific freeze

Formula: L_rank + lambda_aux * mean(active BCE), empty subset = L_rank.
Lambda=0.13182740780834337. Активные веса 1/n, сумма=1; tuned_task_weights не применяются.
Это устраняет рост суммы scalar weights с числом задач, но не обещает равных gradient norms.
Effective pos weights, lr, head lr, weight decay, dropout совпали с актуальным best_params.yaml.
Seed 2026; max_epochs 80; patience 5; min_delta 0; batches 2048/4096; validation каждый epoch.
Best checkpoint только по validation NDCG@10. Без Optuna и per-combination tuning.

Все 16 включают primary next_item:

| Array index | ABCD mask | Auxiliary subset | Scientific run ID |
|---|---|---|---|
| 0 | 0000 | primary_only | `target_combo_0000_primary_only_001` |
| 1 | 1000 | click | `target_combo_1000_click_001` |
| 2 | 0100 | long_view | `target_combo_0100_long_view_001` |
| 3 | 0010 | like | `target_combo_0010_like_001` |
| 4 | 0001 | profile_enter | `target_combo_0001_profile_enter_001` |
| 5 | 1100 | click + long_view | `target_combo_1100_click_long_view_001` |
| 6 | 1010 | click + like | `target_combo_1010_click_like_001` |
| 7 | 1001 | click + profile_enter | `target_combo_1001_click_profile_enter_001` |
| 8 | 0110 | long_view + like | `target_combo_0110_long_view_like_001` |
| 9 | 0101 | long_view + profile_enter | `target_combo_0101_long_view_profile_enter_001` |
| 10 | 0011 | like + profile_enter | `target_combo_0011_like_profile_enter_001` |
| 11 | 1110 | click + long_view + like | `target_combo_1110_click_long_view_like_001` |
| 12 | 1101 | click + long_view + profile_enter | `target_combo_1101_click_long_view_profile_enter_001` |
| 13 | 1011 | click + like + profile_enter | `target_combo_1011_click_like_profile_enter_001` |
| 14 | 0111 | long_view + like + profile_enter | `target_combo_0111_long_view_like_profile_enter_001` |
| 15 | 1111 | click + long_view + like + profile_enter | `target_combo_1111_click_long_view_like_profile_enter_001` |

## Preflight

23/23 local tests passed на scientific commit и после CPU submission correction.
Powerset/unique IDs, 16 нормированных losses/active optimizers, diagnostic RNG,
known factorial fixture, partial/mixed-SHA summary, overwrite и TEST/file guards проверены.
82 historical file hashes unchanged. Source-bound scientific verification сохранена
отдельно в deployment/scientific_verification.json; текущая verification.json относится
к последней версии submit script.

Fingerprint: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.
Полный data audit проверил 1 062 567 TRAIN sequential examples и 23 951 VALID examples,
истории 23 951 users, исходные labels/IDs, исключение текущего source_row из history,
проверку всех .inter и source parquet hashes. Данные не готовились заново и не копировались.
Existing Stage3 target audit запущен; four-target train counts/missing rates проверены.
Доказательства: deployment/data_audit.json и deployment/target_audit.json.

TEST dataset loaded: NO. TEST dataloader: NO. TEST evaluations: 0.
File guard + train/valid-only factory + config/summary checks остаются активны в jobs.
Full-sort universe 7111, повторные items разрешены; seen masking не вводится.

Environment: Python 3.10.14, torch 2.3.0+cu118, CUDA 11.8, RecBole 1.2.0,
NumPy 1.26.4, mamba_ssm 2.2.2, causal_conv1d 1.2.2.post1, shared Git module 2.50.1.
Используется уже работающее окружение challengers read-only; cache каждого job —
в новой директории. Старое env задерживалось в Lustre cl_sync_io_wait; новых установок
и изменений shared env не выполнялось. Официальный data-prep Python использовался
только для CPU audit. GPU name будет записано после фактического выделения GPU.

## Обнаруженные расхождения

1. Во время нового preflight найден вызов historical load_json со строкой вместо Path.
   Исправлен в новом sibling runner до submit (a69759b); после этого проверки прошли.
   Historical Stage3 это не затрагивает.
2. Existing Stage3 derive_binary_columns расходится с multitask manifest в трёх
   НЕ используемых здесь composite signals. Для строки с profile_enter=1 и остальными
   нулями Stage3 выдаёт strong/explicit/deep=(1,0,0), manifest требует (0,1,1).
   Minimal reproduction: deployment/historical_composite_discrepancy.json.
   Четыре raw targets и их Stage3 ablations не затронуты; composite-specific исторические
   интерпретации требуют отдельного review. Историю не исправляли и не удаляли.
3. CPU memory policy и единственная повторная CPU submission описаны выше.

## Outputs и ограничения

Все пути ниже относительны к isolated cluster checkout:

- runs: `experiments/target_combination_analysis/runs/`;
- checkpoints/progress: `experiments/target_combination_analysis/artifacts/<run_id>/`;
- logs: `logs/slurm/target-combo-*`;
- future ranked CSV: `experiments/target_combination_analysis/target_combinations.csv`;
- future canonical-order CSV: `experiments/target_combination_analysis/target_combinations_canonical.csv`;
- future JSON: `experiments/target_combination_analysis/summary.json`;
- future report: `reports/TARGET_COMBINATION_ANALYSIS.md`.

Summary заранее реализует completeness, valid-only ranking, best by n_aux, matched
marginals (8 per target), pair interactions (4 per pair), gradients, near ties и
historical primary/single regression flags. Missing/failed cells не заменяются нулями
или старыми runs. Один seed: только descriptive screening, никаких p-values,
statistical significance или causal gradient claims. При существенном расхождении
с historical anchors требуется расследование перед научной интерпретацией.

На момент снимка scientific metrics отсутствуют; smoke ещё не завершился.
Задание по реализации и постановке pipeline выполнено; качество методов ещё не оценено.
Canonical README, reports/RESULTS.md, experiments/results.csv, PAPER_RESULTS.md не изменены.
PR и merge не выполнялись.
