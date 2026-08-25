# Behavior-MoE TiM4Rec

Эта ветка проверяет минимальную behavior-specialized MoE-надстройку поверх tuned fixed MultitaskTiM4Rec для KuaiRand Protocol B.

Границы этапа:

- backbone остаётся тем же TiM4Rec;
- split, targets, ranking objective и time-aware path не меняются;
- базовая loss configuration берётся из `multitask_tim4rec_optuna_v1`, trial `110`;
- MoE добавляется только после shared representation `h`;
- router не получает current behavior labels;
- smoke использует только real train batches;
- `behavior_moe_sanity_001` использует полный train и validation-only full-ranking evaluation;
- full training, Optuna, load balancing и test не запускаются.

Структура:

- `config.yaml` - зафиксированная конфигурация smoke;
- `model.py` - `BehaviorMoETiM4Rec`;
- `smoke_test.py` - smoke на real train batches и routing diagnostics;
- `sanity_train.py` - ровно 5 эпох plain Behavior-MoE без load balancing и test;
- `AUDIT.md` - архитектурный аудит и результаты smoke;
- `runs/behavior_moe_smoke_001.json` - compact JSON-артефакт;
- `runs/behavior_moe_smoke_001_notes.md` - краткий отчёт smoke.
- `runs/behavior_moe_sanity_001.json` - 5-epoch validation sanity;
- `runs/behavior_moe_sanity_001_routing.csv` - routing trajectory по эпохам.

Архитектура:

- 4 experts: `interest`, `consumption`, `positive`, `shared`;
- каждый expert: `Linear(hidden, hidden) -> GELU -> Dropout -> Linear(hidden, hidden)`;
- task-conditioned routing: отдельный learned `Linear(hidden, 4)` router для `ranking`, `click`, `long_view`, `like`, `profile`;
- soft routing: `softmax(logits / temperature)`;
- residual: `h_task = h + residual_scale * sum_e p(task,e|h) * expert_e(h)`;
- load-balancing loss подготовлен, но в первом smoke выключен.

Запуск на кластере:

```bash
sbatch slurm/behavior_moe_tim4rec.sh
```

По умолчанию smoke делает `5` optimization steps на train batches и сохраняет только диагностические артефакты. Locked test не загружается и не оценивается.

Финальный `behavior_moe_smoke_001` выполнен на Slurm job `4276396`, `test/type_h`, node `cn-050`, GPU `NVIDIA H200 NVL`. Smoke подтвердил finite forward/backward, gradients и updates для experts/router/heads, отсутствие router collapse и слабый task-specific routing signal.

`behavior_moe_sanity_001` выполнен на Slurm job `4276720`, `test/type_e`, node `cn-045`, GPU `NVIDIA A100-SXM4-80GB`. Запуск прошёл 5 эпох без test access. Best validation `NDCG@10=0.0562` на epoch 4, epoch 5 `NDCG@10=0.0546`. Router collapse, dead expert и shared domination не обнаружены, но task specialization по mean required-pair L1 снизилась от `0.9368` до `0.6816`; full run лучше не запускать до анализа routing architecture.
