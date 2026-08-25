# Behavior-MoE TiM4Rec

Эта ветка проверяет минимальную behavior-specialized MoE-надстройку поверх tuned fixed MultitaskTiM4Rec для KuaiRand Protocol B.

Границы первого этапа:

- backbone остаётся тем же TiM4Rec;
- split, targets, ranking objective и time-aware path не меняются;
- базовая loss configuration берётся из `multitask_tim4rec_optuna_v1`, trial `110`;
- MoE добавляется только после shared representation `h`;
- router не получает current behavior labels;
- smoke использует только real train batches;
- full validation, 5-epoch sanity, full training, Optuna и test не запускаются.

Структура:

- `config.yaml` - зафиксированная конфигурация smoke;
- `model.py` - `BehaviorMoETiM4Rec`;
- `smoke_test.py` - smoke на real train batches и routing diagnostics;
- `AUDIT.md` - архитектурный аудит и результаты smoke;
- `runs/behavior_moe_smoke_001.json` - compact JSON-артефакт;
- `runs/behavior_moe_smoke_001_notes.md` - краткий отчёт smoke.

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

Финальный `behavior_moe_smoke_001` выполнен на Slurm job `4276396`, `test/type_h`, node `cn-050`, GPU `NVIDIA H200 NVL`. Smoke подтвердил finite forward/backward, gradients и updates для experts/router/heads, отсутствие router collapse и слабый task-specific routing signal. Следующий sanity run: plain Behavior-MoE без load balancing.
