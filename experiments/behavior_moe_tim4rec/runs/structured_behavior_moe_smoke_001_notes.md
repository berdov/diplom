# Structured Behavior-MoE smoke 001

## Цель

Проверить техническую работоспособность `StructuredBehaviorMoE` поверх tuned MultitaskTiM4Rec на реальных train batches KuaiRand Protocol B.

## Архитектура

- Routing mode: `structured`.
- Experts: `interest, consumption, engagement, shared`.
- Expert MLP: `Linear(hidden, hidden) -> GELU -> Dropout -> Linear(hidden, hidden)`.
- Router: `separate learned Linear(hidden, num_experts) router head per task; structured mode masks forbidden logits before softmax`.
- Residual: `h_task = h + residual_scale * sum_e p(task,e|h) * expert_e(h)`.
- Load balance на smoke: `False`.

## Allowed expert masks

| task | allowed experts | forbidden experts | exact zeros |
|---|---|---|---:|
| ranking | interest, consumption, engagement, shared | none | True |
| click | interest, shared | consumption, engagement | True |
| long_view | consumption, shared | interest, engagement | True |
| like | engagement, shared | interest, consumption | True |
| profile | engagement, shared | interest, consumption | True |

## Smoke

- Batches: `5`.
- Batch size: `2048`.
- Slurm job: `4278026`, `COMPLETED`, `ExitCode=0:0`, node `cn-044`, `test/type_e`.
- Slurm elapsed: `00:05:32`.
- Slurm batch MaxRSS: `3014080K`.
- Losses finite: `True`.
- Gradients finite: `True`.
- Test evaluations: `0`.

## Routing после smoke

| task | interest | consumption | engagement | shared |
|---|---:|---:|---:|---:|
| ranking | 0.2282 | 0.2470 | 0.2784 | 0.2465 |
| click | 0.5070 | 0.0000 | 0.0000 | 0.4930 |
| long_view | 0.0000 | 0.5362 | 0.0000 | 0.4638 |
| like | 0.0000 | 0.0000 | 0.4639 | 0.5361 |
| profile | 0.0000 | 0.0000 | 0.4873 | 0.5127 |

## Specialization score

| metric | value |
|---|---:|
| click_interest_share | 0.5070 |
| long_consumption_share | 0.5362 |
| like_engagement_share | 0.4639 |
| profile_engagement_share | 0.4873 |
| average_specialist_share | 0.4986 |
| uniform_specialist_share_baseline | 0.5000 |
| specialization_above_uniform | -0.0014 |
- Shared domination: `False`.

## Collapse checks

- Expert collapse: `False`.
- Shared domination: `False`.
- All auxiliary shared domination: `False`.
- Forbidden paths exact zero: `True`.
- All-task same routing: `False`.
- Minimum experts used per task: `2`.

## Specialization signal

- Strongest pair: `click` vs `long_view`.
- L1 distance: `1.072466`.
- Cosine distance: `0.54394922`.

## Gradients and updates

- Experts updated: `True`.
- Router updated: `True`.
- Auxiliary heads updated: `True`.
- Expected structured connectivity met: `True`.

## Cost

- Tuned fixed mean step: `0.405553` sec.
- Tuned fixed trimmed mean step: `0.048874` sec.
- Behavior-MoE mean step: `0.128540` sec.
- Behavior-MoE trimmed mean step: `0.070668` sec.
- Step-time overhead: `0.3169`.
- Peak VRAM: `1847309312` bytes.
- Generic reference mean step: `0.038666875846683976` sec.
- Generic reference trimmed mean step: `0.03586672650029262` sec.
- Generic reference GPU: `NVIDIA H200 NVL`.

## Вывод

Structured smoke технически корректен, но выявил риск до 5-epoch sanity: средний specialist share `0.4986` чуть ниже uniform baseline `0.5`. Запрещенные routing paths занулены точно, gradients идут по ожидаемым task-to-expert связям, shared domination нет.
