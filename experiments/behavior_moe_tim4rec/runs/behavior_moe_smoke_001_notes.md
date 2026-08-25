# Behavior-MoE smoke 001

## Цель

Проверить техническую работоспособность compact Behavior-MoE поверх tuned MultitaskTiM4Rec на реальных train batches KuaiRand Protocol B.

## Архитектура

- Experts: `interest, consumption, positive, shared`.
- Expert MLP: `Linear(hidden, hidden) -> GELU -> Dropout -> Linear(hidden, hidden)`.
- Router: `separate learned Linear(hidden, 4) router head per task; softmax(logits / temperature)`.
- Residual: `h_task = h + residual_scale * sum_e p(task,e|h) * expert_e(h)`.
- Load balance на smoke: `False`.

## Smoke

- Batches: `5`.
- Batch size: `2048`.
- Losses finite: `True`.
- Gradients finite: `True`.
- Test evaluations: `0`.

## Routing после smoke

| task | interest | consumption | positive | shared |
|---|---:|---:|---:|---:|
| ranking | 0.2270 | 0.2568 | 0.2783 | 0.2379 |
| click | 0.2682 | 0.2394 | 0.2454 | 0.2470 |
| long_view | 0.2702 | 0.2593 | 0.2337 | 0.2369 |
| like | 0.2272 | 0.2678 | 0.2521 | 0.2530 |
| profile | 0.2296 | 0.2505 | 0.2582 | 0.2617 |

## Collapse checks

- Expert collapse: `False`.
- Shared domination: `False`.
- All-task same routing: `False`.
- Minimum experts used per task: `4`.

## Specialization signal

- Strongest pair: `ranking` vs `click`.
- L1 distance: `0.100712`.
- Cosine distance: `0.00631456`.

## Gradients and updates

- Experts updated: `True`.
- Router updated: `True`.
- Auxiliary heads updated: `True`.

## Cost

- Tuned fixed mean step: `0.124420` sec.
- Behavior-MoE mean step: `0.038667` sec.
- Step-time overhead: `0.3108`.
- Trimmed step-time overhead без первого measured fixed step: `1.3783`.
- Peak VRAM: `1898208256` bytes.

Raw ratio не интерпретируется как ускорение MoE: первый measured fixed step содержит остаточный CUDA/cache overhead. Для smoke-level overhead разумнее смотреть trimmed ratio.

## Вывод

Smoke pipeline корректен: Behavior-MoE делает forward/backward/optimizer step на real train batches, router и experts получают gradients, routing не collapsed. Следующий sanity лучше запускать как plain Behavior-MoE без load balancing.
