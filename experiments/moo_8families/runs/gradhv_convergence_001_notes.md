# gradhv_convergence_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `convergence_screening`.
- Method: `HV-Gradient / GradHV-style`.
- Implementation: `HV-Gradient / GradHV-style`.
- Representative fidelity: `family-level adaptation`.
- Exact method reproduction: `False`.

## Dataset

- Protocol: `B`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Ranking Operating Point

- Selection: `best_validation_NDCG@10_among_preference_free_finite_solutions`.
- Selection is validation oracle: `True`.
| HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0544 | 0.0874 | 0.1382 | 0.2440 | 0.0380 | 0.0486 | 0.0613 | 0.0820 |

## Oracle Best Validation Point

- id: `2`.
- NDCG@10: `0.0486`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `None` | 0.0068 | 0.0038 | 0.6931 | 0.6931 | 0.6931 | 0.6931 |
| `None` | 0.0612 | 0.0353 | 0.8475 | 0.8167 | 0.1243 | 0.2092 |
| `None` | 0.0874 | 0.0486 | 0.6574 | 0.6223 | 0.1441 | 0.2190 |

## Early Stopping

- Requested max epochs: `100`.
- Validation interval: `5`.
- Best epoch: `50`.
- Stop epoch: `65`.
- Validation checks: `13`.
- Early stopped: `True`.
- Stop reason: `early_stopping_patience`.

## Cost

- Runtime sec: `5264.388`.
- Peak VRAM bytes: `5246018560`.
- Params: `1781274` trainable.
