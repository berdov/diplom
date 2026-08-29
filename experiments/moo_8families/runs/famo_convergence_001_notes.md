# famo_convergence_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `convergence_screening`.
- Method: `FAMO`.
- Implementation: `FAMO`.
- Representative fidelity: `exact_or_close_reproduction`.
- Exact method reproduction: `True`.

## Dataset

- Protocol: `B`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Ranking Operating Point

- Selection: `single_solution`.
- Selection is validation oracle: `False`.
| HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0472 | 0.0719 | 0.1102 | 0.1935 | 0.0332 | 0.0412 | 0.0508 | 0.0672 |

## Oracle Best Validation Point

- id: `0`.
- NDCG@10: `0.0412`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `0` | 0.0719 | 0.0412 | 0.6444 | 0.6148 | 0.1291 | 0.1750 |

## Early Stopping

- Requested max epochs: `100`.
- Validation interval: `5`.
- Best epoch: `15`.
- Stop epoch: `30`.
- Validation checks: `6`.
- Early stopped: `True`.
- Stop reason: `early_stopping_patience`.

## Cost

- Runtime sec: `1233.014`.
- Peak VRAM bytes: `2420673536`.
- Params: `593758` trainable.
