# pcgrad_convergence_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `convergence_screening`.
- Method: `PCGrad`.
- Implementation: `PCGrad`.
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
| 0.0496 | 0.0790 | 0.1259 | 0.2253 | 0.0350 | 0.0444 | 0.0562 | 0.0757 |

## Oracle Best Validation Point

- id: `0`.
- NDCG@10: `0.0444`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `0` | 0.0790 | 0.0444 | 0.6685 | 0.6334 | 0.1523 | 0.2074 |

## Early Stopping

- Requested max epochs: `100`.
- Validation interval: `5`.
- Best epoch: `25`.
- Stop epoch: `40`.
- Validation checks: `8`.
- Early stopped: `True`.
- Stop reason: `early_stopping_patience`.

## Cost

- Runtime sec: `4295.712`.
- Peak VRAM bytes: `2420672512`.
- Params: `593758` trainable.
