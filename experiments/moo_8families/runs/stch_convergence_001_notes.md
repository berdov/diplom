# stch_convergence_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `convergence_screening`.
- Method: `STCH`.
- Implementation: `STCH`.
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
| 0.0476 | 0.0749 | 0.1163 | 0.2082 | 0.0336 | 0.0424 | 0.0528 | 0.0709 |

## Oracle Best Validation Point

- id: `0`.
- NDCG@10: `0.0424`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `0` | 0.0749 | 0.0424 | 0.7018 | 0.6760 | 0.1421 | 0.2044 |

## Early Stopping

- Requested max epochs: `100`.
- Validation interval: `5`.
- Best epoch: `80`.
- Stop epoch: `95`.
- Validation checks: `19`.
- Early stopped: `True`.
- Stop reason: `early_stopping_patience`.

## Cost

- Runtime sec: `2872.418`.
- Peak VRAM bytes: `2420672512`.
- Params: `593758` trainable.
