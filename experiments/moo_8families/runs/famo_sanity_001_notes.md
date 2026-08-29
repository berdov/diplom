# famo_sanity_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `sanity`.
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
| 0.0423 | 0.0660 | 0.1033 | 0.1785 | 0.0301 | 0.0377 | 0.0471 | 0.0619 |

## Oracle Best Validation Point

- id: `0`.
- NDCG@10: `0.0377`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `0` | 0.0660 | 0.0377 | 0.6391 | 0.6040 | 0.1219 | 0.1647 |

## Cost

- Runtime sec: `362.100`.
- Peak VRAM bytes: `2420673536`.
- Params: `593758` trainable.

