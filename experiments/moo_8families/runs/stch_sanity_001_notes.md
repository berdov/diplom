# stch_sanity_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `sanity`.
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
| 0.0378 | 0.0580 | 0.0930 | 0.1632 | 0.0256 | 0.0321 | 0.0409 | 0.0547 |

## Oracle Best Validation Point

- id: `0`.
- NDCG@10: `0.0321`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `0` | 0.0580 | 0.0321 | 0.6415 | 0.6049 | 0.1285 | 0.1635 |

## Cost

- Runtime sec: `230.170`.
- Peak VRAM bytes: `2472282112`.
- Params: `593758` trainable.

