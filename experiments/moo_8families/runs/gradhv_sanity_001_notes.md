# gradhv_sanity_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `sanity`.
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
| 0.0524 | 0.0795 | 0.1259 | 0.2310 | 0.0365 | 0.0452 | 0.0568 | 0.0775 |

## Oracle Best Validation Point

- id: `0`.
- NDCG@10: `0.0452`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `None` | 0.0795 | 0.0452 | 0.6359 | 0.6016 | 0.1330 | 0.1640 |
| `None` | 0.0492 | 0.0281 | 0.6505 | 0.6200 | 0.1816 | 0.1934 |
| `None` | 0.0238 | 0.0134 | 0.6915 | 0.6665 | 0.3572 | 0.4026 |

## Cost

- Runtime sec: `449.652`.
- Peak VRAM bytes: `5297492992`.
- Params: `1781274` trainable.

