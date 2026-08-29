# epo_convergence_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `convergence_screening`.
- Method: `EPO`.
- Implementation: `EPO`.
- Representative fidelity: `exact_or_close_reproduction`.
- Exact method reproduction: `True`.

## Dataset

- Protocol: `B`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Ranking Operating Point

- Selection: `predefined_preference_id:rank_heavy`.
- Selection is validation oracle: `False`.
| HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0644 | 0.1078 | 0.1767 | 0.3171 | 0.0445 | 0.0584 | 0.0756 | 0.1033 |

## Oracle Best Validation Point

- id: `rank_heavy`.
- NDCG@10: `0.0584`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.0596 | 0.0350 | 0.7369 | 0.6864 | 0.1693 | 0.2129 |
| `rank_heavy` | 0.1078 | 0.0584 | 0.6656 | 0.6700 | 0.4863 | 0.5552 |
| `click_heavy` | 0.0404 | 0.0232 | 0.7970 | 0.6424 | 0.1770 | 0.2120 |
| `long_heavy` | 0.0409 | 0.0229 | 0.6610 | 0.7538 | 0.1988 | 0.1992 |
| `like_heavy` | 0.0624 | 0.0317 | 0.6636 | 0.6255 | 0.1336 | 0.1877 |
| `profile_heavy` | 0.0386 | 0.0209 | 0.6727 | 0.6336 | 0.1591 | 0.1915 |

## Early Stopping

- Requested max epochs: `100`.
- Validation interval: `5`.
- Best epoch: `15`.
- Stop epoch: `30`.
- Validation checks: `6`.
- Early stopped: `True`.
- Stop reason: `early_stopping_patience`.

## Cost

- Runtime sec: `18209.927`.
- Peak VRAM bytes: `2432410624`.
- Params: `3562548` trainable.
