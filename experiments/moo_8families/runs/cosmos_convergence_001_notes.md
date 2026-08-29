# cosmos_convergence_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `convergence_screening`.
- Method: `COSMOS-style`.
- Implementation: `COSMOS-style direct preference conditioning`.
- Representative fidelity: `method-level adaptation`.
- Exact method reproduction: `False`.

## Dataset

- Protocol: `B`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Ranking Operating Point

- Selection: `predefined_preference_id:rank_heavy`.
- Selection is validation oracle: `False`.
| HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0519 | 0.0810 | 0.1257 | 0.2252 | 0.0360 | 0.0453 | 0.0565 | 0.0761 |

## Oracle Best Validation Point

- id: `click_heavy`.
- NDCG@10: `0.0454`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.0809 | 0.0451 | 0.7236 | 0.6849 | 0.1921 | 0.1747 |
| `rank_heavy` | 0.0810 | 0.0453 | 0.7256 | 0.6845 | 0.1465 | 0.1712 |
| `click_heavy` | 0.0812 | 0.0454 | 0.7254 | 0.6844 | 0.1471 | 0.1712 |
| `long_heavy` | 0.0812 | 0.0454 | 0.7255 | 0.6844 | 0.1468 | 0.1711 |
| `like_heavy` | 0.0797 | 0.0446 | 0.7268 | 0.6860 | 0.6079 | 0.1733 |
| `profile_heavy` | 0.0810 | 0.0453 | 0.7248 | 0.6842 | 0.1500 | 0.1721 |

## Early Stopping

- Requested max epochs: `100`.
- Validation interval: `5`.
- Best epoch: `25`.
- Stop epoch: `40`.
- Validation checks: `8`.
- Early stopped: `True`.
- Stop reason: `early_stopping_patience`.

## Cost

- Runtime sec: `1449.899`.
- Peak VRAM bytes: `2422317056`.
- Params: `602398` trainable.
