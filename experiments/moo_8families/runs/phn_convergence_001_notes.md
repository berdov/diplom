# phn_convergence_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `convergence_screening`.
- Method: `PHN-adapter`.
- Implementation: `PHN-adapter`.
- Representative fidelity: `family-level adaptation`.
- Exact method reproduction: `False`.

## Dataset

- Protocol: `B`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Ranking Operating Point

- Selection: `predefined_preference_id:rank_heavy`.
- Selection is validation oracle: `False`.
| HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0483 | 0.0746 | 0.1155 | 0.2027 | 0.0339 | 0.0423 | 0.0526 | 0.0698 |

## Oracle Best Validation Point

- id: `balanced`.
- NDCG@10: `0.0424`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.0752 | 0.0424 | 0.6646 | 0.6310 | 0.1302 | 0.1724 |
| `rank_heavy` | 0.0746 | 0.0423 | 0.6636 | 0.6296 | 0.1279 | 0.1705 |
| `click_heavy` | 0.0750 | 0.0424 | 0.6645 | 0.6309 | 0.1299 | 0.1721 |
| `long_heavy` | 0.0750 | 0.0424 | 0.6646 | 0.6309 | 0.1301 | 0.1723 |
| `like_heavy` | 0.0752 | 0.0424 | 0.6647 | 0.6312 | 0.1304 | 0.1724 |
| `profile_heavy` | 0.0752 | 0.0424 | 0.6647 | 0.6311 | 0.1303 | 0.1725 |

## Early Stopping

- Requested max epochs: `100`.
- Validation interval: `5`.
- Best epoch: `60`.
- Stop epoch: `75`.
- Validation checks: `15`.
- Early stopped: `True`.
- Stop reason: `early_stopping_patience`.

## Cost

- Runtime sec: `2403.091`.
- Peak VRAM bytes: `2421269504`.
- Params: `602462` trainable.
