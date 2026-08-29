# phn_sanity_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `sanity`.
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
| 0.0412 | 0.0630 | 0.0970 | 0.1706 | 0.0289 | 0.0358 | 0.0444 | 0.0589 |

## Oracle Best Validation Point

- id: `click_heavy`.
- NDCG@10: `0.0359`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.0631 | 0.0358 | 0.6420 | 0.6058 | 0.1416 | 0.1676 |
| `rank_heavy` | 0.0630 | 0.0358 | 0.6421 | 0.6059 | 0.1397 | 0.1666 |
| `click_heavy` | 0.0630 | 0.0359 | 0.6422 | 0.6059 | 0.1422 | 0.1674 |
| `long_heavy` | 0.0630 | 0.0358 | 0.6421 | 0.6058 | 0.1425 | 0.1683 |
| `like_heavy` | 0.0630 | 0.0358 | 0.6420 | 0.6058 | 0.1402 | 0.1669 |
| `profile_heavy` | 0.0632 | 0.0359 | 0.6419 | 0.6057 | 0.1413 | 0.1677 |

## Cost

- Runtime sec: `333.174`.
- Peak VRAM bytes: `2421269504`.
- Params: `602462` trainable.

