# cosmos_sanity_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `sanity`.
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
| 0.0487 | 0.0743 | 0.1160 | 0.2056 | 0.0341 | 0.0423 | 0.0527 | 0.0704 |

## Oracle Best Validation Point

- id: `rank_heavy`.
- NDCG@10: `0.0423`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.0742 | 0.0422 | 0.6424 | 0.6069 | 0.1714 | 0.1872 |
| `rank_heavy` | 0.0743 | 0.0423 | 0.6424 | 0.6069 | 0.1714 | 0.1872 |
| `click_heavy` | 0.0743 | 0.0423 | 0.6424 | 0.6069 | 0.1714 | 0.1872 |
| `long_heavy` | 0.0743 | 0.0423 | 0.6424 | 0.6069 | 0.1714 | 0.1872 |
| `like_heavy` | 0.0743 | 0.0423 | 0.6424 | 0.6070 | 0.1715 | 0.1872 |
| `profile_heavy` | 0.0742 | 0.0422 | 0.6424 | 0.6069 | 0.1714 | 0.1872 |

## Cost

- Runtime sec: `333.246`.
- Peak VRAM bytes: `2422317056`.
- Params: `602398` trainable.

