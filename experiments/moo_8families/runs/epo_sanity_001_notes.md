# epo_sanity_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `sanity`.
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
| 0.0646 | 0.1050 | 0.1684 | 0.2982 | 0.0449 | 0.0577 | 0.0737 | 0.0992 |

## Oracle Best Validation Point

- id: `rank_heavy`.
- NDCG@10: `0.0577`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.0635 | 0.0359 | 0.6362 | 0.6041 | 0.1421 | 0.2037 |
| `rank_heavy` | 0.1050 | 0.0577 | 0.7022 | 0.6580 | 0.5231 | 0.4909 |
| `click_heavy` | 0.0371 | 0.0211 | 0.6405 | 0.6064 | 0.1647 | 0.1963 |
| `long_heavy` | 0.0422 | 0.0226 | 0.6401 | 0.6082 | 0.1619 | 0.2039 |
| `like_heavy` | 0.0358 | 0.0188 | 0.6827 | 0.6407 | 0.1320 | 0.2067 |
| `profile_heavy` | 0.0359 | 0.0186 | 0.6741 | 0.6326 | 0.6931 | 0.1693 |

## Cost

- Runtime sec: `2404.992`.
- Peak VRAM bytes: `2482134016`.
- Params: `3562548` trainable.

