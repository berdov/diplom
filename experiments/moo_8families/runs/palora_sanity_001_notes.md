# palora_sanity_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `sanity`.
- Method: `PaLoRA`.
- Implementation: `PaLoRA low-rank adapter combination`.
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
| 0.0400 | 0.0618 | 0.0970 | 0.1731 | 0.0282 | 0.0352 | 0.0440 | 0.0590 |

## Oracle Best Validation Point

- id: `balanced`.
- NDCG@10: `0.0354`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.0623 | 0.0354 | 0.6397 | 0.6042 | 0.1202 | 0.1608 |
| `rank_heavy` | 0.0618 | 0.0352 | 0.6387 | 0.6040 | 0.1198 | 0.1613 |
| `click_heavy` | 0.0623 | 0.0353 | 0.6397 | 0.6043 | 0.1203 | 0.1606 |
| `long_heavy` | 0.0623 | 0.0354 | 0.6397 | 0.6042 | 0.1202 | 0.1610 |
| `like_heavy` | 0.0624 | 0.0354 | 0.6397 | 0.6042 | 0.1202 | 0.1609 |
| `profile_heavy` | 0.0625 | 0.0354 | 0.6397 | 0.6042 | 0.1201 | 0.1609 |

## Cost

- Runtime sec: `363.585`.
- Peak VRAM bytes: `2426233344`.
- Params: `596318` trainable.

