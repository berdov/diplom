# palora_convergence_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `convergence_screening`.
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
| 0.0483 | 0.0750 | 0.1159 | 0.2080 | 0.0336 | 0.0422 | 0.0525 | 0.0706 |

## Oracle Best Validation Point

- id: `rank_heavy`.
- NDCG@10: `0.0422`.

## Validation Points

| id | HR@10 | NDCG@10 | click BCE | long BCE | like BCE | profile BCE |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.0749 | 0.0421 | 0.6632 | 0.6309 | 0.1437 | 0.1919 |
| `rank_heavy` | 0.0750 | 0.0422 | 0.6621 | 0.6300 | 0.1431 | 0.1913 |
| `click_heavy` | 0.0749 | 0.0422 | 0.6632 | 0.6308 | 0.1437 | 0.1919 |
| `long_heavy` | 0.0748 | 0.0421 | 0.6634 | 0.6310 | 0.1436 | 0.1919 |
| `like_heavy` | 0.0751 | 0.0422 | 0.6632 | 0.6308 | 0.1437 | 0.1919 |
| `profile_heavy` | 0.0749 | 0.0421 | 0.6632 | 0.6308 | 0.1437 | 0.1919 |

## Early Stopping

- Requested max epochs: `100`.
- Validation interval: `5`.
- Best epoch: `35`.
- Stop epoch: `50`.
- Validation checks: `10`.
- Early stopped: `True`.
- Stop reason: `early_stopping_patience`.

## Cost

- Runtime sec: `1807.899`.
- Peak VRAM bytes: `2426233344`.
- Params: `596318` trainable.
