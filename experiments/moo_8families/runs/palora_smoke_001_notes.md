# palora_smoke_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `smoke`.
- Method: `PaLoRA`.
- Implementation: `PaLoRA low-rank adapter combination`.
- Representative fidelity: `method-level adaptation`.
- Exact method reproduction: `False`.

## Dataset

- Protocol: `B`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Smoke

- Train batches: `3`.
- Mean scalar loss: `0.9910372893015543`.
- Validation не запускалась.

## Preference Sensitivity

- Split: `train`.
- Preferences: `rank_heavy` vs `like_heavy`.
- Representation L2: `0.45807185769081116`.
- Ranking score L2: `0.011004646308720112`.
- Ranking score mean abs: `0.00017211033264175057`.
- Passed: `True`.

## Cost

- Runtime sec: `171.493`.
- Peak VRAM bytes: `2426233344`.
- Params: `596318` trainable.
