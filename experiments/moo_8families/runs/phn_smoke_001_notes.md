# phn_smoke_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `smoke`.
- Method: `PHN-adapter`.
- Implementation: `PHN-adapter`.
- Representative fidelity: `family-level adaptation`.
- Exact method reproduction: `False`.

## Dataset

- Protocol: `B`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Smoke

- Train batches: `3`.
- Mean scalar loss: `0.98927108446757`.
- Validation не запускалась.

## Preference Sensitivity

- Split: `train`.
- Preferences: `rank_heavy` vs `like_heavy`.
- Representation L2: `0.1052011102437973`.
- Ranking score L2: `0.0028109245467931032`.
- Ranking score mean abs: `5.0135160563513637e-05`.
- Passed: `True`.

## Cost

- Runtime sec: `165.442`.
- Peak VRAM bytes: `2421269504`.
- Params: `602462` trainable.
