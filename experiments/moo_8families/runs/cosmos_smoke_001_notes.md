# cosmos_smoke_001

## Safety

- `test_evaluation_count = 0`.
- Test dataset не загружался, test dataloader не создавался.
- Stage: `smoke`.
- Method: `COSMOS-style`.
- Implementation: `COSMOS-style direct preference conditioning`.
- Representative fidelity: `method-level adaptation`.
- Exact method reproduction: `False`.

## Dataset

- Protocol: `B`.
- Identity hash: `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.

## Smoke

- Train batches: `3`.
- Mean scalar loss: `-0.623458206653595`.
- Validation не запускалась.

## Preference Sensitivity

- Split: `train`.
- Preferences: `rank_heavy` vs `like_heavy`.
- Representation L2: `0.08870463818311691`.
- Ranking score L2: `0.0022082447540014982`.
- Ranking score mean abs: `3.9365506381727755e-05`.
- Passed: `True`.

## Cost

- Runtime sec: `166.985`.
- Peak VRAM bytes: `2422317056`.
- Params: `602398` trainable.
