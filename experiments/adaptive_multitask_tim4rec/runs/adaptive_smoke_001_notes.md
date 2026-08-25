# adaptive_smoke_001

Smoke adaptive multitask optimization на реальных train batches KuaiRand Protocol B.

## Test safety

- `test_evaluation_count`: `0`.
- Test dataloader created: `False`.

## Fixed tuned gradient diagnostics

| task | raw loss | weighted loss | shared grad norm | cosine with rank |
|---|---:|---:|---:|---:|
| rank | 5.959172 | 5.959172 | 0.411454 | 1.000000 |
| is_click | 0.653172 | 0.022006 | 0.002114 | 0.001247 |
| long_view | 0.679393 | 0.021175 | 0.001902 | 0.001515 |
| is_like | 0.317194 | 0.135772 | 0.278751 | 0.085518 |
| is_profile_enter | 0.510580 | 0.017568 | 0.004159 | 0.005433 |

- Negative pairs: `3` / `10`.

## Adaptive smoke

- GradNorm: `completed`, weights `{'is_click': 0.255569189786911, 'long_view': 0.23643110692501068, 'is_like': 3.2469875812530518, 'is_profile_enter': 0.26101210713386536}` -> `{'is_click': 0.2878153622150421, 'long_view': 0.2661304175853729, 'is_like': 3.152064561843872, 'is_profile_enter': 0.29398998618125916}`, mean step `2.4663s`.
- PCGrad: `completed`, mode `ranking_anchored`, first-step conflicts `3` -> `3`, mean step `0.1972s`.
- MetaBalance: `completed`, relax `0.7`, beta `0.9`, mean step `0.2493s`.

## Recommendation

- Next sanity methods: `ranking_anchored_pcgrad, gradnorm_auxiliary_only`.
- Most relevant for ranking-primary setup: `ranking_anchored_pcgrad`.
- Real gradient conflicts detected: `True`.
