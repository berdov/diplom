# SISO dual/triple confirmation

KuaiRand Protocol B, VALID full-ranking; TEST NOT RUN. Seed2026: exploratory pilot, использованный для решения продолжить серию.
Seeds2027-2030 заранее зафиксированы. Нет заявлений о статистической значимости; эпохи не являются независимыми повторениями.
Время runs включает JIT. Историческая separate=.0633 не входит в агрегаты.

## new_four_pairs: full

Пары: 4/4; incomplete=False

| seed | dual | triple | paired delta | status / reason |
|---|---:|---:|---:|---|
| 2027 | 0.0632 | 0.0628 | -0.00040000000000001146 | dual: PASS ; triple: PASS  |
| 2028 | 0.0627 | 0.063 | 0.0002999999999999947 | dual: PASS ; triple: PASS  |
| 2029 | 0.0625 | 0.0635 | 0.0010000000000000009 | dual: PASS ; triple: PASS  |
| 2030 | 0.0627 | 0.0633 | 0.0005999999999999894 | dual: PASS ; triple: PASS  |

Все доступные успешные runs: {"dual": {"n": 4, "mean": 0.062775, "sample_std_ddof1": 0.0002986078811194839}, "triple": {"n": 4, "mean": 0.06315, "sample_std_ddof1": 0.00031091263510296197}}
Парные разности (только полные пары): {"n": 4, "mean": 0.0003749999999999934, "sample_std_ddof1": 0.0005909032633745321}
Положительные/отрицательные/нулевые: 3/1/0; relative gain: 0.5973715651135025%.

## new_four_pairs: first27

Пары: 4/4; incomplete=False

| seed | dual | triple | paired delta | status / reason |
|---|---:|---:|---:|---|
| 2027 | 0.0614 | 0.0623 | 0.000899999999999998 | dual: PASS ; triple: PASS  |
| 2028 | 0.0618 | 0.0623 | 0.0005000000000000004 | dual: PASS ; triple: PASS  |
| 2029 | 0.0619 | 0.0625 | 0.0006000000000000033 | dual: PASS ; triple: PASS  |
| 2030 | 0.0627 | 0.0625 | -0.00020000000000000573 | dual: PASS ; triple: PASS  |

Все доступные успешные runs: {"dual": {"n": 4, "mean": 0.061950000000000005, "sample_std_ddof1": 0.0005446711546122746}, "triple": {"n": 4, "mean": 0.0624, "sample_std_ddof1": 0.00011547005383792445}}
Парные разности (только полные пары): {"n": 4, "mean": 0.000449999999999999, "sample_std_ddof1": 0.00046547466812563377}
Положительные/отрицательные/нулевые: 3/1/0; relative gain: 0.726392251815966%.

## all_five_pairs: full

Пары: 5/5; incomplete=False

| seed | dual | triple | paired delta | status / reason |
|---|---:|---:|---:|---|
| 2026 | 0.0615 | 0.0623 | 0.0008000000000000021 | dual: PASS ; triple: PASS  |
| 2027 | 0.0632 | 0.0628 | -0.00040000000000001146 | dual: PASS ; triple: PASS  |
| 2028 | 0.0627 | 0.063 | 0.0002999999999999947 | dual: PASS ; triple: PASS  |
| 2029 | 0.0625 | 0.0635 | 0.0010000000000000009 | dual: PASS ; triple: PASS  |
| 2030 | 0.0627 | 0.0633 | 0.0005999999999999894 | dual: PASS ; triple: PASS  |

Все доступные успешные runs: {"dual": {"n": 5, "mean": 0.06252, "sample_std_ddof1": 0.000626099033699944}, "triple": {"n": 5, "mean": 0.06298, "sample_std_ddof1": 0.00046583258795408414}}
Парные разности (только полные пары): {"n": 5, "mean": 0.00045999999999999514, "sample_std_ddof1": 0.000545893762558252}
Положительные/отрицательные/нулевые: 4/1/0; relative gain: 0.7357645553422776%.

## all_five_pairs: first27

Пары: 5/5; incomplete=False

| seed | dual | triple | paired delta | status / reason |
|---|---:|---:|---:|---|
| 2026 | 0.0615 | 0.0623 | 0.0008000000000000021 | dual: PASS ; triple: PASS  |
| 2027 | 0.0614 | 0.0623 | 0.000899999999999998 | dual: PASS ; triple: PASS  |
| 2028 | 0.0618 | 0.0623 | 0.0005000000000000004 | dual: PASS ; triple: PASS  |
| 2029 | 0.0619 | 0.0625 | 0.0006000000000000033 | dual: PASS ; triple: PASS  |
| 2030 | 0.0627 | 0.0625 | -0.00020000000000000573 | dual: PASS ; triple: PASS  |

Все доступные успешные runs: {"dual": {"n": 5, "mean": 0.06186, "sample_std_ddof1": 0.0005128352561983251}, "triple": {"n": 5, "mean": 0.06238, "sample_std_ddof1": 0.00010954451150103256}}
Парные разности (только полные пары): {"n": 5, "mean": 0.0005199999999999996, "sample_std_ddof1": 0.0004324349662087955}
Положительные/отрицательные/нулевые: 4/1/0; relative gain: 0.8406078241189885%.


## Source index

| seed | mode | job | commit | source | SHA256 |
|---|---|---|---|---|---|
| 2026 | dual | 4353980 | 6b5618a769d8e3424df0b7232605426fac8c573a | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/validation_pilot/runs/mamba3_three_time_siso_dual_seed2026_001.json | 5a23da8bb5317a0a8f298902fe8b39e3e5e2b0c7893524595d84e5631f44caa7 |
| 2026 | triple | 4353980 | 6b5618a769d8e3424df0b7232605426fac8c573a | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/validation_pilot/runs/mamba3_three_time_siso_triple_seed2026_001.json | 0ec970f6b161ae39737cdf9b5e793b7d37d133d9e30d7b6cbd4882d6dd379500 |
| 2027 | dual | 4355052 | 3d3b305c4c74e208cd226d3ed4ccec37b4fd8311 | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/runs/mamba3_three_time_confirm_siso_dual_seed2027_001.json | 50ee336280fe241f3eaa97f56acc85d96243531623d5072841c4daa47baa0b58 |
| 2027 | triple | 4355314 | 995c5cde6449ea429c1d80ca6ab276b9791041c0 | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/runs/attempt_003/mamba3_three_time_confirm_siso_triple_seed2027_001.json | 4f11aad099ad252282a64e2e6aeaed6a235ed9779df8399c9f8fb0b762d56691 |
| 2028 | dual | 4355314 | 995c5cde6449ea429c1d80ca6ab276b9791041c0 | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/runs/attempt_003/mamba3_three_time_confirm_siso_dual_seed2028_001.json | 4dca5f313733e2d5a6978450705115aeaefbc4d97c47e380bdb254cc9edfdc68 |
| 2028 | triple | 4355314 | 995c5cde6449ea429c1d80ca6ab276b9791041c0 | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/runs/attempt_003/mamba3_three_time_confirm_siso_triple_seed2028_001.json | bcb43d53d6f08df479de21e05868edff6dcaf22e47590fb40dd4acae41b0829b |
| 2029 | dual | 4355314 | 995c5cde6449ea429c1d80ca6ab276b9791041c0 | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/runs/attempt_003/mamba3_three_time_confirm_siso_dual_seed2029_001.json | 61356e7c79e95eda91534a56fc9bcd87be557e12295ec9937ec0353a648aa105 |
| 2029 | triple | 4355314 | 995c5cde6449ea429c1d80ca6ab276b9791041c0 | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/runs/attempt_003/mamba3_three_time_confirm_siso_triple_seed2029_001.json | e5b7a84b1a5ba258b55a0fb5f1e06688fca0ac2e4fc4cced81727863ec4de54f |
| 2030 | dual | 4355314 | 995c5cde6449ea429c1d80ca6ab276b9791041c0 | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/runs/attempt_003/mamba3_three_time_confirm_siso_dual_seed2030_001.json | 7bd16779958cd08cb82a5b50dbf13ed6b51ec245d33a81bb0673925f9f7365f6 |
| 2030 | triple | 4355314 | 995c5cde6449ea429c1d80ca6ab276b9791041c0 | /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/runs/attempt_003/mamba3_three_time_confirm_siso_triple_seed2030_001.json | f8f1e4638552926b5cf72388adb7b101ca401158397289cb885dabd1d7098f85 |
