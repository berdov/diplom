# SISO dual/triple confirmation

KuaiRand Protocol B, VALID full-ranking; TEST NOT RUN. Seed2026: exploratory pilot, использованный для решения продолжить серию.
Seeds2027-2030 заранее зафиксированы. Нет заявлений о статистической значимости; эпохи не являются независимыми повторениями.
Время runs включает JIT. Историческая separate=.0633 не входит в агрегаты.

## new_four_pairs: full

Пары: 0/4; incomplete=True

| seed | dual | triple | paired delta | status / reason |
|---|---:|---:|---:|---|
| 2027 | 0.0632 | None | None | dual: PASS ; triple: FAIL ValueError('Pair mismatch: optimizer_settings') |
| 2028 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |
| 2029 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |
| 2030 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |

Все доступные успешные runs: {"dual": {"n": 1, "mean": 0.0632, "sample_std_ddof1": null}, "triple": {"n": 0, "mean": null, "sample_std_ddof1": null}}
Парные разности (только полные пары): {"n": 0, "mean": null, "sample_std_ddof1": null}
Положительные/отрицательные/нулевые: 0/0/0; relative gain: None%.

## new_four_pairs: first27

Пары: 0/4; incomplete=True

| seed | dual | triple | paired delta | status / reason |
|---|---:|---:|---:|---|
| 2027 | 0.0614 | None | None | dual: PASS ; triple: FAIL ValueError('Pair mismatch: optimizer_settings') |
| 2028 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |
| 2029 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |
| 2030 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |

Все доступные успешные runs: {"dual": {"n": 1, "mean": 0.0614, "sample_std_ddof1": null}, "triple": {"n": 0, "mean": null, "sample_std_ddof1": null}}
Парные разности (только полные пары): {"n": 0, "mean": null, "sample_std_ddof1": null}
Положительные/отрицательные/нулевые: 0/0/0; relative gain: None%.

## all_five_pairs: full

Пары: 1/5; incomplete=True

| seed | dual | triple | paired delta | status / reason |
|---|---:|---:|---:|---|
| 2026 | 0.0615 | 0.0623 | 0.0008000000000000021 | dual: PASS ; triple: PASS  |
| 2027 | 0.0632 | None | None | dual: PASS ; triple: FAIL ValueError('Pair mismatch: optimizer_settings') |
| 2028 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |
| 2029 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |
| 2030 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |

Все доступные успешные runs: {"dual": {"n": 2, "mean": 0.06235, "sample_std_ddof1": 0.0012020815280171357}, "triple": {"n": 1, "mean": 0.0623, "sample_std_ddof1": null}}
Парные разности (только полные пары): {"n": 1, "mean": 0.0008000000000000021, "sample_std_ddof1": null}
Положительные/отрицательные/нулевые: 1/0/0; relative gain: None%.

## all_five_pairs: first27

Пары: 1/5; incomplete=True

| seed | dual | triple | paired delta | status / reason |
|---|---:|---:|---:|---|
| 2026 | 0.0615 | 0.0623 | 0.0008000000000000021 | dual: PASS ; triple: PASS  |
| 2027 | 0.0614 | None | None | dual: PASS ; triple: FAIL ValueError('Pair mismatch: optimizer_settings') |
| 2028 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |
| 2029 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |
| 2030 | None | None | None | dual: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log'); triple: NOT_RUN RuntimeError('experiments.mamba3_three_time.confirmation.runner exited 1; see /home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/mamba3_three_time_confirm_siso_triple_seed2027_001/stderr.log') |

Все доступные успешные runs: {"dual": {"n": 2, "mean": 0.061450000000000005, "sample_std_ddof1": 7.071067811865187e-05}, "triple": {"n": 1, "mean": 0.0623, "sample_std_ddof1": null}}
Парные разности (только полные пары): {"n": 1, "mean": 0.0008000000000000021, "sample_std_ddof1": null}
Положительные/отрицательные/нулевые: 1/0/0; relative gain: None%.

