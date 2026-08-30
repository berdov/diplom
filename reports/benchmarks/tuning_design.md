# Controlled validation-only tuning design

Статус: `prepared_not_launched`.

Этот этап разрешает tuning только для четырех candidates: EPO, GradHV, COSMOS и PCGrad. STCH, FAMO, PHN и PaLoRA остаются rows family screening и не тюнятся в этой фазе.

## Protocol locks

- Split: validation only.
- TEST: loader не создается, evaluation не запускается, `test_evaluation_count = 0`.
- Dataset: KuaiRand Protocol B, identity hash `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.
- Seed: one fixed tuning seed `2026`; multi-seed confirmation является следующим этапом, не частью этого tuning search.
- Hardware: primary comparison on `A100-SXM4-80GB`, Slurm `type_e`.
- Max epochs: `100`; validation every `5`; minimum epochs before early stopping `20`; patience `3` validation checks.
- Primary Optuna objective: maximize validation ranking operating-point `NDCG@10`.
- Optuna pruning: disabled initially.

## Frozen scientific choices

- Objective definitions do not change.
- Normalization does not change.
- Training HV reference for GradHV does not change.
- Evaluation Pareto reference is frozen as `[1.0, 2.0, 2.0, 2.0, 2.0]` with `invalid_reference_policy=raise`.
- Ranking operating-point selection does not change.
- PCGrad projection algorithm remains ranking-anchored.
- `lambda_aux` and arbitrary per-task weights are frozen because the current MOO protocol optimizes normalized task objectives, not the historical scalar multitask total.

## Search spaces

The exact ranges and rationales live in `configs/moo_tuning_spaces.yaml`.

Shared parameters:

- `learning_rate`: log `3e-4 ... 3e-3`.
- `weight_decay`: log `1e-7 ... 1e-4`.
- `dropout_prob`: linear `0.0 ... 0.25`.
- `head_lr_multiplier`: log `0.25 ... 2.0`.

Method-specific parameters:

- COSMOS additionally tunes `dirichlet_alpha` and `lambda_cosine`.
- EPO keeps preference set and finite solution count fixed.
- GradHV keeps train HV reference and finite solution count fixed.
- PCGrad keeps projection algorithm, shared selector and task order fixed.

## Baseline trial guard

Trial 0 for every study is the current frozen long-convergence configuration. If trial 0 differs from the corresponding current run by more than `0.001` absolute NDCG@10, the study aborts before further search.

Current expected trial-0 anchors:

| Method | Baseline run | NDCG@10 | HR@10 |
| --- | --- | ---: | ---: |
| EPO | `epo_convergence_001` | 0.0584 | 0.1078 |
| GradHV | `gradhv_convergence_001` | 0.0486 | 0.0874 |
| COSMOS | `cosmos_convergence_001` | 0.0453 | 0.0810 |
| PCGrad | `pcgrad_convergence_001` | 0.0444 | 0.0790 |

PCGrad tuning is allowed only under the audit conclusion `EXPECTED_PROTOCOL_DIFFERENCE`; historical `pcgrad_001` is not a retroactive current default.

## Cluster commands

Submit one serial Optuna study per method to avoid unsafe concurrent writes to one SQLite study. Paths in `configs/moo_tuning_spaces.yaml` are relative to the active repo/worktree, so the same commit can run from a clean detached worktree when the named branch is occupied by existing cluster artifacts.

Run validation-only data preparation once before submitting parallel jobs; per-job wrapper does not rewrite shared RecBole files by default. The RecBole path used by the inherited multitask Optuna config is `/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/validation_only_recbole`.

```bash
/home/daryumin/iberdov/diplom/.conda/bin/python experiments/multitask_tim4rec_optuna/prepare_validation_only.py
```

```bash
MOO_TUNING_METHOD=epo sbatch --job-name=moo-epo-tuning slurm/moo_tuning.sh
MOO_TUNING_METHOD=gradhv sbatch --job-name=moo-gradhv-tuning slurm/moo_tuning.sh
MOO_TUNING_METHOD=cosmos sbatch --job-name=moo-cosmos-tuning slurm/moo_tuning.sh
MOO_TUNING_METHOD=pcgrad sbatch --job-name=moo-pcgrad-tuning slurm/moo_tuning.sh
```

Each method targets 24 completed trials unless overridden with `MOO_TUNING_TARGET_COMPLETE`.

After cluster completion, build compact reports and best configs:

```bash
python -m experiments.moo_8families.build_tuning_results --write-best-configs
```
