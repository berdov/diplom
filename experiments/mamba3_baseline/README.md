# Mamba-3 sequential baseline

This experiment starts the **new architecture stage** after the completed MTL/MOO study.

## Scientific role

`Mamba3Rec` is a deliberately simple **primary-only next-item baseline**:

`item sequence -> item embeddings -> stacked Mamba-3 blocks -> last valid state -> tied item scoring`

It does **not** contain TiM4Rec time-aware mechanisms, auxiliary tasks, EPO/MOO, prototypes, routing, MoE, flow matching, or any proposed novelty. Its purpose is to establish a clean Mamba-3 reference point before modifying the architecture.

The outer recommender scale stays close to the TiM4Rec reproduction (`hidden_size=64`, two sequence layers, residual FFN, same Protocol B, same initial learning rate and batch size). The Mamba-3 mixer itself uses the official SISO implementation with `d_state=128`, `headdim=64`, `chunk_size=64`, and bf16 mixer weights/activations. SISO is the default Mamba-3 formulation in the paper/code; MIMO is a later optional comparison, not part of this first reference run.

## Upstream

Official repository: `state-spaces/mamba`

Pinned source commit for this experiment:

`e9594ce1c732d97440f0332fdc43170a2294dbfa`

Mamba-3 paper: *Mamba-3: Improved Sequence Modeling using State Space Principles* (2026), arXiv:2603.15569.

The runner uses only full-sequence `Mamba3.forward(...)`. It does not use incremental `step()` / generation cache code.

## Evaluation protocol

Dataset and split are unchanged KuaiRand Protocol B:

- 23,951 users
- 7,111 items
- 1,134,420 interactions
- TRAIN 1,086,518
- VALID 23,951
- TEST 23,951
- chronological leave-one-out
- max sequence length 50
- full-catalog evaluation
- model selection by VALID NDCG@10

`run.py` checks the Protocol B manifest and, by default, sha256 of the RecBole `.inter` file.

**TEST is not evaluated by this experiment runner.** Every result JSON records `test_evaluation_count: 0`.

## Cluster isolation

Use an isolated checkout/worktree at:

`/home/daryumin/iberdov/diplom_exp_mamba3_baseline`

The immutable Protocol B data remain under `/home/daryumin/iberdov/diplom/data/processed/protocol_b`; Mamba-3 checkpoints/logs stay in the isolated experiment checkout. The Mamba-3 Python environment is separate as well: `/home/daryumin/iberdov/diplom/envs/mamba3`.

## Files

- `model.py` — RecBole `Mamba3Rec` model.
- `config_kuairand.yaml` — fixed initial baseline config; no hyperparameter tuning.
- `run.py` — smoke or full validation-only training.
- `ENVIRONMENT.md` — isolated environment requirements.
- `slurm/mamba3_baseline.sh` — cHARISMa A100 launcher.
- `runs/` — compact JSON summaries; checkpoints/logs remain untracked.

## Run sequence

1. Create/verify the isolated `envs/mamba3` environment.
2. Run `MAMBA3_STAGE=smoke sbatch slurm/mamba3_baseline.sh`.
3. Only if smoke has finite loss and gradients, run `MAMBA3_STAGE=train sbatch slurm/mamba3_baseline.sh`.
4. Compare VALID metrics with the appropriate validation controls. Do not use historical TEST TiM4Rec as if it were a same-split comparison.
5. Freeze the baseline before any Mamba-3 architectural modification.

No tuning should be started from the first result. The next research step is a literature-guided modification of Mamba-3 inside the full proposed pipeline.
