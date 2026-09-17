#!/usr/bin/env bash
#SBATCH --job-name=m3-time-confirm
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --array=0-8%1
#SBATCH --no-requeue
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/mamba3_time_confirmation/slurm_logs/%A_%a.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/mamba3_time_confirmation/slurm_logs/%A_%a.err
set -euo pipefail
cd /home/daryumin/iberdov/diplom
: "${RUN_COMMIT:?login-node checked exact commit required}"
: "${EXPECTED_CORE_HASH:?required}"
: "${EXPECTED_STUDY_HASH:?required}"
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN_ID=$(envs/mamba3/bin/python -c 'import os; from experiments.mamba3_time_confirmation.config import plan; print(plan()["tasks"][int(os.environ["SLURM_ARRAY_TASK_ID"])]["run_id"])')
export TRITON_CACHE_DIR="$PWD/experiments/mamba3_time_confirmation/slurm_logs/$RUN_ID/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/experiments/mamba3_time_confirmation/slurm_logs/$RUN_ID/cache/inductor"
export XDG_CACHE_HOME="$PWD/experiments/mamba3_time_confirmation/slurm_logs/$RUN_ID/cache/xdg"
export TORCH_EXTENSIONS_DIR="$PWD/experiments/mamba3_time_confirmation/slurm_logs/$RUN_ID/cache/torch_extensions"
export CUDA_CACHE_PATH="$PWD/experiments/mamba3_time_confirmation/slurm_logs/$RUN_ID/cache/cuda"
exec envs/mamba3/bin/python -m experiments.mamba3_time_confirmation.run
