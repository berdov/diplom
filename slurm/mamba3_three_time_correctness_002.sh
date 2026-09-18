#!/usr/bin/env bash
#SBATCH --job-name=m3-three-check2
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/mamba3_three_time/slurm_logs/attempt_002/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/mamba3_three_time/slurm_logs/attempt_002/%x-%j.err
set -euo pipefail
cd /home/daryumin/iberdov/diplom
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TRITON_CACHE_DIR="$PWD/experiments/mamba3_three_time/slurm_logs/attempt_002/triton_cache"
export TILELANG_CACHE_DIR="$PWD/experiments/mamba3_three_time/slurm_logs/attempt_002/tilelang_cache"
: "${RUN_COMMIT:?login-verified execution commit required}"
: "${EXPECTED_STUDY_HASH:?source hash required}"
: "${EXPECTED_CORE_HASH:?core hash required}"
envs/mamba3/bin/python -m experiments.mamba3_three_time.pipeline_002
