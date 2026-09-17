#!/usr/bin/env bash
#SBATCH --job-name=time-mech-eq
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/mamba3_time_mechanisms/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/mamba3_time_mechanisms/slurm_logs/%x-%j.err
set -euo pipefail
cd /home/daryumin/iberdov/diplom
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TRITON_CACHE_DIR="$PWD/experiments/mamba3_time_mechanisms/slurm_logs/triton_cache"
envs/mamba3/bin/python -m experiments.mamba3_time_mechanisms.gpu_equivalence \
  --output experiments/mamba3_time_mechanisms/runs/gpu_equivalence_001.json
