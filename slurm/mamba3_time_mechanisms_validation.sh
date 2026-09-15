#!/usr/bin/env bash
#SBATCH --job-name=time-mech-val
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/mamba3_time_mechanisms/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/mamba3_time_mechanisms/slurm_logs/%x-%j.err
set -euo pipefail
case "${1:-}" in decay_only|scan_only|separate) MODE="$1";; *) exit 2;; esac
cd /home/daryumin/iberdov/diplom
: "${RUN_COMMIT:?export exact git SHA before future submit}"
test "$(git rev-parse HEAD)" = "$RUN_COMMIT"
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TRITON_CACHE_DIR="$PWD/experiments/mamba3_time_mechanisms/slurm_logs/triton_cache"
envs/mamba3/bin/python -m experiments.mamba3_time_mechanisms.run --mode "$MODE" \
  --equivalence-json experiments/mamba3_time_mechanisms/runs/gpu_equivalence_001.json
