#!/usr/bin/env bash
#SBATCH --job-name=rt-mamba3
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/mamba3_timeaware/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/mamba3_timeaware/slurm_logs/%x-%j.err

set -euo pipefail
MODE="${1:?smoke or train required}"
case "$MODE" in smoke|train) ;; *) exit 2 ;; esac
cd /home/daryumin/iberdov/diplom
: "${RUN_COMMIT:?export submit-side git SHA with sbatch}"
ENV_DIR=/home/daryumin/iberdov/diplom/envs/mamba3
module load Python/miniconda || true
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TRITON_CACHE_DIR="$PWD/experiments/mamba3_timeaware/slurm_logs/triton_cache"
echo "commit=$RUN_COMMIT job=$SLURM_JOB_ID mode=$MODE"
"${ENV_DIR}/bin/python" -m experiments.mamba3_timeaware.run --mode "$MODE"
