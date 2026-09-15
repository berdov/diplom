#!/usr/bin/env bash
#SBATCH --job-name=mamba3-equiv
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/mamba3_timeaware/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/mamba3_timeaware/slurm_logs/%x-%j.err

set -euo pipefail

# Create slurm_logs before submitting; this script never submits another job.
REPO_DIR=/home/daryumin/iberdov/diplom
ENV_DIR=/home/daryumin/iberdov/diplom/envs/mamba3
cd "${REPO_DIR}"
module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TRITON_CACHE_DIR="${REPO_DIR}/experiments/mamba3_timeaware/slurm_logs/triton_cache"

test -x "${ENV_DIR}/bin/python"
echo "repo=$(git rev-parse HEAD)"
echo "branch=$(git branch --show-current)"
echo "job=${SLURM_JOB_ID:-unknown} node=${SLURM_JOB_NODELIST:-unknown}"
"${ENV_DIR}/bin/python" -m experiments.mamba3_timeaware.gpu_equivalence
