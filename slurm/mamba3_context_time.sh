#!/usr/bin/env bash
#SBATCH --job-name=m3-context-time
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/mamba3_context_time/slurm_logs/%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/mamba3_context_time/slurm_logs/%j.err
set -euo pipefail
cd /home/daryumin/iberdov/diplom
: "${RUN_COMMIT:?login verified exact commit required}"
: "${EXPECTED_CORE_HASH:?required}"
: "${EXPECTED_STUDY_HASH:?required}"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec envs/mamba3/bin/python -m experiments.mamba3_context_time.pipeline
