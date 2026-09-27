#!/usr/bin/env bash
#SBATCH --job-name=m3-siso-resume
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/attempt_003/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/mamba3_three_time/confirmation/slurm_logs/attempt_003/%x-%j.err
set -euo pipefail
cd /home/daryumin/iberdov/diplom
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TILELANG_CACHE_DIR="$PWD/experiments/mamba3_three_time/confirmation/slurm_logs/attempt_003/tilelang_cache"
: "${RUN_COMMIT:?exact published commit required}"
: "${EXPECTED_STUDY_HASH:?source hash required}"
: "${EXPECTED_CORE_HASH:?core hash required}"
if [[ "${1:-}" == "--runtime-preflight-only" ]]; then
    test "$#" -eq 2
    exec /home/daryumin/iberdov/diplom/envs/mamba3/bin/python -m experiments.mamba3_three_time.confirmation.resume_preflight --runtime-preflight-only "$2"
fi
test "$#" -eq 0
exec /home/daryumin/iberdov/diplom/envs/mamba3/bin/python -m experiments.mamba3_three_time.confirmation.resume_pipeline
