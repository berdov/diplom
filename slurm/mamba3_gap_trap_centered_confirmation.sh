#!/bin/bash
#SBATCH --job-name=m3-gapconfirm
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@600
set -euo pipefail
cd "${REPO_ROOT:-/home/daryumin/iberdov/diplom}"
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON="$PWD/envs/mamba3/bin/python"
if [[ "${1:-}" == "--preflight-only" ]]; then
    shift
    CUDA_VISIBLE_DEVICES="" exec "$PYTHON" -B -m experiments.mamba3_gap_trap.centered.confirmation.preflight "$@"
fi
test "$#" -eq 1
test "$1" = 001 -o "$1" = 002
export TILELANG_CACHE_DIR="$PWD/experiments/mamba3_gap_trap/centered/confirmation/slurm_logs/attempt_$1/tilelang_cache"
: "${RUN_COMMIT:?exact published commit required}"
: "${EXPECTED_STUDY_HASH:?source hash required}"
: "${RESERVATION_TOKEN:?immutable reservation required}"
exec "$PYTHON" -B -m experiments.mamba3_gap_trap.centered.confirmation.pipeline --attempt "$1"
