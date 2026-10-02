#!/bin/bash
#SBATCH --job-name=m3-gapcenter2
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@600
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/mamba3_gap_trap/centered/slurm_logs/attempt_002/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/mamba3_gap_trap/centered/slurm_logs/attempt_002/%x-%j.err
set -euo pipefail
cd "${REPO_ROOT:-/home/daryumin/iberdov/diplom}"
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TILELANG_CACHE_DIR="$PWD/experiments/mamba3_gap_trap/centered/slurm_logs/attempt_002/tilelang_cache"
PYTHON="$PWD/envs/mamba3/bin/python"
if [[ "${1:-}" == "--preflight-only" ]]; then
    shift
    CUDA_VISIBLE_DEVICES="" exec "$PYTHON" -B -m experiments.mamba3_gap_trap.centered.preflight "$@"
fi
test "$#" -eq 0
: "${RUN_COMMIT:?exact published commit required}"
: "${EXPECTED_STUDY_HASH:?source hash required}"
: "${RESERVATION_TOKEN:?immutable reservation required}"
exec "$PYTHON" -B -m experiments.mamba3_gap_trap.centered.pipeline
