#!/bin/bash
#SBATCH --job-name=m3-layer-time
#SBATCH --partition=rocky
#SBATCH --account=proj_1833
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@600
#SBATCH --output=experiments/mamba3_layer_temporal/slurm_logs/attempt_001/m3-layer-time-%j.out
#SBATCH --error=experiments/mamba3_layer_temporal/slurm_logs/attempt_001/m3-layer-time-%j.err
set -euo pipefail
cd "${REPO_ROOT:-/home/daryumin/iberdov/diplom}"
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TILELANG_CACHE_DIR="$PWD/experiments/mamba3_layer_temporal/slurm_logs/attempt_001/tilelang_cache"
PYTHON="$PWD/envs/mamba3/bin/python"
if [[ "${1:-}" == "--preflight-only" ]]; then
    shift
    CUDA_VISIBLE_DEVICES="" exec "$PYTHON" -B -m experiments.mamba3_layer_temporal.preflight "$@"
fi
: "${RUN_COMMIT:?exact published commit required}"
: "${EXPECTED_STUDY_HASH:?source hash required}"
: "${RESERVATION_TOKEN:?immutable reservation required}"
exec "$PYTHON" -B -m experiments.mamba3_layer_temporal.pipeline
