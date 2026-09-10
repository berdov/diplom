#!/usr/bin/env bash
set -euo pipefail
: "${SC_REPO:?}" "${SC_COMMIT:?}" "${SC_PYTHON:?}"
cd "$SC_REPO"
module load Python/miniconda
module load EasyBuild/modules_rocky
module load git/2.50.1-GCCcore-14.3.0
[[ "$(git rev-parse HEAD)" == "$SC_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$SC_REPO"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XDG_CACHE_HOME="$SC_REPO/experiments/stage_confirmation/runtime/cache/$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
mkdir -p "$XDG_CACHE_HOME"
if [[ "${SC_MODE:-full}" == summary ]]; then
 exec "$SC_PYTHON" -m experiments.stage_confirmation.summarize
fi
exec "$SC_PYTHON" -m experiments.stage_confirmation.run --index "${SLURM_ARRAY_TASK_ID:?}"
