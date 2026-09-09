#!/usr/bin/env bash
set -euo pipefail
: "${TC_REPO:?}" "${TC_COMMIT:?}" "${TC_PYTHON:?}" "${TC_MODE:?}"
cd "${TC_REPO}"
module load Python/miniconda
module load EasyBuild/modules_rocky
module load git/2.50.1-GCCcore-14.3.0
[[ "$(git rev-parse HEAD)" == "${TC_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ -x "${TC_PYTHON}" ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="${TC_REPO}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Per-job caches avoid writes/races in other experiments or environments.
export XDG_CACHE_HOME="${TC_REPO}/experiments/target_combination_analysis/runtime/cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton" CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda" MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib" TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
mkdir -p "${XDG_CACHE_HOME}"
case "${TC_MODE}" in
 smoke) exec "${TC_PYTHON}" -m experiments.target_combination_analysis.run --smoke ;;
 full) : "${SLURM_ARRAY_TASK_ID:?}"; exec "${TC_PYTHON}" -m experiments.target_combination_analysis.run --index "${SLURM_ARRAY_TASK_ID}" ;;
 summary) exec "${TC_PYTHON}" -m experiments.target_combination_analysis.summarize ;;
 *) exit 2 ;;
esac
