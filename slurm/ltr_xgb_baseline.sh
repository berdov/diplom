#!/usr/bin/env bash
#SBATCH --job-name=ltr-xgb-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --partition=cpu-e-quick
#SBATCH --constraint=type_e
#SBATCH --output=experiments/ltr_xgb_baseline/slurm_logs/%x-%j.out
#SBATCH --error=experiments/ltr_xgb_baseline/slurm_logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="/home/daryumin/iberdov/diplom"
PYTHON="${PROJECT_ROOT}/.conda/bin/python"
CONFIG="${PROJECT_ROOT}/experiments/ltr_xgb_baseline/config.yaml"
RUNNER="${PROJECT_ROOT}/experiments/ltr_xgb_baseline/run_experiment.py"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LTR_GIT_BRANCH="${LTR_GIT_BRANCH:-exp/ltr-xgb-baseline}"

if [[ -n "${LTR_SANITY:-}" && "${LTR_SANITY}" != "0" ]]; then
  SANITY_ARGS=(--sanity)
  WRITE_RESULTS="${LTR_WRITE_RESULTS:-0}"
else
  SANITY_ARGS=()
  WRITE_RESULTS="${LTR_WRITE_RESULTS:-1}"
fi

FORCE="${LTR_FORCE:-1}"
FORCE_ARGS=()
if [[ "${FORCE}" != "0" ]]; then
  FORCE_ARGS=(--force)
fi

RESULT_ARGS=()
if [[ "${WRITE_RESULTS}" != "0" ]]; then
  RESULT_ARGS=(--write-results)
fi

set +u
source /etc/profile >/dev/null 2>&1 || true
module load python/miniconda >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${PROJECT_ROOT}/.conda" || true
fi
set -u

cd "${PROJECT_ROOT}"
mkdir -p experiments/ltr_xgb_baseline/slurm_logs

export LTR_GIT_COMMIT="${LTR_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"

echo "Slurm: partition=${SLURM_JOB_PARTITION:-cpu-e-quick}"
echo "Slurm: constraint=type_e"
echo "Slurm: gres=none"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-8}"
echo "Slurm: mem=0"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Branch: ${LTR_GIT_BRANCH}"
echo "Commit: ${LTR_GIT_COMMIT}"
echo "Sanity: ${LTR_SANITY:-0}"
echo "Write results: ${WRITE_RESULTS}"

"${PYTHON}" --version
"${PYTHON}" - <<'PY'
import xgboost
print("xgboost", xgboost.__version__)
PY
"${PYTHON}" -m py_compile "${RUNNER}"
"${PYTHON}" "${RUNNER}" \
  --config "${CONFIG}" \
  --stage all \
  "${FORCE_ARGS[@]}" \
  "${SANITY_ARGS[@]}" \
  "${RESULT_ARGS[@]}"
