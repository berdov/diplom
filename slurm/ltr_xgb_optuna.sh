#!/usr/bin/env bash
#SBATCH --job-name=ltr-xgb-optuna-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-e-quick
#SBATCH --constraint=type_e
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/ltr_xgb_optuna/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/ltr_xgb_optuna/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/daryumin/iberdov/diplom}"
PYTHON="${LTR_OPTUNA_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
CONDA_ENV="${LTR_OPTUNA_CONDA_ENV:-/home/daryumin/iberdov/diplom/.conda}"
CONFIG="${LTR_OPTUNA_CONFIG:-${PROJECT_ROOT}/experiments/ltr_xgb_optuna/config.yaml}"
SEARCH_SPACE="${LTR_OPTUNA_SEARCH_SPACE:-${PROJECT_ROOT}/experiments/ltr_xgb_optuna/search_space.yaml}"
RUNNER="${PROJECT_ROOT}/experiments/ltr_xgb_optuna/optuna_search.py"
N_TRIALS="${LTR_OPTUNA_N_TRIALS:-1}"
REQUESTED_CONSTRAINT="${LTR_OPTUNA_CONSTRAINT:-${SLURM_JOB_CONSTRAINT:-type_e}}"
MODE="${LTR_OPTUNA_MODE:-smoke}"
TARGET_COMPLETE="${LTR_OPTUNA_TARGET_COMPLETE:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${PROJECT_ROOT}"
mkdir -p experiments/ltr_xgb_optuna/slurm_logs experiments/ltr_xgb_optuna/runs

set +u
source /etc/profile >/dev/null 2>&1 || true
module load python/miniconda >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${CONDA_ENV}" || true
fi
set -u

export LTR_OPTUNA_GIT_BRANCH="${LTR_OPTUNA_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
export LTR_OPTUNA_GIT_COMMIT="${LTR_OPTUNA_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"

echo "Slurm: partition=${SLURM_JOB_PARTITION:-cpu-e-quick}"
echo "Slurm: constraint=${REQUESTED_CONSTRAINT}"
echo "Slurm: gres=none"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-8}"
echo "Slurm: mem=0"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Branch: ${LTR_OPTUNA_GIT_BRANCH}"
echo "Commit: ${LTR_OPTUNA_GIT_COMMIT}"
echo "Mode: ${MODE}"
echo "Trials: ${N_TRIALS}"
echo "Target complete: ${TARGET_COMPLETE:-none}"
echo "Config: ${CONFIG}"
echo "Search space: ${SEARCH_SPACE}"

"${PYTHON}" --version
"${PYTHON}" - <<'PY'
import optuna
import xgboost
print("xgboost", xgboost.__version__)
print("optuna", optuna.__version__)
PY
"${PYTHON}" -m py_compile "${RUNNER}" "${PROJECT_ROOT}/experiments/ltr_xgb_optuna/run_best.py"
RUN_ARGS=(
  --config "${CONFIG}"
  --search-space "${SEARCH_SPACE}"
)
if [[ "${MODE}" == "smoke" ]]; then
  RUN_ARGS+=(--smoke --n-trials "${N_TRIALS}")
elif [[ "${MODE}" == "search" ]]; then
  if [[ -n "${TARGET_COMPLETE}" ]]; then
    RUN_ARGS+=(--target-complete "${TARGET_COMPLETE}")
  else
    RUN_ARGS+=(--n-trials "${N_TRIALS}")
  fi
else
  echo "Unknown LTR_OPTUNA_MODE=${MODE}; expected smoke or search" >&2
  exit 2
fi

"${PYTHON}" "${RUNNER}" "${RUN_ARGS[@]}"
