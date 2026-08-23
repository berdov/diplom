#!/usr/bin/env bash
#SBATCH --job-name=ltr-xgb-optuna-final
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/daryumin/iberdov/diplom}"
PYTHON="${LTR_OPTUNA_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
CONFIG="${LTR_OPTUNA_CONFIG:-${PROJECT_ROOT}/experiments/ltr_xgb_optuna/config.yaml}"
BEST_PARAMS="${LTR_OPTUNA_BEST_PARAMS:-${PROJECT_ROOT}/experiments/ltr_xgb_optuna/best_params.yaml}"
RESULTS_CSV="${LTR_OPTUNA_RESULTS_CSV:-${PROJECT_ROOT}/experiments/results.csv}"
RUNNER="${PROJECT_ROOT}/experiments/ltr_xgb_optuna/run_best.py"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LTR_GIT_BRANCH="${LTR_GIT_BRANCH:-unknown}"
export LTR_GIT_COMMIT="${LTR_GIT_COMMIT:-unknown}"

echo "Slurm: job=${SLURM_JOB_ID:-unknown}"
echo "Slurm: partition=${SLURM_JOB_PARTITION:-unknown}"
echo "Slurm: constraint=${SLURM_JOB_CONSTRAINT:-type_d}"
echo "Slurm: gres=${SLURM_JOB_GRES:-none}"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-8}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Branch: ${LTR_GIT_BRANCH}"
echo "Commit: ${LTR_GIT_COMMIT}"
echo "Config: ${CONFIG}"
echo "Best params: ${BEST_PARAMS}"
echo "Results CSV: ${RESULTS_CSV}"

cd "${PROJECT_ROOT}"
"${PYTHON}" - <<'PY'
import xgboost
print("Python ok")
print("xgboost", xgboost.__version__)
PY
"${PYTHON}" -m py_compile "${RUNNER}"
"${PYTHON}" "${RUNNER}" \
  --config "${CONFIG}" \
  --best-params "${BEST_PARAMS}" \
  --results-csv "${RESULTS_CSV}" \
  --allow-test-evaluation \
  --write-results
