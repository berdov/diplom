#!/usr/bin/env bash
#SBATCH --job-name=mamba3-base
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom_exp_mamba3_baseline/experiments/mamba3_baseline/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom_exp_mamba3_baseline/experiments/mamba3_baseline/slurm_logs/%x-%j.err

set -euo pipefail

DEFAULT_REPO_DIR="/home/daryumin/iberdov/diplom_exp_mamba3_baseline"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_REPO_DIR}}}"
if [ ! -f "${REPO_DIR}/experiments/mamba3_baseline/run.py" ]; then
  REPO_DIR="${DEFAULT_REPO_DIR}"
fi

ENV_DIR="${MAMBA3_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/mamba3}"
PYTHON="${ENV_DIR}/bin/python"
CONFIG="${MAMBA3_CONFIG:-${REPO_DIR}/experiments/mamba3_baseline/config_kuairand.yaml}"
STAGE="${MAMBA3_STAGE:-smoke}"

case "${STAGE}" in
  smoke)
    RUN_ID="${MAMBA3_RUN_ID:-mamba3_smoke_001}"
    ;;
  train)
    RUN_ID="${MAMBA3_RUN_ID:-mamba3_validation_001}"
    ;;
  *)
    echo "Unknown MAMBA3_STAGE=${STAGE}; expected smoke or train" >&2
    exit 2
    ;;
esac

RUNS_DIR="${REPO_DIR}/experiments/mamba3_baseline/runs"
RESULT_JSON="${MAMBA3_RESULT_JSON:-${RUNS_DIR}/${RUN_ID}.json}"

cd "${REPO_DIR}"
mkdir -p "${RUNS_DIR}" "${REPO_DIR}/experiments/mamba3_baseline/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/experiments/mamba3_baseline:${REPO_DIR}:${PYTHONPATH:-}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing Mamba-3 env: ${ENV_DIR}" >&2
  echo "See experiments/mamba3_baseline/ENVIRONMENT.md" >&2
  exit 2
fi

if command -v git >/dev/null 2>&1; then
  echo "repo=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
else
  echo "repo=unknown"
  echo "branch=unknown"
fi

echo "stage=${STAGE}"
echo "run_id=${RUN_ID}"
echo "node=${SLURM_JOB_NODELIST:-unknown}"

"${PYTHON}" experiments/mamba3_baseline/run.py \
  --config "${CONFIG}" \
  --mode "${STAGE}" \
  --run-id "${RUN_ID}" \
  --result-json "${RESULT_JSON}"

echo "result=${RESULT_JSON}"
