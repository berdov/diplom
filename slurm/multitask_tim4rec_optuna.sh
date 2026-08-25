#!/usr/bin/env bash
#SBATCH --job-name=multitask-optuna
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/daryumin/iberdov/diplom}"
ENV_DIR="${MULTITASK_OPTUNA_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PREP_PYTHON="${MULTITASK_OPTUNA_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
PYTHON="${ENV_DIR}/bin/python"
STAGE="${MULTITASK_OPTUNA_STAGE:-smoke}"
CONFIG="${MULTITASK_OPTUNA_CONFIG:-${REPO_DIR}/experiments/multitask_tim4rec_optuna/config.yaml}"
SEARCH_SPACE="${MULTITASK_OPTUNA_SEARCH_SPACE:-${REPO_DIR}/experiments/multitask_tim4rec_optuna/search_space.yaml}"
TARGET_COMPLETE="${MULTITASK_OPTUNA_TARGET_COMPLETE:-60}"

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/experiments/multitask_tim4rec_optuna/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${ENV_DIR}" >&2
  exit 2
fi

if [ ! -x "${PREP_PYTHON}" ]; then
  echo "Missing prep Python: ${PREP_PYTHON}" >&2
  exit 2
fi

echo "Slurm: partition=${SLURM_JOB_PARTITION:-gpu-ef-quick}"
echo "Slurm: constraint=${SLURM_JOB_CONSTRAINT:-type_e}"
echo "Slurm: gpus=${SLURM_JOB_GPUS:-gpu:1}"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-4}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Stage: ${STAGE}"
echo "Config: ${CONFIG}"

"${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py

case "${STAGE}" in
  smoke)
    "${PYTHON}" experiments/multitask_tim4rec_optuna/optuna_search.py \
      --config "${CONFIG}" \
      --search-space "${SEARCH_SPACE}" \
      --smoke
    ;;
  search)
    "${PYTHON}" experiments/multitask_tim4rec_optuna/optuna_search.py \
      --config "${CONFIG}" \
      --search-space "${SEARCH_SPACE}" \
      --target-complete "${TARGET_COMPLETE}"
    ;;
  summary)
    "${PYTHON}" experiments/multitask_tim4rec_optuna/optuna_search.py \
      --config "${CONFIG}" \
      --search-space "${SEARCH_SPACE}" \
      --summary-only
    ;;
  *)
    echo "Unknown MULTITASK_OPTUNA_STAGE=${STAGE}; expected smoke, search, or summary" >&2
    exit 2
    ;;
esac
