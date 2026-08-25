#!/usr/bin/env bash
#SBATCH --job-name=multitask-tuned
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
ENV_DIR="${MULTITASK_TUNED_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PREP_PYTHON="${MULTITASK_TUNED_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
PYTHON="${ENV_DIR}/bin/python"
CONFIG="${MULTITASK_TUNED_CONFIG:-${REPO_DIR}/experiments/multitask_tim4rec_optuna/config.yaml}"
BEST_PARAMS="${MULTITASK_TUNED_BEST_PARAMS:-${REPO_DIR}/experiments/multitask_tim4rec_optuna/best_params.yaml}"
RUN_ID="${MULTITASK_TUNED_RUN_ID:-multitask_tim4rec_tuned_001}"
VALIDATION_TOLERANCE_NDCG10="${MULTITASK_TUNED_VALIDATION_TOLERANCE_NDCG10:-0.0005}"
VALIDATION_TOLERANCE_HR10="${MULTITASK_TUNED_VALIDATION_TOLERANCE_HR10:-0.0005}"

cd "${REPO_DIR}"
mkdir -p /home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/slurm_logs

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
echo "Repo: ${REPO_DIR}"
echo "Config: ${CONFIG}"
echo "Best params: ${BEST_PARAMS}"
echo "Run id: ${RUN_ID}"
echo "Validation tolerance NDCG@10: ${VALIDATION_TOLERANCE_NDCG10}"
echo "Validation tolerance HR@10: ${VALIDATION_TOLERANCE_HR10}"

EXTRA_ARGS=()
if [ "${MULTITASK_TUNED_RESUME_AFTER_GATE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--resume-after-validation-gate-diagnostic)
elif [ "${MULTITASK_TUNED_RECOVER_COMPLETED_GUARD:-0}" = "1" ]; then
  EXTRA_ARGS+=(--recover-completed-test-guard)
else
  "${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py
fi

"${PYTHON}" experiments/multitask_tim4rec_optuna/run_locked_tuned.py \
  --config "${CONFIG}" \
  --best-params "${BEST_PARAMS}" \
  --run-id "${RUN_ID}" \
  --prep-python "${PREP_PYTHON}" \
  --validation-tolerance-ndcg10 "${VALIDATION_TOLERANCE_NDCG10}" \
  --validation-tolerance-hr10 "${VALIDATION_TOLERANCE_HR10}" \
  "${EXTRA_ARGS[@]}"
