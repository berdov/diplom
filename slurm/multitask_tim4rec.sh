#!/usr/bin/env bash
#SBATCH --job-name=multitask-tim4rec
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/daryumin/iberdov/diplom}"
ENV_DIR="${MULTITASK_TIM4REC_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PYTHON="${ENV_DIR}/bin/python"
RUN_ID="${MULTITASK_TIM4REC_RUN_ID:-multitask_tim4rec_sanity_001}"
CONFIG="${MULTITASK_TIM4REC_CONFIG:-${REPO_DIR}/experiments/multitask_tim4rec/config.yaml}"
RUNS_DIR="${REPO_DIR}/experiments/multitask_tim4rec/runs"
ARTIFACT_ROOT="${MULTITASK_TIM4REC_ARTIFACT_ROOT:-/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec}"

cd "${REPO_DIR}"
mkdir -p "${RUNS_DIR}" "${REPO_DIR}/experiments/multitask_tim4rec/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${ENV_DIR}" >&2
  exit 2
fi

echo "Slurm: partition=${SLURM_JOB_PARTITION:-gpu-ef-quick}"
echo "Slurm: constraint=${SLURM_JOB_CONSTRAINT:-type_e}"
echo "Slurm: gpus=${SLURM_JOB_GPUS:-gpu:1}"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-4}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Run ID: ${RUN_ID}"

"${PYTHON}" experiments/multitask_tim4rec/train.py \
  --config "${CONFIG}" \
  --run-id "${RUN_ID}" \
  --epochs "${MULTITASK_TIM4REC_EPOCHS:-5}" \
  --artifact-dir "${ARTIFACT_ROOT}/${RUN_ID}" \
  --result-json "${RUNS_DIR}/${RUN_ID}.json" \
  --notes "${RUNS_DIR}/${RUN_ID}_notes.md"
