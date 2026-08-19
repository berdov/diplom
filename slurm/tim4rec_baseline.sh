#!/usr/bin/env bash
#SBATCH --job-name=tim4rec-baseline
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/daryumin/iberdov/diplom}"
ENV_DIR="${TIM4REC_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PYTHON="${ENV_DIR}/bin/python"
STAGE="${TIM4REC_STAGE:-smoke}"
RUN_ID="${TIM4REC_RUN_ID:-tim4rec_sanity_001}"
CONFIG="${TIM4REC_CONFIG:-${REPO_DIR}/experiments/tim4rec_baseline/config_kuairand.yaml}"
RUNS_DIR="${REPO_DIR}/experiments/tim4rec_baseline/runs"
ARTIFACT_ROOT="${TIM4REC_ARTIFACT_ROOT:-/home/daryumin/iberdov/diplom/experiments/tim4rec_baseline}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [ "${STAGE}" = "reproduce" ] && [ "${RUN_ID}" = "tim4rec_sanity_001" ]; then
  RUN_ID="tim4rec_001"
fi

cd "${REPO_DIR}"
mkdir -p "${RUNS_DIR}" "${REPO_DIR}/experiments/tim4rec_baseline/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${ENV_DIR}" >&2
  echo "Create it using experiments/tim4rec_baseline/environment.txt" >&2
  exit 2
fi
export PATH="${ENV_DIR}/bin:${PATH}"

"${PYTHON}" - <<'PY'
import sys
print(sys.version)
PY

echo "Slurm: partition=${SLURM_JOB_PARTITION:-gpu-ef-quick}"
echo "Slurm: constraint=type_e"
echo "Slurm: gres=${SLURM_JOB_GPUS:-gpu:1}"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-4}"
echo "Slurm: mem=0"
echo "Slurm: mem per node=${SLURM_MEM_PER_NODE:-unknown}"
echo "Slurm: mem per cpu=${SLURM_MEM_PER_CPU:-unknown}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Stage: ${STAGE}"
echo "Run ID: ${RUN_ID}"

case "${STAGE}" in
  smoke)
    "${PYTHON}" experiments/tim4rec_baseline/smoke_test.py \
      --config "${CONFIG}" \
      --result-json "${RUNS_DIR}/smoke_${STAMP}.json"
    ;;
  train)
    "${PYTHON}" experiments/tim4rec_baseline/train.py \
      --config "${CONFIG}" \
      --result-json "${RUNS_DIR}/train_${STAMP}.json"
    ;;
  sanity)
    "${PYTHON}" experiments/tim4rec_baseline/sanity_train.py \
      --config "${CONFIG}" \
      --run-id "${RUN_ID}" \
      --epochs "${TIM4REC_SANITY_EPOCHS:-5}" \
      --artifact-dir "${ARTIFACT_ROOT}/${RUN_ID}" \
      --result-json "${RUNS_DIR}/${RUN_ID}.json"
    ;;
  reproduce)
    "${PYTHON}" experiments/tim4rec_baseline/reproduce.py \
      --config "${CONFIG}" \
      --run-id "${RUN_ID}" \
      --epochs "${TIM4REC_EPOCHS:-300}" \
      --artifact-dir "${ARTIFACT_ROOT}/${RUN_ID}" \
      --result-json "${RUNS_DIR}/${RUN_ID}.json"
    ;;
  *)
    echo "Unknown TIM4REC_STAGE=${STAGE}; expected smoke, train, sanity or reproduce" >&2
    exit 2
    ;;
esac
