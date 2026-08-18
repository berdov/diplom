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
CONFIG="${TIM4REC_CONFIG:-${REPO_DIR}/experiments/tim4rec_baseline/config_kuairand.yaml}"
RUNS_DIR="${REPO_DIR}/experiments/tim4rec_baseline/runs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

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
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Stage: ${STAGE}"

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
  *)
    echo "Unknown TIM4REC_STAGE=${STAGE}; expected smoke or train" >&2
    exit 2
    ;;
esac
