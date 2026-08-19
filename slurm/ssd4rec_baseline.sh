#!/usr/bin/env bash
#SBATCH --job-name=ssd4rec-smoke
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/ssd4rec_baseline/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/ssd4rec_baseline/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/daryumin/iberdov/diplom}"
ENV_DIR="${SSD4REC_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/ssd4rec}"
PYTHON="${ENV_DIR}/bin/python"
CONFIG="${SSD4REC_CONFIG:-${REPO_DIR}/experiments/ssd4rec_baseline/config_kuairand.yaml}"
MANIFEST="${SSD4REC_MANIFEST:-${REPO_DIR}/outputs/data/protocol_b_manifest.json}"
RUNS_DIR="${REPO_DIR}/experiments/ssd4rec_baseline/runs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

cd "${REPO_DIR}"
mkdir -p "${RUNS_DIR}" "${REPO_DIR}/experiments/ssd4rec_baseline/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1

if [ ! -x "${PYTHON}" ]; then
  echo "Missing SSD4Rec env: ${ENV_DIR}" >&2
  echo "Create it using experiments/ssd4rec_baseline/environment.txt" >&2
  exit 2
fi
export PATH="${ENV_DIR}/bin:${PATH}"

"${PYTHON}" - <<'PY'
import sys
print(sys.version)
PY

echo "Slurm: partition=${SLURM_JOB_PARTITION:-gpu-ef-quick}"
echo "Slurm: constraint=type_e"
echo "Slurm: job gpus=${SLURM_JOB_GPUS:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unknown}"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-4}"
echo "Slurm: mem=0"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Config: ${CONFIG}"
echo "Manifest: ${MANIFEST}"

"${PYTHON}" experiments/ssd4rec_baseline/smoke_test.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --result-json "${RUNS_DIR}/smoke_${STAMP}.json"
