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
STAGE="${SSD4REC_STAGE:-smoke}"
if [ -n "${SSD4REC_RUN_ID:-}" ]; then
  RUN_ID="${SSD4REC_RUN_ID}"
elif [ "${STAGE}" = "full" ]; then
  RUN_ID="ssd4rec_001"
else
  RUN_ID="ssd4rec_sanity_001"
fi
CONFIG="${SSD4REC_CONFIG:-${REPO_DIR}/experiments/ssd4rec_baseline/config_kuairand.yaml}"
MANIFEST="${SSD4REC_MANIFEST:-${REPO_DIR}/outputs/data/protocol_b_manifest.json}"
RUNS_DIR="${REPO_DIR}/experiments/ssd4rec_baseline/runs"
ARTIFACT_ROOT="${SSD4REC_ARTIFACT_ROOT:-/home/daryumin/iberdov/diplom/experiments/ssd4rec_baseline}"
REQUESTED_CONSTRAINT="${SSD4REC_CONSTRAINT:-${SLURM_JOB_CONSTRAINT:-type_e}}"
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
echo "Slurm: constraint=${REQUESTED_CONSTRAINT}"
echo "Slurm: requested gpus=1"
echo "Slurm: job GPU ids=${SLURM_JOB_GPUS:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unknown}"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-4}"
echo "Slurm: mem=0"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Config: ${CONFIG}"
echo "Manifest: ${MANIFEST}"
echo "Stage: ${STAGE}"
echo "Run ID: ${RUN_ID}"

case "${STAGE}" in
  smoke)
    "${PYTHON}" experiments/ssd4rec_baseline/smoke_test.py \
      --config "${CONFIG}" \
      --manifest "${MANIFEST}" \
      --result-json "${RUNS_DIR}/smoke_${STAMP}.json"
    ;;
  sanity)
    "${PYTHON}" experiments/ssd4rec_baseline/sanity_train.py \
      --config "${CONFIG}" \
      --manifest "${MANIFEST}" \
      --run-id "${RUN_ID}" \
      --epochs 5 \
      --artifact-dir "${ARTIFACT_ROOT}/${RUN_ID}" \
      --result-json "${RUNS_DIR}/${RUN_ID}.json" \
      --notes-md "${RUNS_DIR}/${RUN_ID}_notes.md"
    ;;
  full)
    "${PYTHON}" experiments/ssd4rec_baseline/full_train.py \
      --config "${CONFIG}" \
      --manifest "${MANIFEST}" \
      --run-id "${RUN_ID}" \
      --epochs 300 \
      --artifact-dir "${ARTIFACT_ROOT}/${RUN_ID}" \
      --result-json "${RUNS_DIR}/${RUN_ID}.json" \
      --notes-md "${RUNS_DIR}/${RUN_ID}_notes.md"
    ;;
  *)
    echo "Unknown SSD4REC_STAGE=${STAGE}; expected smoke, sanity or full" >&2
    exit 2
    ;;
esac
