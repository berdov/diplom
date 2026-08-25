#!/usr/bin/env bash
#SBATCH --job-name=adaptive-pcgrad-full
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e|type_f
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/adaptive_multitask_tim4rec/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/adaptive_multitask_tim4rec/slurm_logs/%x-%j.err

set -euo pipefail

RUN_ID="${ADAPTIVE_MTL_RUN_ID:-pcgrad_001}"
if [ "${RUN_ID}" != "pcgrad_001" ]; then
  echo "adaptive_multitask_pcgrad_full.sh is locked to RUN_ID=pcgrad_001, got ${RUN_ID}" >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-/home/daryumin/iberdov/diplom}"
ENV_DIR="${ADAPTIVE_MTL_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PREP_PYTHON="${ADAPTIVE_MTL_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
PYTHON="${ENV_DIR}/bin/python"
CONFIG="${ADAPTIVE_MTL_CONFIG:-${REPO_DIR}/experiments/adaptive_multitask_tim4rec/config.yaml}"
ARTIFACT_DIR="${ADAPTIVE_MTL_ARTIFACT_DIR:-${REPO_DIR}/experiments/adaptive_multitask_tim4rec/${RUN_ID}}"
OUTPUT_JSON="${ADAPTIVE_MTL_OUTPUT_JSON:-${REPO_DIR}/experiments/adaptive_multitask_tim4rec/runs/${RUN_ID}.json}"
NOTES_MD="${ADAPTIVE_MTL_NOTES_MD:-${REPO_DIR}/experiments/adaptive_multitask_tim4rec/runs/${RUN_ID}_notes.md}"
MAX_EPOCHS="${ADAPTIVE_MTL_MAX_EPOCHS:-300}"
PATIENCE="${ADAPTIVE_MTL_PATIENCE:-10}"
DIAGNOSTIC_BATCHES="${ADAPTIVE_MTL_DIAGNOSTIC_BATCHES:-10}"
DIAGNOSTIC_EPOCHS="${ADAPTIVE_MTL_DIAGNOSTIC_EPOCHS:-1,3,5,10}"
SOFT_TIME_LIMIT_SEC="${ADAPTIVE_MTL_SOFT_TIME_LIMIT_SEC:-0}"
RESUME_ARGS=()
if [ "${ADAPTIVE_MTL_RESUME:-0}" = "1" ]; then
  RESUME_ARGS+=(--resume)
fi
if [ "${SOFT_TIME_LIMIT_SEC}" != "0" ]; then
  RESUME_ARGS+=(--soft-time-limit-sec "${SOFT_TIME_LIMIT_SEC}")
fi

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/experiments/adaptive_multitask_tim4rec/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export ADAPTIVE_MTL_GIT_COMMIT="${ADAPTIVE_MTL_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export ADAPTIVE_MTL_GIT_BRANCH="${ADAPTIVE_MTL_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
export ADAPTIVE_MTL_GIT_REMOTE="${ADAPTIVE_MTL_GIT_REMOTE:-$(git config --get remote.origin.url 2>/dev/null || echo unknown)}"
if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
  ACTUAL_JOB_FEATURES="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | sed -n 's/.*Features=\([^ ]*\).*/\1/p' | head -n 1)"
else
  ACTUAL_JOB_FEATURES=""
fi
export ADAPTIVE_MTL_SLURM_CONSTRAINT="${ADAPTIVE_MTL_SLURM_CONSTRAINT:-${ACTUAL_JOB_FEATURES:-${SLURM_JOB_CONSTRAINT:-type_e|type_f}}}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${ENV_DIR}" >&2
  exit 2
fi

if [ ! -x "${PREP_PYTHON}" ]; then
  echo "Missing prep Python: ${PREP_PYTHON}" >&2
  exit 2
fi

echo "Slurm: partition=${SLURM_JOB_PARTITION:-gpu-ef-quick}"
echo "Slurm: constraint=${ADAPTIVE_MTL_SLURM_CONSTRAINT}"
echo "Slurm: gpus=${SLURM_JOB_GPUS:-gpu:1}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Repo: ${REPO_DIR}"
echo "Run id: ${RUN_ID}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Patience: ${PATIENCE}"
echo "Diagnostic batches: ${DIAGNOSTIC_BATCHES}"
echo "Diagnostic epochs: ${DIAGNOSTIC_EPOCHS}"
echo "Soft time limit sec: ${SOFT_TIME_LIMIT_SEC}"
echo "Resume: ${ADAPTIVE_MTL_RESUME:-0}"
echo "Config: ${CONFIG}"
echo "Artifact dir: ${ARTIFACT_DIR}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "Notes: ${NOTES_MD}"
echo "Git commit: ${ADAPTIVE_MTL_GIT_COMMIT}"
echo "Git branch: ${ADAPTIVE_MTL_GIT_BRANCH}"

"${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py

"${PYTHON}" experiments/adaptive_multitask_tim4rec/pcgrad_train.py \
  --run-id "${RUN_ID}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --config "${CONFIG}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --result-json "${OUTPUT_JSON}" \
  --notes "${NOTES_MD}" \
  --diagnostic-detail-epochs "${DIAGNOSTIC_EPOCHS}" \
  --diagnostic-batches "${DIAGNOSTIC_BATCHES}" \
  "${RESUME_ARGS[@]}"
