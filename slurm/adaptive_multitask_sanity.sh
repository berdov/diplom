#!/usr/bin/env bash
#SBATCH --job-name=adaptive-mtl-sanity
#SBATCH --partition=test
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/adaptive_multitask_tim4rec/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/adaptive_multitask_tim4rec/slurm_logs/%x-%j.err

set -euo pipefail

METHOD="${ADAPTIVE_MTL_METHOD:-${1:-pcgrad}}"
case "${METHOD}" in
  pcgrad)
    RUN_ID="${ADAPTIVE_MTL_RUN_ID:-pcgrad_sanity_001}"
    ;;
  metabalance)
    RUN_ID="${ADAPTIVE_MTL_RUN_ID:-metabalance_sanity_001}"
    ;;
  *)
    echo "Unsupported ADAPTIVE_MTL_METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

REPO_DIR="${REPO_DIR:-/home/daryumin/iberdov/diplom}"
ENV_DIR="${ADAPTIVE_MTL_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PREP_PYTHON="${ADAPTIVE_MTL_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
PYTHON="${ENV_DIR}/bin/python"
CONFIG="${ADAPTIVE_MTL_CONFIG:-${REPO_DIR}/experiments/adaptive_multitask_tim4rec/config.yaml}"
ARTIFACT_DIR="${ADAPTIVE_MTL_ARTIFACT_DIR:-${REPO_DIR}/experiments/adaptive_multitask_tim4rec/${RUN_ID}}"
OUTPUT_JSON="${ADAPTIVE_MTL_OUTPUT_JSON:-${REPO_DIR}/experiments/adaptive_multitask_tim4rec/runs/${RUN_ID}.json}"
NOTES_MD="${ADAPTIVE_MTL_NOTES_MD:-${REPO_DIR}/experiments/adaptive_multitask_tim4rec/runs/${RUN_ID}_notes.md}"
EPOCHS="${ADAPTIVE_MTL_EPOCHS:-5}"

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/experiments/adaptive_multitask_tim4rec/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export ADAPTIVE_MTL_GIT_COMMIT="${ADAPTIVE_MTL_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export ADAPTIVE_MTL_GIT_BRANCH="${ADAPTIVE_MTL_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
export ADAPTIVE_MTL_GIT_REMOTE="${ADAPTIVE_MTL_GIT_REMOTE:-$(git config --get remote.origin.url 2>/dev/null || echo unknown)}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${ENV_DIR}" >&2
  exit 2
fi

if [ ! -x "${PREP_PYTHON}" ]; then
  echo "Missing prep Python: ${PREP_PYTHON}" >&2
  exit 2
fi

echo "Slurm: partition=${SLURM_JOB_PARTITION:-test}"
echo "Slurm: constraint=${SLURM_JOB_CONSTRAINT:-type_e}"
echo "Slurm: gpus=${SLURM_JOB_GPUS:-gpu:1}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Repo: ${REPO_DIR}"
echo "Method: ${METHOD}"
echo "Run id: ${RUN_ID}"
echo "Epochs: ${EPOCHS}"
echo "Config: ${CONFIG}"
echo "Artifact dir: ${ARTIFACT_DIR}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "Notes: ${NOTES_MD}"
echo "Git commit: ${ADAPTIVE_MTL_GIT_COMMIT}"
echo "Git branch: ${ADAPTIVE_MTL_GIT_BRANCH}"

"${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py

"${PYTHON}" experiments/adaptive_multitask_tim4rec/sanity_train.py \
  --method "${METHOD}" \
  --run-id "${RUN_ID}" \
  --epochs "${EPOCHS}" \
  --config "${CONFIG}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --result-json "${OUTPUT_JSON}" \
  --notes "${NOTES_MD}" \
  --diagnostic-epochs "1,3,5"
