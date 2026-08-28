#!/usr/bin/env bash
#SBATCH --job-name=moo8-families
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/moo_8families/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/moo_8families/slurm_logs/%x-%j.err

set -euo pipefail

METHOD="${MOO_METHOD:-${1:-stch}}"
STAGE="${MOO_STAGE:-${2:-smoke}}"
REPO_DIR="${REPO_DIR:-/home/daryumin/iberdov/diplom}"
ENV_DIR="${MOO_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PREP_PYTHON="${MOO_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
PYTHON="${MOO_PYTHON:-${ENV_DIR}/bin/python}"
CONFIG="${MOO_CONFIG:-${REPO_DIR}/experiments/moo_8families/config.yaml}"

case "${METHOD}_${STAGE}" in
  stch_smoke) DEFAULT_RUN_ID="stch_smoke_001" ;;
  famo_smoke) DEFAULT_RUN_ID="famo_smoke_001" ;;
  epo_smoke) DEFAULT_RUN_ID="epo_smoke_001" ;;
  gradhv_smoke) DEFAULT_RUN_ID="gradhv_smoke_001" ;;
  phn_smoke) DEFAULT_RUN_ID="phn_smoke_001" ;;
  cosmos_smoke) DEFAULT_RUN_ID="cosmos_smoke_001" ;;
  palora_smoke) DEFAULT_RUN_ID="palora_smoke_001" ;;
  stch_sanity) DEFAULT_RUN_ID="stch_sanity_001" ;;
  famo_sanity) DEFAULT_RUN_ID="famo_sanity_001" ;;
  epo_sanity) DEFAULT_RUN_ID="epo_sanity_001" ;;
  gradhv_sanity) DEFAULT_RUN_ID="gradhv_sanity_001" ;;
  phn_sanity) DEFAULT_RUN_ID="phn_sanity_001" ;;
  cosmos_sanity) DEFAULT_RUN_ID="cosmos_sanity_001" ;;
  palora_sanity) DEFAULT_RUN_ID="palora_sanity_001" ;;
  *)
    echo "Unsupported MOO_METHOD/MOO_STAGE: ${METHOD}/${STAGE}" >&2
    exit 2
    ;;
esac

RUN_ID="${MOO_RUN_ID:-${DEFAULT_RUN_ID}}"
ARTIFACT_DIR="${MOO_ARTIFACT_DIR:-${REPO_DIR}/experiments/moo_8families/${RUN_ID}}"
RESULT_JSON="${MOO_RESULT_JSON:-${REPO_DIR}/experiments/moo_8families/runs/${RUN_ID}.json}"
NOTES_MD="${MOO_NOTES_MD:-${REPO_DIR}/experiments/moo_8families/runs/${RUN_ID}_notes.md}"

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/experiments/moo_8families/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export MOO_GIT_COMMIT="${MOO_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export MOO_GIT_BRANCH="${MOO_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
export MOO_GIT_REMOTE="${MOO_GIT_REMOTE:-$(git config --get remote.origin.url 2>/dev/null || echo unknown)}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${PYTHON}" >&2
  exit 2
fi

if [ ! -x "${PREP_PYTHON}" ]; then
  echo "Missing prep Python: ${PREP_PYTHON}" >&2
  exit 2
fi

echo "Slurm: partition=${SLURM_JOB_PARTITION:-gpu-ef-quick}"
echo "Slurm: constraint=${SLURM_JOB_CONSTRAINT:-type_e}"
echo "Slurm: gpus=${SLURM_JOB_GPUS:-gpu:1}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Repo: ${REPO_DIR}"
echo "Method: ${METHOD}"
echo "Stage: ${STAGE}"
echo "Run id: ${RUN_ID}"
echo "Config: ${CONFIG}"
echo "Artifact dir: ${ARTIFACT_DIR}"
echo "Result JSON: ${RESULT_JSON}"
echo "Git commit: ${MOO_GIT_COMMIT}"
echo "Git branch: ${MOO_GIT_BRANCH}"

"${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py

"${PYTHON}" -m experiments.moo_8families.train \
  --method "${METHOD}" \
  --stage "${STAGE}" \
  --run-id "${RUN_ID}" \
  --config "${CONFIG}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --result-json "${RESULT_JSON}" \
  --notes "${NOTES_MD}"
