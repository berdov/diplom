#!/usr/bin/env bash
#SBATCH --job-name=epo-moe
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/epo_moe/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/epo_moe/slurm_logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"
ENV_DIR="${EPO_MOE_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PYTHON="${EPO_MOE_PYTHON:-${ENV_DIR}/bin/python}"
PREP_PYTHON="${EPO_MOE_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
GIT_BIN="${EPO_MOE_GIT_BIN:-/usr/bin/git}"
CONFIG="${EPO_MOE_CONFIG:-${REPO_DIR}/experiments/epo_moe/configs/epo_moe.yaml}"
RUN_KEY="${EPO_MOE_RUN_KEY:-${1:-m2}}"
STAGE="${EPO_MOE_STAGE:-${2:-validation}}"
RUN_ID="${EPO_MOE_RUN_ID:-}"
RESUME="${EPO_MOE_RESUME:-0}"
ALLOW_OVERWRITE="${EPO_MOE_ALLOW_OVERWRITE:-0}"
VALIDATION_ONLY_ROOT="${EPO_MOE_VALIDATION_ONLY_ROOT:-/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/validation_only_recbole}"
VALIDATION_ONLY_DATASET="${EPO_MOE_VALIDATION_ONLY_DATASET:-kuairand_multitask_validonly}"
VALIDATION_SUMMARY="${EPO_MOE_VALIDATION_ONLY_SUMMARY:-${VALIDATION_ONLY_ROOT}/validation_only_dataset.json}"

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/experiments/epo_moe/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
if [ ! -x "${GIT_BIN}" ]; then
  GIT_BIN="$(command -v git || true)"
fi
if [ -n "${GIT_BIN}" ]; then
  export EPO_MOE_GIT_COMMIT="${EPO_MOE_GIT_COMMIT:-$("${GIT_BIN}" rev-parse HEAD 2>/dev/null || echo unknown)}"
  export EPO_MOE_GIT_BRANCH="${EPO_MOE_GIT_BRANCH:-$("${GIT_BIN}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
  export EPO_MOE_GIT_REMOTE="${EPO_MOE_GIT_REMOTE:-$("${GIT_BIN}" config --get remote.origin.url 2>/dev/null || echo unknown)}"
else
  export EPO_MOE_GIT_COMMIT="${EPO_MOE_GIT_COMMIT:-unknown}"
  export EPO_MOE_GIT_BRANCH="${EPO_MOE_GIT_BRANCH:-unknown}"
  export EPO_MOE_GIT_REMOTE="${EPO_MOE_GIT_REMOTE:-unknown}"
fi

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${PYTHON}" >&2
  exit 2
fi

echo "Slurm: partition=${SLURM_JOB_PARTITION:-rocky}"
echo "Slurm: constraint=${SLURM_JOB_CONSTRAINT:-type_e}"
echo "Slurm: gpus=${SLURM_JOB_GPUS:-gpu:a100:1}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Repo: ${REPO_DIR}"
echo "Stage: ${STAGE}"
echo "Run key: ${RUN_KEY}"
echo "Config: ${CONFIG}"
echo "Git commit: ${EPO_MOE_GIT_COMMIT}"
echo "Git branch: ${EPO_MOE_GIT_BRANCH}"

if [ "${STAGE}" != "final_test" ]; then
  if [ ! -s "${VALIDATION_SUMMARY}" ]; then
    if [ ! -x "${PREP_PYTHON}" ]; then
      echo "Missing prep Python: ${PREP_PYTHON}" >&2
      exit 2
    fi
    "${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py
  fi
  for suffix in train.inter valid.inter item; do
    file="${VALIDATION_ONLY_ROOT}/${VALIDATION_ONLY_DATASET}/${VALIDATION_ONLY_DATASET}.${suffix}"
    if [ ! -s "${file}" ]; then
      echo "Missing or empty validation-only RecBole file: ${file}" >&2
      exit 2
    fi
  done
fi

"${PYTHON}" -m py_compile \
  experiments/epo_moe/model.py \
  experiments/epo_moe/run.py \
  experiments/epo_moe/summarize.py

cmd=(
  "${PYTHON}" -m experiments.epo_moe.run
  --config "${CONFIG}"
  --run-key "${RUN_KEY}"
  --stage "${STAGE}"
)

if [ -n "${RUN_ID}" ]; then
  cmd+=(--run-id "${RUN_ID}")
fi
if [ "${RESUME}" = "1" ]; then
  cmd+=(--resume)
fi
if [ "${ALLOW_OVERWRITE}" = "1" ]; then
  cmd+=(--allow-overwrite)
fi
if [ -n "${EPO_MOE_EPOCHS:-}" ]; then
  cmd+=(--epochs "${EPO_MOE_EPOCHS}")
fi
if [ -n "${EPO_MOE_MAX_BATCHES:-}" ]; then
  cmd+=(--max-batches "${EPO_MOE_MAX_BATCHES}")
fi
if [ -n "${EPO_MOE_FINAL_TEST_CHECKPOINT_JSON:-}" ]; then
  cmd+=(--final-test-checkpoint-json "${EPO_MOE_FINAL_TEST_CHECKPOINT_JSON}")
fi
if [ -n "${EPO_MOE_FINAL_TEST_RUN_JSON:-}" ]; then
  cmd+=(--final-test-run-json "${EPO_MOE_FINAL_TEST_RUN_JSON}")
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
