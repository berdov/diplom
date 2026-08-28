#!/usr/bin/env bash
#SBATCH --job-name=behavior-moe-sanity
#SBATCH --partition=test
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/home/daryumin/iberdov/diplom/experiments/behavior_moe_tim4rec/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/experiments/behavior_moe_tim4rec/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/daryumin/iberdov/diplom}"
ENV_DIR="${BEHAVIOR_MOE_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PREP_PYTHON="${BEHAVIOR_MOE_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
PYTHON="${ENV_DIR}/bin/python"
RUN_ID="${BEHAVIOR_MOE_RUN_ID:-behavior_moe_sanity_001}"
CONFIG="${BEHAVIOR_MOE_CONFIG:-${REPO_DIR}/experiments/behavior_moe_tim4rec/config.yaml}"
RUNS_DIR="${REPO_DIR}/experiments/behavior_moe_tim4rec/runs"
ARTIFACT_ROOT="${BEHAVIOR_MOE_ARTIFACT_ROOT:-/home/daryumin/iberdov/diplom/experiments/behavior_moe_tim4rec}"
EPOCHS="${BEHAVIOR_MOE_EPOCHS:-5}"
ROUTING_BATCHES="${BEHAVIOR_MOE_ROUTING_BATCHES:-5}"

cd "${REPO_DIR}"
mkdir -p "${RUNS_DIR}" "${REPO_DIR}/experiments/behavior_moe_tim4rec/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export BEHAVIOR_MOE_GIT_COMMIT="${BEHAVIOR_MOE_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export BEHAVIOR_MOE_GIT_BRANCH="${BEHAVIOR_MOE_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
export BEHAVIOR_MOE_GIT_REMOTE="${BEHAVIOR_MOE_GIT_REMOTE:-$(git config --get remote.origin.url 2>/dev/null || echo unknown)}"

if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  export BEHAVIOR_MOE_SLURM_CONSTRAINT="$(
    scontrol show job "${SLURM_JOB_ID}" | tr ' ' '\n' | awk -F= '$1=="Features"{print $2; exit}'
  )"
fi

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${ENV_DIR}" >&2
  exit 2
fi

if [ ! -x "${PREP_PYTHON}" ]; then
  echo "Missing prep Python: ${PREP_PYTHON}" >&2
  exit 2
fi

echo "Slurm: partition=${SLURM_JOB_PARTITION:-test}"
echo "Slurm: constraint=${BEHAVIOR_MOE_SLURM_CONSTRAINT:-${SLURM_JOB_CONSTRAINT:-type_e}}"
echo "Slurm: gpus=${SLURM_JOB_GPUS:-gpu:1}"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-4}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Run ID: ${RUN_ID}"
echo "Epochs: ${EPOCHS}"
echo "Routing diagnostic batches: ${ROUTING_BATCHES}"

"${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py

"${PYTHON}" experiments/behavior_moe_tim4rec/sanity_train.py \
  --config "${CONFIG}" \
  --run-id "${RUN_ID}" \
  --epochs "${EPOCHS}" \
  --artifact-dir "${ARTIFACT_ROOT}/${RUN_ID}" \
  --result-json "${RUNS_DIR}/${RUN_ID}.json" \
  --notes "${RUNS_DIR}/${RUN_ID}_notes.md" \
  --routing-csv "${RUNS_DIR}/${RUN_ID}_routing.csv" \
  --routing-diagnostic-batches "${ROUTING_BATCHES}"
