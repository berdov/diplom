#!/usr/bin/env bash
#SBATCH --job-name=stage3-aux
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom/logs/slurm/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/logs/slurm/%x-%j.err

set -euo pipefail

REPO_DIR="${STAGE3_REPO_DIR:-/home/daryumin/iberdov/diplom}"
MODE="${STAGE3_MODE:-ablation}"
RUN_KEY="${STAGE3_RUN_KEY:-sanity_click}"
CONFIG="${STAGE3_CONFIG:-experiments/stage3_auxiliary_analysis/config.yaml}"
ENV_DIR="${STAGE3_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PYTHON="${STAGE3_PYTHON:-${ENV_DIR}/bin/python}"
PREP_PYTHON="${STAGE3_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
GIT_BIN="${STAGE3_GIT_BIN:-/usr/bin/git}"

cd "${REPO_DIR}"
mkdir -p logs/slurm experiments/stage3_auxiliary_analysis/runs experiments/stage3_auxiliary_analysis/artifacts

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export STAGE3_GIT_BIN="${GIT_BIN}"
if [ -x "${GIT_BIN}" ]; then
  export STAGE3_GIT_COMMIT="${STAGE3_GIT_COMMIT:-$("${GIT_BIN}" rev-parse HEAD 2>/dev/null || echo unknown)}"
  export STAGE3_GIT_BRANCH="${STAGE3_GIT_BRANCH:-$("${GIT_BIN}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
fi

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${PYTHON}" >&2
  exit 2
fi

if [ ! -x "${PREP_PYTHON}" ]; then
  echo "Missing prep Python: ${PREP_PYTHON}" >&2
  exit 2
fi

if [[ "${MODE}" == "target-audit" ]]; then
  exec "${PREP_PYTHON}" -m experiments.stage3_auxiliary_analysis.target_audit \
    --config "${CONFIG}"
fi

"${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py

exec "${PYTHON}" -m experiments.stage3_auxiliary_analysis.run \
  --config "${CONFIG}" \
  --mode ablation \
  --run-key "${RUN_KEY}"
