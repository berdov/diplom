#!/usr/bin/env bash
#SBATCH --job-name=stage3-aux
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom/logs/slurm/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom/logs/slurm/%x-%j.err

set -euo pipefail

REPO_DIR="${STAGE3_REPO_DIR:-/home/daryumin/iberdov/diplom}"
MODE="${STAGE3_MODE:-ablation}"
RUN_KEY="${STAGE3_RUN_KEY:-sanity_click}"
CONFIG="${STAGE3_CONFIG:-experiments/stage3_auxiliary_analysis/config.yaml}"
PYTHON="${STAGE3_PYTHON:-/home/daryumin/iberdov/diplom/.venv/bin/python}"
PREP_PYTHON="${STAGE3_PREP_PYTHON:-/home/daryumin/.conda/envs/py310/bin/python}"

cd "${REPO_DIR}"
mkdir -p logs/slurm experiments/stage3_auxiliary_analysis/runs experiments/stage3_auxiliary_analysis/artifacts

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

if [[ "${MODE}" == "target-audit" ]]; then
  exec "${PREP_PYTHON}" -m experiments.stage3_auxiliary_analysis.run \
    --config "${CONFIG}" \
    --mode target-audit
fi

"${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py

exec "${PYTHON}" -m experiments.stage3_auxiliary_analysis.run \
  --config "${CONFIG}" \
  --mode ablation \
  --run-key "${RUN_KEY}"
