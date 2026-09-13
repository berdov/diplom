#!/usr/bin/env bash
#SBATCH --job-name=mamba3-test
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom_exp_mamba3_baseline/experiments/mamba3_baseline/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom_exp_mamba3_baseline/experiments/mamba3_baseline/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="/home/daryumin/iberdov/diplom_exp_mamba3_baseline"
ENV_DIR="/home/daryumin/iberdov/diplom/envs/mamba3"
PYTHON="${ENV_DIR}/bin/python"

RESULT="${REPO_DIR}/experiments/mamba3_baseline/runs/mamba3_final_test_001.json"

if [ -e "${RESULT}" ]; then
    echo "ABORT: final TEST JSON already exists: ${RESULT}" >&2
    exit 3
fi

if [ ! -x "${PYTHON}" ]; then
    echo "Missing Python environment: ${PYTHON}" >&2
    exit 2
fi

cd "${REPO_DIR}"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/experiments/mamba3_baseline:${REPO_DIR}:${PYTHONPATH:-}"

"${PYTHON}" experiments/mamba3_baseline/mamba3_final_test.py
