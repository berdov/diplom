#!/usr/bin/env bash
#SBATCH --job-name=kuairand-27k-eda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --partition=test
#SBATCH --constraint=type_a
#SBATCH --gres=gpu:v100:1
#SBATCH --output=outputs/eda/logs/%x-%j.out
#SBATCH --error=outputs/eda/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="/home/daryumin/iberdov/diplom"
PYTHON="${PROJECT_ROOT}/.conda/bin/python"
DATA_ROOT="/home/daryumin/iberdov/Corpora"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/eda"

export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

set +u
source /etc/profile >/dev/null 2>&1 || true
module load python/miniconda >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${PROJECT_ROOT}/.conda" || true
fi
set -u

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}/logs"

echo "Slurm config: partition=${SLURM_JOB_PARTITION:-test}"
echo "Slurm config: constraint=type_a"
echo "Slurm config: gres=gpu:v100:1"
echo "Slurm config: cpus=${SLURM_CPUS_PER_TASK:-10}"
echo "Slurm config: mem=0 (cluster policy exposes node memory through this setting)"
echo "Slurm node list: ${SLURM_JOB_NODELIST:-unknown}"

"${PYTHON}" --version
"${PYTHON}" -m compileall src

"${PYTHON}" src/eda_27k.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
