#!/usr/bin/env bash
#SBATCH --job-name=kuairand-protocol-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-e-quick
#SBATCH --constraint=type_e
#SBATCH --output=outputs/data/logs/%x-%j.out
#SBATCH --error=outputs/data/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="/home/daryumin/iberdov/diplom"
PYTHON="${PROJECT_ROOT}/.conda/bin/python"
DATA_ROOT="/home/daryumin/iberdov/Corpora"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed/protocol_b"
REPO_OUTPUT_DIR="${PROJECT_ROOT}/outputs/data"
REPORT_PATH="${PROJECT_ROOT}/reports/kuairand_protocol_b_data_report.md"
SANITY_ARGS=()

if [[ -n "${PROTOCOL_B_SANITY_LIMIT:-}" ]]; then
  OUTPUT_DIR="${PROJECT_ROOT}/data/processed/protocol_b_sanity"
  REPO_OUTPUT_DIR="${PROJECT_ROOT}/outputs/data_sanity"
  REPORT_PATH="${PROJECT_ROOT}/reports/kuairand_protocol_b_data_report_sanity.md"
  SANITY_ARGS=(--sanity-limit "${PROTOCOL_B_SANITY_LIMIT}")
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

set +u
source /etc/profile >/dev/null 2>&1 || true
module load python/miniconda >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${PROJECT_ROOT}/.conda" || true
fi
set -u

cd "${PROJECT_ROOT}"
mkdir -p "${REPO_OUTPUT_DIR}/logs" "${OUTPUT_DIR}"

GIT_COMMIT="${PROTOCOL_B_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"

echo "Slurm: partition=${SLURM_JOB_PARTITION:-cpu-e-quick}"
echo "Slurm: constraint=type_e"
echo "Slurm: gres=none"
echo "Slurm: cpus=${SLURM_CPUS_PER_TASK:-8}"
echo "Slurm: mem=0"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Commit кода препроцессинга: ${GIT_COMMIT}"
echo "Sanity limit: ${PROTOCOL_B_SANITY_LIMIT:-none}"

"${PYTHON}" --version
"${PYTHON}" -m py_compile src/prepare_kuairand_protocol_b.py
"${PYTHON}" src/prepare_kuairand_protocol_b.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --repo-output-dir "${REPO_OUTPUT_DIR}" \
  --report-path "${REPORT_PATH}" \
  --git-commit "${GIT_COMMIT}" \
  "${SANITY_ARGS[@]}"
