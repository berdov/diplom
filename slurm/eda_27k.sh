#!/usr/bin/env bash
#SBATCH --job-name=kuairand-27k-eda
#SBATCH --output=outputs/eda/%x-%j.out
#SBATCH --error=outputs/eda/%x-%j.err

# TODO(cHARISMa): set the cluster-specific resources before submitting.
# Examples, intentionally commented until confirmed:
#SBATCH --nodes=1
#SBATCH --ntasks=1
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>
# #SBATCH --time=<HH:MM:SS>
# #SBATCH --mem=<memory>

set -euo pipefail

PROJECT_ROOT="/home/daryumin/iberdov/diplom"
PYTHON="${PROJECT_ROOT}/.conda/bin/python"
DATA_ROOT="/home/daryumin/iberdov/Corpora"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/eda"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" src/eda_27k.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
