#!/usr/bin/env bash
# One invocation submits smoke -> array 0-15%4 -> afterany summary.
set -euo pipefail
: "${TC_REPO:?Isolated checkout required}" "${TC_PYTHON:?Verified Python required}"
cd "${TC_REPO}"
module load Python/miniconda
module load EasyBuild/modules_rocky
module load git/2.50.1-GCCcore-14.3.0
export TC_COMMIT="$(git rev-parse HEAD)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="${TC_REPO}"
exec "${TC_PYTHON}" -m experiments.target_combination_analysis.submit
