#!/usr/bin/env bash
#SBATCH --job-name=moo-challenger
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/daryumin/iberdov/diplom_exp_moo_challengers/experiments/moo_representative_challengers/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom_exp_moo_challengers/experiments/moo_representative_challengers/slurm_logs/%x-%j.err
set -euo pipefail

: "${REPO_DIR:?Separate challenger checkout required}"
: "${MOO_GIT_COMMIT:?Published exact commit required}"
: "${MOO_METHOD:?Method required}"
: "${MOO_STAGE:?Stage required}"
: "${MOO_PYTHON:?Verified runtime Python required}"
case "${MOO_METHOD}" in ferero|most|phn_hvi) ;; *) exit 2 ;; esac
case "${MOO_STAGE}" in smoke|sanity|convergence_screening) ;; *) exit 2 ;; esac
[[ "${REPO_DIR}" != /home/daryumin/iberdov/diplom ]]
cd "${REPO_DIR}"
# Compute nodes do not provide system Git. Load the shared Rocky build before
# both shell provenance checks and Python subprocess calls to Git.
module load Python/miniconda
module load EasyBuild/modules_rocky
module load git/2.50.1-GCCcore-14.3.0
git --version
[[ "$(git rev-parse HEAD)" == "${MOO_GIT_COMMIT}" ]]
[[ "$(git branch --show-current)" == exp/moo-representative-challengers ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XDG_CACHE_HOME="${REPO_DIR}/experiments/moo_representative_challengers/runtime_cache"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "${XDG_CACHE_HOME}"
printf 'Commit: %s\nMethod: %s\nStage: %s\nPython: %s\n' "${MOO_GIT_COMMIT}" "${MOO_METHOD}" "${MOO_STAGE}" "${MOO_PYTHON}"
# Shared validation-only data is reused. No preparation command is permitted.
exec "${MOO_PYTHON}" -m experiments.moo_representative_challengers.run --method "${MOO_METHOD}" --stage "${MOO_STAGE}"
