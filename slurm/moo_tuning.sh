#!/usr/bin/env bash
#SBATCH --job-name=moo-tuning
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=/home/daryumin/iberdov/diplom_exp_moo_8families/experiments/moo_8families/slurm_logs/%x-%j.out
#SBATCH --error=/home/daryumin/iberdov/diplom_exp_moo_8families/experiments/moo_8families/slurm_logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"
ENV_DIR="${MOO_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PREP_PYTHON="${MOO_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
PYTHON="${MOO_PYTHON:-${ENV_DIR}/bin/python}"
SPACES="${MOO_TUNING_SPACES:-${REPO_DIR}/configs/moo_tuning_spaces.yaml}"
METHOD="${MOO_TUNING_METHOD:-${1:-epo}}"
TARGET_COMPLETE="${MOO_TUNING_TARGET_COMPLETE:-}"
N_TRIALS="${MOO_TUNING_N_TRIALS:-}"
SUMMARY_ONLY="${MOO_TUNING_SUMMARY_ONLY:-0}"
ALLOW_DIRTY="${MOO_TUNING_ALLOW_DIRTY:-0}"

case "${METHOD}" in
  epo|gradhv|cosmos|pcgrad|all) ;;
  *)
    echo "Unknown MOO_TUNING_METHOD=${METHOD}; expected epo, gradhv, cosmos, pcgrad, or all" >&2
    exit 2
    ;;
esac

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/experiments/moo_8families/slurm_logs"

module load Python/miniconda || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export MOO_GIT_COMMIT="${MOO_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export MOO_GIT_BRANCH="${MOO_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
export MOO_GIT_REMOTE="${MOO_GIT_REMOTE:-$(git config --get remote.origin.url 2>/dev/null || echo unknown)}"
export MOO_TUNING_PRINT_COMMANDS="${MOO_TUNING_PRINT_COMMANDS:-1}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${PYTHON}" >&2
  exit 2
fi

if [ ! -x "${PREP_PYTHON}" ]; then
  echo "Missing prep Python: ${PREP_PYTHON}" >&2
  exit 2
fi

if [ "${ALLOW_DIRTY}" != "1" ] && [ -n "$(git status --short --untracked-files=no)" ]; then
  echo "Tracked files are dirty; set MOO_TUNING_ALLOW_DIRTY=1 only for an intentional resume." >&2
  git status --short --untracked-files=no >&2
  exit 2
fi

echo "Slurm: partition=${SLURM_JOB_PARTITION:-rocky}"
echo "Slurm: constraint=${SLURM_JOB_CONSTRAINT:-type_e}"
echo "Slurm: gpus=${SLURM_JOB_GPUS:-gpu:a100:1}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Repo: ${REPO_DIR}"
echo "Method: ${METHOD}"
echo "Spaces: ${SPACES}"
echo "Git commit: ${MOO_GIT_COMMIT}"
echo "Git branch: ${MOO_GIT_BRANCH}"

"${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py

"${PYTHON}" -m py_compile \
  experiments/moo_8families/train.py \
  experiments/moo_8families/tune.py \
  experiments/moo_8families/build_tuning_results.py

cmd=(
  "${PYTHON}" -m experiments.moo_8families.tune
  --spaces "${SPACES}"
  --method "${METHOD}"
)

if [ -n "${TARGET_COMPLETE}" ]; then
  cmd+=(--target-complete "${TARGET_COMPLETE}")
fi

if [ -n "${N_TRIALS}" ]; then
  cmd+=(--n-trials "${N_TRIALS}")
fi

if [ "${SUMMARY_ONLY}" = "1" ]; then
  cmd+=(--summary-only)
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
