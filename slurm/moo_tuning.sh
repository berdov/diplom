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
DEFAULT_REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"
ENV_DIR="${MOO_ENV_DIR:-/home/daryumin/iberdov/diplom/envs/tim4rec}"
PREP_PYTHON="${MOO_PREP_PYTHON:-/home/daryumin/iberdov/diplom/.conda/bin/python}"
PYTHON="${MOO_PYTHON:-${ENV_DIR}/bin/python}"
SPACES="${MOO_TUNING_SPACES:-${REPO_DIR}/configs/moo_tuning_spaces.yaml}"
GIT_BIN="${MOO_GIT_BIN:-/usr/bin/git}"
VALIDATION_ONLY_ROOT="${MOO_VALIDATION_ONLY_ROOT:-/home/daryumin/iberdov/diplom/experiments/multitask_tim4rec_optuna/validation_only_recbole}"
VALIDATION_ONLY_DATASET="${MOO_VALIDATION_ONLY_DATASET:-kuairand_multitask_validonly}"
VALIDATION_SUMMARY="${MOO_VALIDATION_ONLY_SUMMARY:-${VALIDATION_ONLY_ROOT}/validation_only_dataset.json}"
METHOD="${MOO_TUNING_METHOD:-${1:-epo}}"
TUNING_STAGE="${MOO_TUNING_STAGE:-stage_a}"
TARGET_COMPLETE="${MOO_TUNING_TARGET_COMPLETE:-}"
N_TRIALS="${MOO_TUNING_N_TRIALS:-}"
STATUS_ONLY="${MOO_TUNING_STATUS_ONLY:-0}"
SUMMARY_ONLY="${MOO_TUNING_SUMMARY_ONLY:-0}"
ALLOW_DIRTY="${MOO_TUNING_ALLOW_DIRTY:-0}"
PREPARE_VALIDATION="${MOO_TUNING_PREPARE_VALIDATION:-0}"
MIN_RUNTIME_BUFFER_SEC="${MOO_TUNING_MIN_RUNTIME_BUFFER_SEC:-1800}"
ESTIMATED_TRIAL_RUNTIME_SEC="${MOO_TUNING_ESTIMATED_TRIAL_RUNTIME_SEC:-}"
MAX_WORKER_RUNTIME_SEC="${MOO_TUNING_MAX_WORKER_RUNTIME_SEC:-}"

timelimit_to_seconds() {
  local value="$1"
  if [ -z "${value}" ] || [ "${value}" = "UNLIMITED" ] || [ "${value}" = "NOT_SET" ]; then
    echo ""
    return
  fi
  if [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo $((value * 60))
    return
  fi
  local days=0 rest="${value}"
  if [[ "${rest}" == *-* ]]; then
    days="${rest%%-*}"
    rest="${rest#*-}"
  fi
  IFS=: read -r a b c <<< "${rest}"
  if [ -n "${c:-}" ]; then
    echo $((days * 86400 + 10#${a} * 3600 + 10#${b} * 60 + 10#${c}))
  else
    echo $((days * 86400 + 10#${a} * 60 + 10#${b}))
  fi
}

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
if [ ! -x "${GIT_BIN}" ]; then
  GIT_BIN="$(command -v git || true)"
fi
if [ -n "${GIT_BIN}" ]; then
  export MOO_GIT_COMMIT="${MOO_GIT_COMMIT:-$("${GIT_BIN}" rev-parse HEAD 2>/dev/null || echo unknown)}"
  export MOO_GIT_BRANCH="${MOO_GIT_BRANCH:-$("${GIT_BIN}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
  export MOO_GIT_REMOTE="${MOO_GIT_REMOTE:-$("${GIT_BIN}" config --get remote.origin.url 2>/dev/null || echo unknown)}"
else
  export MOO_GIT_COMMIT="${MOO_GIT_COMMIT:-unknown}"
  export MOO_GIT_BRANCH="${MOO_GIT_BRANCH:-unknown}"
  export MOO_GIT_REMOTE="${MOO_GIT_REMOTE:-unknown}"
fi
export MOO_TUNING_PRINT_COMMANDS="${MOO_TUNING_PRINT_COMMANDS:-1}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing TiM4Rec env: ${PYTHON}" >&2
  exit 2
fi

if [ ! -x "${PREP_PYTHON}" ]; then
  echo "Missing prep Python: ${PREP_PYTHON}" >&2
  exit 2
fi

if [ "${ALLOW_DIRTY}" != "1" ] && [ -n "${GIT_BIN}" ] && [ -n "$("${GIT_BIN}" status --short --untracked-files=no)" ]; then
  echo "Tracked files are dirty; set MOO_TUNING_ALLOW_DIRTY=1 only for an intentional resume." >&2
  "${GIT_BIN}" status --short --untracked-files=no >&2
  exit 2
fi

echo "Slurm: partition=${SLURM_JOB_PARTITION:-rocky}"
echo "Slurm: constraint=${SLURM_JOB_CONSTRAINT:-type_e}"
echo "Slurm: gpus=${SLURM_JOB_GPUS:-gpu:a100:1}"
echo "Slurm: node list=${SLURM_JOB_NODELIST:-unknown}"
echo "Repo: ${REPO_DIR}"
echo "Method: ${METHOD}"
echo "Tuning stage: ${TUNING_STAGE}"
echo "Spaces: ${SPACES}"
echo "Git commit: ${MOO_GIT_COMMIT}"
echo "Git branch: ${MOO_GIT_BRANCH}"
echo "Prepare validation-only data: ${PREPARE_VALIDATION}"
echo "Validation-only summary: ${VALIDATION_SUMMARY}"

if [ -z "${MAX_WORKER_RUNTIME_SEC}" ]; then
  MAX_WORKER_RUNTIME_SEC="$(timelimit_to_seconds "${SLURM_TIMELIMIT:-}")"
fi
echo "Worker max runtime sec: ${MAX_WORKER_RUNTIME_SEC:-unknown}"
echo "Min runtime buffer sec: ${MIN_RUNTIME_BUFFER_SEC}"
echo "Estimated trial runtime sec: ${ESTIMATED_TRIAL_RUNTIME_SEC:-auto}"

if [ "${STATUS_ONLY}" = "1" ]; then
  PREPARE_VALIDATION=0
fi

if [ "${PREPARE_VALIDATION}" = "1" ]; then
  "${PREP_PYTHON}" experiments/multitask_tim4rec_optuna/prepare_validation_only.py
elif [ "${STATUS_ONLY}" != "1" ] && [ "${SUMMARY_ONLY}" != "1" ]; then
  if [ ! -s "${VALIDATION_SUMMARY}" ]; then
    echo "Missing validation-only dataset summary; run prepare_validation_only.py once before submitting tuning jobs." >&2
    exit 2
  fi
  for suffix in train.inter valid.inter item; do
    file="${VALIDATION_ONLY_ROOT}/${VALIDATION_ONLY_DATASET}/${VALIDATION_ONLY_DATASET}.${suffix}"
    if [ ! -s "${file}" ]; then
      echo "Missing or empty validation-only RecBole file: ${file}" >&2
      exit 2
    fi
  done
fi

"${PYTHON}" -m py_compile \
  experiments/moo_8families/train.py \
  experiments/moo_8families/tune.py \
  experiments/moo_8families/build_tuning_results.py

cmd=(
  "${PYTHON}" -m experiments.moo_8families.tune
  --spaces "${SPACES}"
  --method "${METHOD}"
  --tuning-stage "${TUNING_STAGE}"
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

if [ "${STATUS_ONLY}" = "1" ]; then
  cmd+=(--status)
fi

if [ -n "${MAX_WORKER_RUNTIME_SEC}" ]; then
  cmd+=(--max-worker-runtime-sec "${MAX_WORKER_RUNTIME_SEC}")
fi

if [ -n "${MIN_RUNTIME_BUFFER_SEC}" ]; then
  cmd+=(--min-runtime-buffer-sec "${MIN_RUNTIME_BUFFER_SEC}")
fi

if [ -n "${ESTIMATED_TRIAL_RUNTIME_SEC}" ]; then
  cmd+=(--estimated-trial-runtime-sec "${ESTIMATED_TRIAL_RUNTIME_SEC}")
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
