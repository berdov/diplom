#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
STAGE="${MOO_TUNING_STAGE:-${1:-stage_a}}"
PARTITION="${MOO_TUNING_PARTITION:-rocky}"
CONSTRAINT="${MOO_TUNING_CONSTRAINT:-type_e}"
GRES="${MOO_TUNING_GRES:-gpu:a100:1}"
CPUS="${MOO_TUNING_CPUS:-4}"
MEM="${MOO_TUNING_MEM:-0}"
BUFFER_SEC="${MOO_TUNING_MIN_RUNTIME_BUFFER_SEC:-1800}"

shift $(( $# > 0 ? 1 : 0 ))
if [ "$#" -gt 0 ]; then
  METHODS=("$@")
else
  METHODS=(epo gradhv cosmos pcgrad)
fi

case "${STAGE}" in
  stage_a|stage_b|full) ;;
  *)
    echo "Unknown MOO_TUNING_STAGE=${STAGE}; expected stage_a, stage_b, or full" >&2
    exit 2
    ;;
esac

method_walltime() {
  case "$1" in
    epo) echo "${MOO_TUNING_EPO_TIME:-36:00:00}" ;;
    gradhv) echo "${MOO_TUNING_GRADHV_TIME:-24:00:00}" ;;
    cosmos) echo "${MOO_TUNING_COSMOS_TIME:-12:00:00}" ;;
    pcgrad) echo "${MOO_TUNING_PCGRAD_TIME:-24:00:00}" ;;
    *) echo "24:00:00" ;;
  esac
}

method_estimate_sec() {
  case "$1" in
    epo) echo "${MOO_TUNING_EPO_ESTIMATED_TRIAL_SEC:-28800}" ;;
    gradhv) echo "${MOO_TUNING_GRADHV_ESTIMATED_TRIAL_SEC:-9000}" ;;
    cosmos) echo "${MOO_TUNING_COSMOS_ESTIMATED_TRIAL_SEC:-3600}" ;;
    pcgrad) echo "${MOO_TUNING_PCGRAD_ESTIMATED_TRIAL_SEC:-9000}" ;;
    *) echo "" ;;
  esac
}

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/experiments/moo_8families/slurm_logs"

for method in "${METHODS[@]}"; do
  case "${method}" in
    epo|gradhv|cosmos|pcgrad) ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 2
      ;;
  esac
  walltime="$(method_walltime "${method}")"
  estimate="$(method_estimate_sec "${method}")"
  job_name="moo-${method}-${STAGE}"
  export_arg="ALL,REPO_DIR=${REPO_DIR},MOO_TUNING_METHOD=${method},MOO_TUNING_STAGE=${STAGE},MOO_TUNING_MIN_RUNTIME_BUFFER_SEC=${BUFFER_SEC}"
  if [ -n "${estimate}" ]; then
    export_arg="${export_arg},MOO_TUNING_ESTIMATED_TRIAL_RUNTIME_SEC=${estimate}"
  fi
  sbatch \
    --parsable \
    --job-name="${job_name}" \
    --partition="${PARTITION}" \
    --constraint="${CONSTRAINT}" \
    --gres="${GRES}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${walltime}" \
    --export="${export_arg}" \
    "${REPO_DIR}/slurm/moo_tuning.sh" "${method}"
done
