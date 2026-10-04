#!/usr/bin/env bash
set -euo pipefail

N_RUNS=1
METHODS=("rffrnd" )

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="${ROOT_DIR}/main.py"

declare -A BASE_CONFIG
BASE_CONFIG["rffrnd"]="${ROOT_DIR}/configs/config_RFF_MontezumaRevenge.conf"


for run in $(seq 1 "${N_RUNS}"); do
  for method in "${METHODS[@]}"; do
    conf="${BASE_CONFIG[$method]:-}"
    if [[ -z "${conf}" ]]; then
      echo "[ERROR] Unknown method: ${method}"
      exit 1
    fi
    if [[ ! -f "${conf}" ]]; then
      echo "[ERROR] Config not found: ${conf}"
      exit 1
    fi

    echo "[RUN ${run}/${N_RUNS}] ${method} (config=${conf})"
    python "${MAIN_PY}" --config="${conf}"
  done
done