#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_nav_env.sh"

PIP_BIN="${PIP_BIN:-${CONDA_ENV_DIR}/bin/pip}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_DIR}/bin/python}"

if [[ ! -x "${PIP_BIN}" ]]; then
    echo "Missing pip binary: ${PIP_BIN}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing python binary: ${PYTHON_BIN}" >&2
    exit 1
fi

check_nav_runtime_prereqs

echo "[1/5] Uninstalling preinstalled rsl-rl-lib from ${ENV_NAME} if present"
"${PIP_BIN}" uninstall -y rsl-rl-lib >/dev/null 2>&1 || true

echo "[2/5] Removing stale Isaac Sim rsl_rl package if present"
shopt -s nullglob
stale_paths=()
if [[ -n "${ISAACSIM_DIR}" ]]; then
    stale_paths=("${ISAACSIM_DIR}"/kit/python/lib/python*/site-packages/rsl_rl)
fi
if (( ${#stale_paths[@]} > 0 )); then
    rm -rf "${stale_paths[@]}"
else
    echo "No stale bundled rsl_rl package found"
fi
shopt -u nullglob

echo "[3/5] Installing SRU-enhanced rsl_rl"
"${PIP_BIN}" install -e "${ROOT_DIR}/sru-navigation-learning"

echo "[4/5] Installing isaaclab_nav_task"
"${PIP_BIN}" install -e "${ROOT_DIR}/sru-navigation-sim"

echo "[5/5] Verifying imports"
"${SCRIPT_DIR}/check_nav_env.sh"
