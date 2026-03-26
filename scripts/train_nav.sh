#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_nav_env.sh"

TASK="${TASK:-Isaac-Nav-PPO-B2W-Dev-v0}"
NUM_ENVS="${NUM_ENVS:-32}"
HEADLESS="${HEADLESS:-1}"

setup_nav_runtime_env

if [[ $# -gt 0 && "${1}" != -* ]]; then
    TASK="${1}"
    shift
fi

cmd=(
    "${CONDA_ENV_DIR}/bin/python"
    "${ROOT_DIR}/sru-navigation-sim/scripts/train.py"
    --task
    "${TASK}"
    --num_envs
    "${NUM_ENVS}"
)

if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

if [[ -n "${RUN_NAME:-}" ]]; then
    cmd+=(--run_name "${RUN_NAME}")
fi

if [[ -n "${MAX_ITERATIONS:-}" ]]; then
    cmd+=(--max_iterations "${MAX_ITERATIONS}")
fi

if [[ -n "${SEED:-}" ]]; then
    cmd+=(--seed "${SEED}")
fi

cmd+=("$@")

printf 'Launching task: %s\n' "${TASK}"
printf 'Num envs: %s\n' "${NUM_ENVS}"
if [[ "${HEADLESS}" != "0" ]]; then
    printf 'Headless: enabled\n'
else
    printf 'Headless: disabled\n'
fi

exec "${cmd[@]}"
