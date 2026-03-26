#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_nav_env.sh"

shell_join() {
    local quoted=()
    local arg
    for arg in "$@"; do
        printf -v arg '%q' "$arg"
        quoted+=("${arg}")
    done
    local IFS=' '
    printf '%s' "${quoted[*]}"
}

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

launch_cmd_display=(
    "HEADLESS=${HEADLESS}"
    "NUM_ENVS=${NUM_ENVS}"
)

if [[ -n "${MAX_ITERATIONS:-}" ]]; then
    launch_cmd_display+=("MAX_ITERATIONS=${MAX_ITERATIONS}")
fi

if [[ -n "${RUN_NAME:-}" ]]; then
    launch_cmd_display+=("RUN_NAME=${RUN_NAME}")
fi

if [[ -n "${SEED:-}" ]]; then
    launch_cmd_display+=("SEED=${SEED}")
fi

launch_cmd_display+=("./scripts/train_nav.sh" "${TASK}")
launch_cmd_display+=("$@")

train_cmd_one_line="$(shell_join "${launch_cmd_display[@]}")"

conda_base="$(_nav_find_conda_base || true)"
launch_cmd_lines=()

if [[ -n "${conda_base}" ]]; then
    launch_cmd_lines+=("$(shell_join source "${conda_base}/etc/profile.d/conda.sh")")
fi

launch_cmd_lines+=("$(shell_join conda activate "${ENV_NAME}")")
launch_cmd_lines+=("$(shell_join cd "${ROOT_DIR}")")
launch_cmd_lines+=("${train_cmd_one_line}")

export NAV_TRAIN_LAUNCH_CMD_ONE_LINE="${train_cmd_one_line}"
export NAV_TRAIN_LAUNCH_CMD_FULL="$(printf '%s\n' "${launch_cmd_lines[@]}")"
export NAV_TRAIN_TASK="${TASK}"
export NAV_TRAIN_NUM_ENVS="${NUM_ENVS}"
export NAV_TRAIN_HEADLESS="${HEADLESS}"
export NAV_TRAIN_RUN_NAME="${RUN_NAME:-}"
export NAV_TRAIN_MAX_ITERATIONS="${MAX_ITERATIONS:-}"
export NAV_TRAIN_SEED="${SEED:-}"
export NAV_CONDA_ENV_NAME="${ENV_NAME}"
export NAV_WORKSPACE_ROOT="${ROOT_DIR}"

printf 'Launching task: %s\n' "${TASK}"
printf 'Num envs: %s\n' "${NUM_ENVS}"
if [[ "${HEADLESS}" != "0" ]]; then
    printf 'Headless: enabled\n'
else
    printf 'Headless: disabled\n'
fi

exec "${cmd[@]}"
