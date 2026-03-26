#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_nav_env.sh"

TASK="${TASK:-Isaac-Nav-PPO-Drone-Static-SwarmCompat-Play-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
HEADLESS="${HEADLESS:-0}"

setup_nav_runtime_env

if [[ $# -gt 0 && "${1}" != -* ]]; then
    TASK="${1}"
    shift
fi

cmd=(
    "${CONDA_ENV_DIR}/bin/python"
    "${ROOT_DIR}/sru-navigation-sim/scripts/play.py"
    --task
    "${TASK}"
    --num_envs
    "${NUM_ENVS}"
)

if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

if [[ -n "${CHECKPOINT:-}" ]]; then
    cmd+=(--checkpoint "${CHECKPOINT}")
fi

if [[ "${VIDEO:-0}" != "0" ]]; then
    cmd+=(--video)
fi

if [[ -n "${VIDEO_LENGTH:-}" ]]; then
    cmd+=(--video_length "${VIDEO_LENGTH}")
fi

printf "Launching play task: %s\n" "${TASK}"
printf "Num envs: %s\n" "${NUM_ENVS}"
if [[ "${HEADLESS}" != "0" ]]; then
    printf "Headless: enabled\n"
else
    printf "Headless: disabled\n"
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
    printf "Checkpoint: %s\n" "${CHECKPOINT}"
else
    printf "Checkpoint: latest from logs\n"
fi

exec "${cmd[@]}" "$@"
