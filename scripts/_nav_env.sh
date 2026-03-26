#!/usr/bin/env bash

# Shared runtime environment for launching SRU navigation from this workspace.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_PARENT_DIR="$(dirname "${ROOT_DIR}")"
ENV_NAME="${ENV_NAME:-${CONDA_DEFAULT_ENV:-env_isaacsim}}"

_nav_find_conda_base() {
    if command -v conda >/dev/null 2>&1; then
        conda info --base 2>/dev/null
    fi
}

_nav_resolve_conda_env_dir() {
    local conda_base=""

    if [[ -n "${CONDA_ENV_DIR:-}" ]]; then
        printf '%s\n' "${CONDA_ENV_DIR}"
        return 0
    fi
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        printf '%s\n' "${CONDA_PREFIX}"
        return 0
    fi

    conda_base="$(_nav_find_conda_base)"
    if [[ -n "${conda_base}" ]]; then
        printf '%s\n' "${conda_base}/envs/${ENV_NAME}"
        return 0
    fi

    return 1
}

_nav_resolve_isaaclab_dir() {
    local candidate

    if [[ -n "${ISAACLAB_DIR:-}" ]]; then
        printf '%s\n' "${ISAACLAB_DIR}"
        return 0
    fi

    for candidate in \
        "${ROOT_PARENT_DIR}"/IsaacLab* \
        "${ROOT_PARENT_DIR}"/isaaclab*; do
        if [[ -d "${candidate}" && -f "${candidate}/apps/isaaclab.python.headless.kit" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

_nav_resolve_isaacsim_dir() {
    local candidate

    if [[ -n "${ISAACSIM_DIR:-}" ]]; then
        printf '%s\n' "${ISAACSIM_DIR}"
        return 0
    fi
    if [[ -n "${ISAACSIM_PATH:-}" ]]; then
        printf '%s\n' "${ISAACSIM_PATH}"
        return 0
    fi

    for candidate in \
        "${ISAACLAB_DIR:-}/_isaac_sim" \
        "${ISAACLAB_DIR:-}" \
        "${ROOT_PARENT_DIR}/isaacsim" \
        "${ROOT_PARENT_DIR}/_isaac_sim" \
        "${HOME}/isaacsim"; do
        if [[ -n "${candidate}" && -f "${candidate}/setup_conda_env.sh" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

CONDA_ENV_DIR="$(_nav_resolve_conda_env_dir || true)"
ISAACLAB_DIR="$(_nav_resolve_isaaclab_dir || true)"
ISAACSIM_DIR="$(_nav_resolve_isaacsim_dir || true)"

RUNTIME_CACHE_DIR="${RUNTIME_CACHE_DIR:-${ROOT_DIR}/.runtime-cache}"

check_nav_runtime_prereqs() {
    if [[ -z "${CONDA_ENV_DIR}" || ! -x "${CONDA_ENV_DIR}/bin/python" ]]; then
        echo "Missing conda env python: ${CONDA_ENV_DIR:-<unresolved>}/bin/python" >&2
        echo "Activate the target conda environment first, or set CONDA_ENV_DIR/ENV_NAME." >&2
        return 1
    fi
}

setup_nav_runtime_env() {
    check_nav_runtime_prereqs

    mkdir -p "${RUNTIME_CACHE_DIR}/warp" "${RUNTIME_CACHE_DIR}/xdg"

    export CONDA_PREFIX="${CONDA_ENV_DIR}"
    export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
    export PYTHONPATH="${ROOT_DIR}/sru-navigation-learning:${ROOT_DIR}/sru-navigation-sim${PYTHONPATH:+:${PYTHONPATH}}"
    export WARP_CACHE_DIR="${RUNTIME_CACHE_DIR}/warp"
    export XDG_CACHE_HOME="${RUNTIME_CACHE_DIR}/xdg"
    export ISAACLAB_DIR
    export ISAACSIM_DIR

    if [[ -n "${ISAACSIM_DIR}" && -f "${ISAACSIM_DIR}/setup_conda_env.sh" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${ISAACSIM_DIR}/setup_conda_env.sh"
        set -u
    fi

    # If Isaac Sim is installed outside the conda env, add omni.client explicitly.
    if [[ -n "${ISAACSIM_DIR}" && -d "${ISAACSIM_DIR}/kit/extscore/omni.client.lib" ]]; then
        export PYTHONPATH="${ISAACSIM_DIR}/kit/extscore/omni.client.lib${PYTHONPATH:+:${PYTHONPATH}}"
    fi
}
