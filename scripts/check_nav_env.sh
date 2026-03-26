#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_nav_env.sh"

setup_nav_runtime_env

"${CONDA_ENV_DIR}/bin/python" -c "import omni.client; import isaaclab; import isaaclab_rl; from isaaclab.app import AppLauncher; from tensordict import TensorDict; from rsl_rl.modules import ActorCriticSRU; print('omni.client / isaaclab / isaaclab_rl / AppLauncher / TensorDict / ActorCriticSRU import OK')"
"${CONDA_ENV_DIR}/bin/python" "${ROOT_DIR}/sru-navigation-sim/scripts/train.py" --help >/dev/null
echo "train.py entrypoint OK"
