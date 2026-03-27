#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Train a navigation policy using RSL-RL (PPO/MDPO algorithms).

Usage:
    python scripts/train.py --task <task_name> --num_envs <num> [options]

Arguments:
    --task               Task name (required)
    --num_envs           Number of parallel environments
    --seed               Random seed
    --max_iterations     Training iterations
    --run_name           Custom run name for logging
    --video              Enable video recording
    --video_length       Video length in steps (default: 200)
    --video_interval     Recording interval in steps (default: 2000)

Examples:
    python scripts/train.py --task Isaac-Navigation-B2W-v0 --num_envs 2048
    python scripts/train.py --task Isaac-Navigation-B2W-v0 --video --seed 42

Logs saved to: logs/rsl_rl/<experiment_name>/<timestamp>/
"""

from __future__ import annotations

import argparse
import pickle
import shlex
import subprocess
import sys

# Add the parent directory to the path so we can import from the extension
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train a navigation policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--experiment_name", type=str, default=None, help="Override experiment folder name.")
parser.add_argument("--run_name", type=str, default=None, help="Custom run name suffix appended to the log directory.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run directory name to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file name or absolute path.")
parser.add_argument(
    "--depth_include_teammates",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="For swarm tasks, choose whether depth ray-casting includes teammate collision boxes.",
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching simulation
import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import Isaac Lab extensions
import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_nav_task.navigation.utils.ippo_rslrl_wrapper import RslRlParameterSharingVecEnvWrapper

# Set torch backends for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class TeeStream:
    """Mirror writes to multiple streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def dump_pickle_file(filename: str, data):
    """Persist configs for later replay/debugging on IsaacLab versions without dump_pickle."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def run_text_command(command: list[str], cwd: str | None = None) -> str | None:
    """Run a subprocess and return stripped stdout on success."""
    try:
        return subprocess.check_output(command, cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def get_workspace_root() -> str:
    """Resolve the workspace root from this script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_root = os.path.dirname(script_dir)
    return os.path.dirname(sim_root)


def get_git_context(repo_root: str) -> dict[str, str]:
    """Collect git branch / commit information for the current workspace."""
    branch = run_text_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root) or "unknown"
    commit = run_text_command(["git", "rev-parse", "HEAD"], cwd=repo_root) or "unknown"
    status = run_text_command(["git", "status", "--short"], cwd=repo_root) or ""
    return {
        "branch": branch,
        "commit": commit,
        "dirty": "yes" if bool(status) else "no",
    }


def infer_play_task(train_task: str) -> str | None:
    """Infer the matching play task from the training task id."""
    if train_task.endswith("-Play-v0") or train_task.endswith("-PlayFast-v0"):
        return train_task
    if train_task.endswith("-Dev-v0"):
        return train_task[: -len("-Dev-v0")] + "-Play-v0"
    if train_task.endswith("-v0"):
        return train_task[: -len("-v0")] + "-Play-v0"
    return None


def build_fallback_train_command(args_cli, workspace_root: str) -> str:
    """Build a reproducible training command when the shell launcher did not provide one."""
    command_tokens = []

    headless_value = os.environ.get("NAV_TRAIN_HEADLESS")
    if headless_value is None:
        headless_value = "1" if getattr(args_cli, "headless", False) else "0"
    command_tokens.append(f"HEADLESS={headless_value}")

    if args_cli.num_envs is not None:
        command_tokens.append(f"NUM_ENVS={args_cli.num_envs}")
    if args_cli.max_iterations is not None:
        command_tokens.append(f"MAX_ITERATIONS={args_cli.max_iterations}")
    if args_cli.run_name is not None:
        command_tokens.append(f"RUN_NAME={args_cli.run_name}")
    if args_cli.seed is not None:
        command_tokens.append(f"SEED={args_cli.seed}")

    command_tokens.extend(["./scripts/train_nav.sh", args_cli.task])
    command_tokens.extend(hydra_args)

    conda_base = run_text_command(["conda", "info", "--base"])
    conda_env_name = os.environ.get("NAV_CONDA_ENV_NAME") or os.environ.get("CONDA_DEFAULT_ENV") or "env_isaacsim"

    lines = []
    if conda_base:
        lines.append(shlex.join(["source", os.path.join(conda_base, "etc", "profile.d", "conda.sh")]))
    lines.append(shlex.join(["conda", "activate", conda_env_name]))
    lines.append(shlex.join(["cd", workspace_root]))
    lines.append(shlex.join(command_tokens))
    return "\n".join(lines)


def build_play_command_snippet(
    workspace_root: str, play_task: str | None, log_dir: str | None
) -> str | None:
    """Build a reproducible play command for the current training run."""
    if play_task is None or log_dir is None:
        return None

    conda_base = run_text_command(["conda", "info", "--base"])
    conda_env_name = os.environ.get("NAV_CONDA_ENV_NAME") or os.environ.get("CONDA_DEFAULT_ENV") or "env_isaacsim"
    log_dir_rel = os.path.relpath(log_dir, workspace_root)

    lines = []
    if conda_base:
        lines.append(shlex.join(["source", os.path.join(conda_base, "etc", "profile.d", "conda.sh")]))
    lines.append(shlex.join(["conda", "activate", conda_env_name]))
    lines.append(shlex.join(["cd", workspace_root]))
    lines.append(
        (
            f'CHECKPOINT="$(find {shlex.quote(log_dir_rel)} -maxdepth 1 -type f -name '
            f'{shlex.quote("model_*.pt")} | sort | tail -n 1)" '
            f'HEADLESS=0 NUM_ENVS=1 ./scripts/play_nav.sh {shlex.quote(play_task)}'
        )
    )
    return "\n".join(lines)


def write_repro_log(log_dir: str, args_cli, agent_cfg) -> None:
    """Persist git context and reproducible commands alongside the training logs."""
    workspace_root = os.environ.get("NAV_WORKSPACE_ROOT") or get_workspace_root()
    git_context = get_git_context(workspace_root)
    play_task = infer_play_task(args_cli.task)

    train_command = os.environ.get("NAV_TRAIN_LAUNCH_CMD_FULL") or build_fallback_train_command(args_cli, workspace_root)
    play_command = build_play_command_snippet(workspace_root, play_task, log_dir)

    repro_log_path = os.path.join(log_dir, "repro_commands.sh")
    log_dir_rel = os.path.relpath(log_dir, workspace_root)

    lines = [
        "#!/usr/bin/env bash",
        f"# Workspace root: {workspace_root}",
        f"# Git branch: {git_context['branch']}",
        f"# Git commit: {git_context['commit']}",
        f"# Git dirty: {git_context['dirty']}",
        f"# Train task: {args_cli.task}",
        f"# Play task: {play_task or 'unavailable'}",
        f"# Log dir: {log_dir_rel}",
        "# Play checkpoint strategy: latest model_*.pt under this run directory",
        "",
        "# Training command",
        train_command,
        "",
    ]

    if play_command is not None:
        lines.extend(
            [
                "# Play command",
                play_command,
                "",
            ]
        )

    with open(repro_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.chmod(repro_log_path, 0o755)

    print(f"[INFO] Workspace git branch: {git_context['branch']}")
    print(f"[INFO] Workspace git commit: {git_context['commit']}")
    print(f"[INFO] Workspace git dirty: {git_context['dirty']}")
    print(f"[INFO] Repro command log written to: {repro_log_path}")
    print("[INFO] Training command for this run:")
    print(train_command)
    if play_command is not None:
        print("[INFO] Suggested play command for this run:")
        print(play_command)


def main():
    """Train navigation policy with RSL-RL."""
    # Load the configurations from the registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # Override config from command line
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.depth_include_teammates is not None and hasattr(env_cfg, "depth_include_teammates"):
        env_cfg.depth_include_teammates = args_cli.depth_include_teammates
        if hasattr(env_cfg, "apply_depth_raycast_mode"):
            env_cfg.apply_depth_raycast_mode()

    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Wrap the environment
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = RslRlParameterSharingVecEnvWrapper(env)
    else:
        env = RslRlVecEnvWrapper(env)

    # Specify log directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    resume_path = None
    if agent_cfg.resume:
        if os.path.isfile(agent_cfg.load_checkpoint):
            resume_path = os.path.abspath(agent_cfg.load_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    # Specify run directory based on timestamp
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    console_log_path = os.path.join(log_dir, "console.log")
    console_log_file = open(console_log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, console_log_file)
    sys.stderr = TeeStream(sys.__stderr__, console_log_file)
    print(f"[INFO] Writing console log to: {console_log_path}")
    write_repro_log(log_dir, args_cli, agent_cfg)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # Write git state to log
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
    # Save configuration
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle_file(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle_file(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation
    simulation_app.close()
