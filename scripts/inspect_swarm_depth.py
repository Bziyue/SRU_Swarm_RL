#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect swarm depth images with and without teammate meshes.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Nav-IPPO-Drone-Swarm-Static-Dev-v0",
    help="Swarm task used to generate the scene and spawn/goal sample.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create for inspection.")
parser.add_argument("--seed", type=int, default=7, help="Torch random seed used before env reset.")
parser.add_argument(
    "--include_teammates",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether the depth ray-caster should include teammate collision meshes.",
)
parser.add_argument(
    "--root_state_file",
    type=str,
    default=None,
    help="Optional .pt file used to save or restore exact per-agent root states for comparison runs.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for saved images. Defaults to logs/swarm_depth_inspect/<timestamp>.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_nav_task  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _make_env(task: str, *, include_teammates: bool, num_envs: int):
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "depth_include_teammates"):
        env_cfg.depth_include_teammates = include_teammates
        if hasattr(env_cfg, "apply_depth_raycast_mode"):
            env_cfg.apply_depth_raycast_mode()
    return gym.make(task, cfg=env_cfg, render_mode=None)


def _zero_actions(env) -> dict[str, torch.Tensor]:
    base_env = env.unwrapped
    actions: dict[str, torch.Tensor] = {}
    for agent in base_env.possible_agents:
        action_space = base_env.action_spaces[agent]
        if hasattr(action_space, "shape") and len(action_space.shape) > 0:
            action_dim = int(action_space.shape[0])
        else:
            action_dim = int(action_space)
        actions[agent] = torch.zeros((base_env.num_envs, action_dim), dtype=torch.float32, device=base_env.device)
    return actions


def _capture_root_states(env) -> dict[str, torch.Tensor]:
    base_env = env.unwrapped
    return {agent: base_env.robots[agent].data.root_state_w.clone() for agent in base_env.agent_ids}


def _restore_root_states(env, root_states: dict[str, torch.Tensor]) -> None:
    base_env = env.unwrapped
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    for agent in base_env.agent_ids:
        root_state = root_states[agent]
        base_env.robots[agent].write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        base_env.robots[agent].write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def _get_depth_images(env, *, downsample_hw: tuple[int, int] = (40, 64)) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base_env = env.unwrapped
    raw_images: dict[str, torch.Tensor] = {}
    downsampled_images: dict[str, torch.Tensor] = {}
    for idx, agent in enumerate(base_env.agent_ids):
        sensor = base_env.scene.sensors[f"raycast_camera_{idx}"]
        raw = sensor.data.output["distance_to_image_plane"][0, ..., 0].detach().cpu().float()
        raw = torch.nan_to_num(raw, nan=float(sensor.cfg.max_distance), posinf=float(sensor.cfg.max_distance), neginf=0.0)
        raw_images[agent] = raw
        down = F.interpolate(
            raw.unsqueeze(0).unsqueeze(0),
            size=downsample_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        downsampled_images[agent] = down
    return raw_images, downsampled_images


def _save_panel(path: str, images: dict[str, torch.Tensor], positions: dict[str, list[float]], image_title_suffix: str) -> None:
    agent_ids = list(images.keys())
    fig, axes = plt.subplots(len(agent_ids), 1, figsize=(4.6, 2.8 * len(agent_ids)), constrained_layout=True)
    if len(agent_ids) == 1:
        axes = [axes]
    for row, agent in enumerate(agent_ids):
        img = images[agent].numpy()
        axes[row].imshow(img, cmap="plasma")
        axes[row].set_title(agent)
        axes[row].axis("off")
        pos = positions[agent]
        axes[row].set_ylabel(f"x={pos[0]:.2f}\ny={pos[1]:.2f}\nz={pos[2]:.2f}", rotation=0, labelpad=38, va="center")

    fig.suptitle(f"Swarm depth capture ({image_title_suffix})", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_individual_images(directory: str, prefix: str, images: dict[str, torch.Tensor]) -> None:
    os.makedirs(directory, exist_ok=True)
    for agent, image in images.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(image.numpy(), cmap="plasma")
        ax.set_title(f"{prefix} {agent}")
        ax.axis("off")
        fig.savefig(os.path.join(directory, f"{prefix}_{agent}.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    torch.manual_seed(args_cli.seed)

    output_dir = args_cli.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            "/home/zdp/CodeField/SRU_Swarm_RL",
            "logs",
            "swarm_depth_inspect",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + ("_with_teammates" if args_cli.include_teammates else "_static_only"),
        )
    os.makedirs(output_dir, exist_ok=True)

    env = _make_env(args_cli.task, include_teammates=args_cli.include_teammates, num_envs=args_cli.num_envs)

    try:
        env.reset(seed=args_cli.seed)

        if args_cli.root_state_file and os.path.exists(args_cli.root_state_file):
            root_states = torch.load(args_cli.root_state_file, map_location=env.unwrapped.device)
            _restore_root_states(env, root_states)
        else:
            root_states = _capture_root_states(env)
            if args_cli.root_state_file:
                torch.save({agent: state.detach().cpu() for agent, state in root_states.items()}, args_cli.root_state_file)

        zero_actions = _zero_actions(env)
        env.step(zero_actions)

        raw_images, downsampled_images = _get_depth_images(env)

        base_env = env.unwrapped
        positions = {
            agent: base_env.robots[agent].data.root_pos_w[0].detach().cpu().tolist()
            for agent in base_env.agent_ids
        }
        pairwise = base_env._compute_pairwise_distances_xy()[0].detach().cpu().tolist()
        metadata = {
            "task": args_cli.task,
            "seed": args_cli.seed,
            "include_teammates": args_cli.include_teammates,
            "output_dir": output_dir,
            "agent_positions": positions,
            "cluster_goal_center": base_env.cluster_goal_center[0].detach().cpu().tolist(),
            "cluster_spawn_center": base_env.cluster_spawn_center[0].detach().cpu().tolist(),
            "source_region_id": int(base_env.source_region_ids[0].item()),
            "target_region_id": int(base_env.target_region_ids[0].item()),
            "pairwise_distance_xy": pairwise,
            "raw_image_shape": list(next(iter(raw_images.values())).shape),
            "downsampled_image_shape": list(next(iter(downsampled_images.values())).shape),
            "root_state_file": args_cli.root_state_file,
        }
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        torch.save(raw_images, os.path.join(output_dir, "raw_depth.pt"))
        torch.save(downsampled_images, os.path.join(output_dir, "downsampled_depth.pt"))

        _save_panel(
            os.path.join(output_dir, "depth_downsampled.png"),
            downsampled_images,
            positions,
            image_title_suffix="downsampled 64x40",
        )
        _save_panel(
            os.path.join(output_dir, "depth_raw.png"),
            raw_images,
            positions,
            image_title_suffix=f"raw {next(iter(raw_images.values())).shape[1]}x{next(iter(raw_images.values())).shape[0]}",
        )
        prefix = "with_teammates" if args_cli.include_teammates else "static_only"
        _save_individual_images(output_dir, f"{prefix}_downsampled", downsampled_images)
        _save_individual_images(output_dir, f"{prefix}_raw", raw_images)

        print(f"[INFO] Saved swarm depth inspection outputs to: {output_dir}")
        print(f"[INFO] Depth mode include teammates: {args_cli.include_teammates}")
        print(f"[INFO] Raw image shape: {tuple(next(iter(raw_images.values())).shape)}")
        print(f"[INFO] Downsampled image shape: {tuple(next(iter(downsampled_images.values())).shape)}")
        print("[INFO] Agent positions:")
        for agent, pos in positions.items():
            print(f"  - {agent}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
