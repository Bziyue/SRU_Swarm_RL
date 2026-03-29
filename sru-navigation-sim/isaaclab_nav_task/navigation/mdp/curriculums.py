# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Curriculum functions for navigation tasks.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def disable_backward_penalty_after_steps(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    term_name: str = "backward_movement_penalty", 
    num_steps: int = 1000
) -> torch.Tensor:
    """Curriculum that disables the backward movement penalty after a certain number of steps.
    
    This helps with early training by preventing backward movement, but removes the constraint
    later to allow more natural movement patterns.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the backward movement penalty term.
        num_steps: The number of steps after which the penalty should be disabled.
        
    Returns:
        Current step counter as float for logging purposes.
    """
    if env.common_step_counter > num_steps:
        # Check if the term exists and has a non-zero weight
        if hasattr(env.reward_manager, 'get_term_cfg'):
            try:
                term_cfg = env.reward_manager.get_term_cfg(term_name)
                if term_cfg.weight != 0.0:
                    # Disable the penalty by setting weight to 0
                    term_cfg.weight = 0.0
                    env.reward_manager.set_term_cfg(term_name, term_cfg)
                    print(f"Disabled backward movement penalty at step {env.common_step_counter}")
            except KeyError:
                # Term doesn't exist, which is fine
                pass
    
    return torch.tensor(float(env.common_step_counter))


def success_gate_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    warmup_steps: int = 32_000,
    ramp_steps: int = 96_000,
    start_distance_threshold: float = 0.6,
    final_distance_threshold: float = 0.5,
    start_max_xy_speed: float = 0.35,
    final_max_xy_speed: float = 0.25,
    start_hold_time_s: float = 0.5,
    final_hold_time_s: float = 1.0,
    termination_terms: tuple[str, ...] = ("time_out", "early_termination"),
    reward_terms: tuple[str, ...] = ("success_bonus",),
) -> dict[str, float]:
    """Gradually tighten the success gate over training.

    The schedule is:
    - Warmup: keep the easier starting thresholds.
    - Ramp: linearly interpolate to the final thresholds.
    - Final: keep the stricter thresholds.
    """
    del env_ids

    steps = int(env.common_step_counter)
    if steps <= warmup_steps:
        progress = 0.0
    elif ramp_steps <= 0:
        progress = 1.0
    else:
        progress = min(max((steps - warmup_steps) / float(ramp_steps), 0.0), 1.0)

    distance_threshold = (1.0 - progress) * start_distance_threshold + progress * final_distance_threshold
    max_xy_speed = (1.0 - progress) * start_max_xy_speed + progress * final_max_xy_speed
    hold_time_s = (1.0 - progress) * start_hold_time_s + progress * final_hold_time_s

    if hasattr(env, "termination_manager") and hasattr(env.termination_manager, "get_term_cfg"):
        for term_name in termination_terms:
            try:
                term_cfg = env.termination_manager.get_term_cfg(term_name)
            except KeyError:
                continue
            params = dict(term_cfg.params or {})
            changed = False
            if "distance_threshold" in params and abs(float(params["distance_threshold"]) - distance_threshold) > 1e-6:
                params["distance_threshold"] = float(distance_threshold)
                changed = True
            if "max_xy_speed" in params and abs(float(params["max_xy_speed"]) - max_xy_speed) > 1e-6:
                params["max_xy_speed"] = float(max_xy_speed)
                changed = True
            if "hold_time_s" in params and abs(float(params["hold_time_s"]) - hold_time_s) > 1e-6:
                params["hold_time_s"] = float(hold_time_s)
                changed = True
            if changed:
                term_cfg.params = params
                env.termination_manager.set_term_cfg(term_name, term_cfg)

    if hasattr(env, "reward_manager") and hasattr(env.reward_manager, "get_term_cfg"):
        for term_name in reward_terms:
            try:
                term_cfg = env.reward_manager.get_term_cfg(term_name)
            except KeyError:
                continue
            params = dict(term_cfg.params or {})
            changed = False
            if "distance_threshold" in params and abs(float(params["distance_threshold"]) - distance_threshold) > 1e-6:
                params["distance_threshold"] = float(distance_threshold)
                changed = True
            if "max_xy_speed" in params and abs(float(params["max_xy_speed"]) - max_xy_speed) > 1e-6:
                params["max_xy_speed"] = float(max_xy_speed)
                changed = True
            if "hold_time_s" in params and abs(float(params["hold_time_s"]) - hold_time_s) > 1e-6:
                params["hold_time_s"] = float(hold_time_s)
                changed = True
            if changed:
                term_cfg.params = params
                env.reward_manager.set_term_cfg(term_name, term_cfg)

    return {
        "common_step_counter": float(steps),
        "distance_threshold": float(distance_threshold),
        "max_xy_speed": float(max_xy_speed),
        "hold_time_s": float(hold_time_s),
        "progress": float(progress),
    }


def staged_episode_horizon_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    first_stage_steps: int = 64_000,
    second_stage_steps: int = 96_000,
    first_stage_seconds: float = 60.0,
    second_stage_seconds: float = 90.0,
    final_stage_seconds: float = 120.0,
) -> dict[str, float]:
    """Stage the episode horizon over training.

    This keeps the early solo-pretrain objective close to the reference setup while still
    exposing longer horizons later for long-path samples.
    """
    del env_ids

    steps = int(env.common_step_counter)
    if steps < first_stage_steps:
        stage = 0
        target_seconds = first_stage_seconds
    elif steps < second_stage_steps:
        stage = 1
        target_seconds = second_stage_seconds
    else:
        stage = 2
        target_seconds = final_stage_seconds

    if abs(float(env.cfg.episode_length_s) - float(target_seconds)) > 1e-6:
        env.cfg.episode_length_s = float(target_seconds)

    return {
        "common_step_counter": float(steps),
        "episode_length_s": float(env.cfg.episode_length_s),
        "stage": float(stage),
    }
