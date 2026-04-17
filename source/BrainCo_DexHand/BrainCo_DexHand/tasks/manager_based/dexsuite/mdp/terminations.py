# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)


def non_finite_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when NaN/Inf appears in key simulation states.

    This typically happens when reset randomization spawns interpenetrating bodies (hand/object/table),
    or when contacts/constraints blow up numerically. Ending the episode early prevents PPO from
    ingesting NaN/Inf rewards/observations and "collapsing".
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Robot joint states
    jp = robot.data.joint_pos
    jv = robot.data.joint_vel

    # Root states
    rpos = robot.data.root_pos_w
    rquat = robot.data.root_quat_w
    opos = obj.data.root_pos_w
    oquat = obj.data.root_quat_w

    finite_robot = (
        torch.isfinite(jp).all(dim=1)
        & torch.isfinite(jv).all(dim=1)
        & torch.isfinite(rpos).all(dim=1)
        & torch.isfinite(rquat).all(dim=1)
    )
    finite_obj = torch.isfinite(opos).all(dim=1) & torch.isfinite(oquat).all(dim=1)

    return ~(finite_robot & finite_obj)


def _get_command_success_streak(env: ManagerBasedRLEnv, command_name: str = "object_pose") -> torch.Tensor:
    """Return per-env consecutive success counts for the given command term.

    The command term is expected to maintain a ``metrics["success"]`` tensor. We cache the
    updated streak once per environment step so multiple termination terms can share it
    without double-counting.
    """

    streak_attr = f"_{command_name}_success_streak"
    step_attr = f"_{command_name}_success_streak_step"

    if not hasattr(env, streak_attr) or getattr(env, streak_attr).shape[0] != env.num_envs:
        setattr(env, streak_attr, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
        setattr(env, step_attr, -1)

    current_step = int(getattr(env, "common_step_counter", 0))
    last_step = getattr(env, step_attr)
    streak = getattr(env, streak_attr)

    if last_step != current_step:
        # After an environment reset, the first post-reset step has episode length 1.
        # Clear the stale streak before using the current success flag.
        new_episode = env.episode_length_buf <= 1
        streak = torch.where(new_episode, torch.zeros_like(streak), streak)

        command_term = env.command_manager.get_term(command_name)
        success = command_term.metrics["success"].to(dtype=torch.bool)
        streak = torch.where(success, streak + 1, torch.zeros_like(streak))

        setattr(env, streak_attr, streak)
        setattr(env, step_attr, current_step)

    return getattr(env, streak_attr)


def consecutive_success_state_with_min_length(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    num_consecutive_successes: int = 5,
    min_episode_length: int = 10,
) -> torch.Tensor:
    """Terminate when the commanded task succeeds for N consecutive steps.

    This mirrors the success gating used by UWLab's RGB data-collection tasks so the
    recorder manager can export only successful trajectories.
    """

    streak = _get_command_success_streak(env, command_name=command_name)
    return (streak >= num_consecutive_successes) & (env.episode_length_buf >= min_episode_length)


def early_success_termination(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    num_consecutive_successes: int = 5,
    min_episode_length: int = 10,
) -> torch.Tensor:
    """Early-stop episodes once they have been successful for enough consecutive steps."""

    return consecutive_success_state_with_min_length(
        env,
        command_name=command_name,
        num_consecutive_successes=num_consecutive_successes,
        min_episode_length=min_episode_length,
    )
