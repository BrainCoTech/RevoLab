from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from BrainCo_DexHand.assets.frames import frame_state_w

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


RIGHT_PALM_FRAME = "right_palm"
LEFT_PALM_FRAME = "left_palm"


def _get_frame_pose(robot: Articulation, env_ids: torch.Tensor, frame_name: str):
    state = frame_state_w(robot, frame_name)[env_ids]
    return state[:, :3], state[:, 3:7]


def _get_handover_term(env: ManagerBasedEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def _cache_initial_palm_state(env: ManagerBasedEnv, robot: Articulation, env_ids: torch.Tensor) -> None:
    right_pos, _ = _get_frame_pose(robot, env_ids, RIGHT_PALM_FRAME)
    left_pos, _ = _get_frame_pose(robot, env_ids, LEFT_PALM_FRAME)
    num_envs = robot.data.root_pos_w.shape[0]
    if not hasattr(env, "_handover_initial_right_palm_pos_w"):
        env._handover_initial_right_palm_pos_w = torch.zeros((num_envs, 3), device=env.device, dtype=right_pos.dtype)
        env._handover_initial_left_palm_pos_w = torch.zeros((num_envs, 3), device=env.device, dtype=left_pos.dtype)
        env._handover_initial_hand_distance = torch.zeros((num_envs,), device=env.device, dtype=right_pos.dtype)
    env._handover_initial_right_palm_pos_w[env_ids] = right_pos
    env._handover_initial_left_palm_pos_w[env_ids] = left_pos
    env._handover_initial_hand_distance[env_ids] = torch.norm(right_pos - left_pos, p=2, dim=-1)


def prepare_handover_reset_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "handover",
) -> None:
    """Sample or preserve handover commands before reset assets are placed."""
    _get_handover_term(env, command_name).prepare_reset(env_ids)

def reset_fingers_by_handover_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_finger_joint_names: tuple[str, ...] = (),
    left_finger_joint_names: tuple[str, ...] = (),
    right_source_finger_joint_pos: tuple[float, ...] = (),
    left_source_finger_joint_pos: tuple[float, ...] = (),
) -> None:
    """Reset both hands to the default open pose."""
    robot: Articulation = env.scene[robot_cfg.name]
    right_joint_ids, _ = robot.find_joints(right_finger_joint_names, preserve_order=True)
    left_joint_ids, _ = robot.find_joints(left_finger_joint_names, preserve_order=True)

    right_open_pos = robot.data.default_joint_pos[env_ids][:, right_joint_ids]
    left_open_pos = robot.data.default_joint_pos[env_ids][:, left_joint_ids]
    right_finger_pos = right_open_pos
    left_finger_pos = left_open_pos
    right_finger_vel = torch.zeros_like(right_finger_pos)
    left_finger_vel = torch.zeros_like(left_finger_pos)

    robot.write_joint_state_to_sim(
        right_finger_pos, right_finger_vel, joint_ids=right_joint_ids, env_ids=env_ids
    )
    robot.write_joint_state_to_sim(
        left_finger_pos, left_finger_vel, joint_ids=left_joint_ids, env_ids=env_ids
    )
    robot.set_joint_position_target(right_finger_pos, joint_ids=right_joint_ids, env_ids=env_ids)
    robot.set_joint_position_target(left_finger_pos, joint_ids=left_joint_ids, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    robot.update(0.0)
    _cache_initial_palm_state(env, robot, env_ids)


def reset_object_by_handover_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "handover",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_frame: str = RIGHT_PALM_FRAME,
    left_frame: str = LEFT_PALM_FRAME,
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    world_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.015),
    mirror_world_x_by_source: bool = False,
    local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    position_noise: float = 0.0,
) -> None:
    env.scene.write_data_to_sim()
    env.sim.forward()
    robot: Articulation = env.scene[robot_cfg.name]
    robot.update(0.0)

    obj: RigidObject = env.scene[asset_cfg.name]
    obj.update(0.0)
    command = _get_handover_term(env, command_name).get_command_for_envs(env_ids)
    source_is_left = command[:, 0] > 0.0

    right_pos, right_quat = _get_frame_pose(robot, env_ids, right_frame)
    left_pos, left_quat = _get_frame_pose(robot, env_ids, left_frame)
    body_pos = torch.where(source_is_left.unsqueeze(-1), left_pos, right_pos)
    body_quat = torch.where(source_is_left.unsqueeze(-1), left_quat, right_quat)

    root_state = obj.data.default_root_state[env_ids].clone()
    local_pos_tensor = torch.tensor(local_pos, device=env.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), -1)
    world_offset_tensor = torch.tensor(world_pos_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), -1).clone()
    if mirror_world_x_by_source:
        mirror_sign = torch.where(source_is_left, torch.ones_like(source_is_left, dtype=torch.float32), -torch.ones_like(source_is_left, dtype=torch.float32))
        world_offset_tensor[:, 0] *= mirror_sign
    local_quat_tensor = torch.tensor(local_quat, device=env.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), -1)
    local_offset = math_utils.quat_apply(body_quat, local_pos_tensor)
    root_state[:, :3] = body_pos + local_offset + world_offset_tensor
    if position_noise > 0.0:
        root_state[:, :3] += math_utils.sample_uniform(
            -position_noise, position_noise, root_state[:, :3].shape, root_state.device
        )
    root_state[:, 3:7] = math_utils.quat_mul(body_quat, local_quat_tensor)
    root_state[:, 7:] = 0.0
    obj.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def reset_goal_by_handover_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "handover",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_frame: str = RIGHT_PALM_FRAME,
    left_frame: str = LEFT_PALM_FRAME,
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    world_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.05),
    world_pos_offset_in_body_frame: bool = False,
    mirror_world_x_by_source: bool = False,
    local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    position_noise: float = 0.0,
) -> None:
    env.scene.write_data_to_sim()
    env.sim.forward()
    robot: Articulation = env.scene[robot_cfg.name]
    robot.update(0.0)

    goal: RigidObject = env.scene[asset_cfg.name]
    goal.update(0.0)
    command = _get_handover_term(env, command_name).get_command_for_envs(env_ids)
    source_is_left = command[:, 0] > 0.0
    throw_mode = command[:, 1] > 0.0

    right_pos, right_quat = _get_frame_pose(robot, env_ids, right_frame)
    left_pos, left_quat = _get_frame_pose(robot, env_ids, left_frame)

    receiver_pos = torch.where(source_is_left.unsqueeze(-1), right_pos, left_pos)
    receiver_quat = torch.where(source_is_left.unsqueeze(-1), right_quat, left_quat)
    source_pos = torch.where(source_is_left.unsqueeze(-1), left_pos, right_pos)
    source_quat = torch.where(source_is_left.unsqueeze(-1), left_quat, right_quat)

    body_pos = torch.where(throw_mode.unsqueeze(-1), receiver_pos, source_pos)
    body_quat = torch.where(throw_mode.unsqueeze(-1), receiver_quat, source_quat)

    root_state = goal.data.default_root_state[env_ids].clone()
    local_pos_tensor = torch.tensor(local_pos, device=env.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), -1)
    world_offset_tensor = torch.tensor(world_pos_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), -1).clone()
    if mirror_world_x_by_source:
        mirror_sign = torch.where(source_is_left, torch.ones_like(source_is_left, dtype=torch.float32), -torch.ones_like(source_is_left, dtype=torch.float32))
        world_offset_tensor[:, 0] *= mirror_sign
    local_quat_tensor = torch.tensor(local_quat, device=env.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), -1)
    local_offset = math_utils.quat_apply(body_quat, local_pos_tensor)
    if world_pos_offset_in_body_frame:
        world_offset_tensor = math_utils.quat_apply(body_quat, world_offset_tensor)
    root_state[:, :3] = body_pos + local_offset + world_offset_tensor
    if position_noise > 0.0:
        root_state[:, :3] += math_utils.sample_uniform(
            -position_noise, position_noise, root_state[:, :3].shape, root_state.device
        )
    root_state[:, 3:7] = math_utils.quat_mul(body_quat, local_quat_tensor)
    root_state[:, 7:] = 0.0
    goal.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    goal.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
