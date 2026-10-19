from __future__ import annotations

import torch
from BrainCo_DexHand.assets.frames import frame_state_w

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

RIGHT_PALM_FRAME = "right_palm"
LEFT_PALM_FRAME = "left_palm"


def _handover_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def _palm_pos_w(robot: Articulation, frame_name: str) -> torch.Tensor:
    return frame_state_w(robot, frame_name)[:, :3]


def _source_receiver_palm_pos_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    command = _handover_command(env, command_name)
    source_is_left = command[:, 0] > 0.0
    robot: Articulation = env.scene[robot_cfg.name]
    right_palm = _palm_pos_w(robot, RIGHT_PALM_FRAME)
    left_palm = _palm_pos_w(robot, LEFT_PALM_FRAME)
    source_palm = torch.where(source_is_left.unsqueeze(-1), left_palm, right_palm)
    receiver_palm = torch.where(source_is_left.unsqueeze(-1), right_palm, left_palm)
    return source_palm, receiver_palm


def _joint_subset_ids(robot: Articulation, joint_names: tuple[str, ...] | list[str]) -> list[int]:
    joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
    return joint_ids


def _handover_frame_components(vector_w: torch.Tensor, source_palm: torch.Tensor, receiver_palm: torch.Tensor) -> torch.Tensor:
    forward = receiver_palm - source_palm
    forward = forward / torch.clamp(torch.norm(forward, p=2, dim=-1, keepdim=True), min=1.0e-6)
    up = torch.zeros_like(forward)
    up[:, 2] = 1.0
    lateral = torch.cross(up, forward, dim=-1)
    lateral = lateral / torch.clamp(torch.norm(lateral, p=2, dim=-1, keepdim=True), min=1.0e-6)
    vertical = torch.cross(forward, lateral, dim=-1)
    return torch.stack(
        (
            torch.sum(vector_w * lateral, dim=-1),
            torch.sum(vector_w * forward, dim=-1),
            torch.sum(vector_w * vertical, dim=-1),
        ),
        dim=-1,
    )


def object_pos_in_robot_root(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)


def goal_pos_in_robot_root(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, goal.data.root_pos_w - robot.data.root_pos_w)


def object_lin_vel_in_robot_root(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w)


def object_ang_vel_in_robot_root(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, obj.data.root_ang_vel_w)


def object_pos_in_handover_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    source_palm, receiver_palm = _source_receiver_palm_pos_w(env, command_name, robot_cfg)
    return _handover_frame_components(obj.data.root_pos_w - source_palm, source_palm, receiver_palm)


def goal_pos_in_handover_frame(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    goal: RigidObject = env.scene[goal_cfg.name]
    source_palm, receiver_palm = _source_receiver_palm_pos_w(env, command_name, robot_cfg)
    return _handover_frame_components(goal.data.root_pos_w - source_palm, source_palm, receiver_palm)


def object_lin_vel_in_handover_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    source_palm, receiver_palm = _source_receiver_palm_pos_w(env, command_name, robot_cfg)
    return _handover_frame_components(obj.data.root_lin_vel_w, source_palm, receiver_palm)


def object_ang_vel_in_handover_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    source_palm, receiver_palm = _source_receiver_palm_pos_w(env, command_name, robot_cfg)
    return _handover_frame_components(obj.data.root_ang_vel_w, source_palm, receiver_palm)


def _palm_pos_in_robot_root(
    env: ManagerBasedRLEnv,
    source_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    frame_name: str,
) -> torch.Tensor:
    target_robot: Articulation = env.scene[target_cfg.name]
    source_robot: Articulation = env.scene[source_cfg.name]
    palm_pos = frame_state_w(source_robot, frame_name)[:, :3]
    return math_utils.quat_apply_inverse(target_robot.data.root_quat_w, palm_pos - target_robot.data.root_pos_w)


def right_palm_pos_in_robot_root(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _palm_pos_in_robot_root(
        env,
        source_cfg=robot_cfg,
        target_cfg=robot_cfg,
        frame_name="right_palm",
    )


def left_palm_pos_in_robot_root(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _palm_pos_in_robot_root(
        env,
        source_cfg=robot_cfg,
        target_cfg=robot_cfg,
        frame_name="left_palm",
    )


def joint_pos_subset_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: tuple[str, ...] = (),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = _joint_subset_ids(robot, joint_names)
    return robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]


def joint_vel_subset_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: tuple[str, ...] = (),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = _joint_subset_ids(robot, joint_names)
    return robot.data.joint_vel[:, joint_ids] - robot.data.default_joint_vel[:, joint_ids]
