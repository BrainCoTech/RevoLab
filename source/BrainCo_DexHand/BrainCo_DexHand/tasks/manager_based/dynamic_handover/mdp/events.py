from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


RIGHT_PALM_BODY_NAMES = ("right_palm", "right_hand_base_link", "palm", "base_link")
LEFT_PALM_BODY_NAMES = ("left_palm", "left_hand_base_link", "palm", "base_link")


def _find_first_body_index(robot: Articulation, body_names: list[str] | tuple[str, ...]) -> int | None:
    for body_name in body_names:
        try:
            body_ids, _ = robot.find_bodies(body_name)
        except ValueError:
            continue
        if body_ids:
            return body_ids[0]
    return None


def _get_body_pose(
    robot: Articulation,
    env_ids: torch.Tensor,
    body_names: list[str] | tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    body_idx = _find_first_body_index(robot, body_names)
    if body_idx is None:
        return robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids]
    return robot.data.body_pos_w[env_ids, body_idx], robot.data.body_quat_w[env_ids, body_idx]


def _get_handover_term(env: ManagerBasedEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def _cache_initial_palm_state(env: ManagerBasedEnv, robot: Articulation, env_ids: torch.Tensor) -> None:
    right_pos, _ = _get_body_pose(robot, env_ids, RIGHT_PALM_BODY_NAMES)
    left_pos, _ = _get_body_pose(robot, env_ids, LEFT_PALM_BODY_NAMES)
    num_envs = robot.data.root_pos_w.shape[0]
    if not hasattr(env, "_handover_initial_right_palm_pos_w"):
        env._handover_initial_right_palm_pos_w = torch.zeros((num_envs, 3), device=env.device, dtype=right_pos.dtype)
        env._handover_initial_left_palm_pos_w = torch.zeros((num_envs, 3), device=env.device, dtype=left_pos.dtype)
        env._handover_initial_hand_distance = torch.zeros((num_envs,), device=env.device, dtype=right_pos.dtype)
    env._handover_initial_right_palm_pos_w[env_ids] = right_pos
    env._handover_initial_left_palm_pos_w[env_ids] = left_pos
    env._handover_initial_hand_distance[env_ids] = torch.norm(right_pos - left_pos, p=2, dim=-1)


class reset_object_pose(ManagerTermBase):
    """Reset the object pose to the IsaacGymEnvs handover start state."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: RigidObject = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("object")).name]
        self._base_pos = torch.tensor(cfg.params["base_pos"], device=env.device, dtype=torch.float32)
        self._position_noise = float(cfg.params.get("position_noise", 0.0))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        base_pos: tuple[float, float, float] = (0.025, -0.38, 0.449),
        position_noise: float = 0.0,
    ) -> None:
        root_state = self._asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self._base_pos + env.scene.env_origins[env_ids]
        if self._position_noise > 0.0:
            root_state[:, :3] += math_utils.sample_uniform(
                -self._position_noise, self._position_noise, root_state[:, :3].shape, root_state.device
            )
        quat = math_utils.quat_from_euler_xyz(
            torch.sign(torch.rand(len(env_ids), device=env.device) - 0.5) * torch.pi,
            torch.sign(torch.rand(len(env_ids), device=env.device) - 0.5) * torch.pi + 1.571,
            torch.sign(torch.rand(len(env_ids), device=env.device) - 0.5) * torch.pi,
        )
        root_state[:, 3:7] = quat
        root_state[:, 7:] = 0.0
        self._asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self._asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


class reset_goal_pose(ManagerTermBase):
    """Reset the kinematic goal marker to the legacy handover target distribution."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: RigidObject = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("goal")).name]
        self._base_pos = torch.tensor(cfg.params["base_pos"], device=env.device, dtype=torch.float32)
        self._x_range = cfg.params.get("x_range", (-0.05, 0.05))
        self._y_offset = float(cfg.params.get("y_offset", -0.55))
        self._y_range = cfg.params.get("y_range", (-0.05, 0.05))
        self._z_offset = float(cfg.params.get("z_offset", 0.10))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
        base_pos: tuple[float, float, float] = (0.025, -0.38, 0.449),
        x_range: tuple[float, float] = (-0.05, 0.05),
        y_offset: float = -0.55,
        y_range: tuple[float, float] = (-0.05, 0.05),
        z_offset: float = 0.10,
    ) -> None:
        root_state = self._asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self._base_pos + env.scene.env_origins[env_ids]
        root_state[:, 0] += math_utils.sample_uniform(
            self._x_range[0], self._x_range[1], (len(env_ids),), env.device
        )
        root_state[:, 1] += self._y_offset + math_utils.sample_uniform(
            self._y_range[0], self._y_range[1], (len(env_ids),), env.device
        )
        root_state[:, 2] += self._z_offset
        quat = math_utils.quat_from_euler_xyz(
            torch.sign(torch.rand(len(env_ids), device=env.device) - 0.5) * torch.pi,
            torch.sign(torch.rand(len(env_ids), device=env.device) - 0.5) * torch.pi + 1.571,
            torch.sign(torch.rand(len(env_ids), device=env.device) - 0.5) * torch.pi,
        )
        root_state[:, 3:7] = quat
        root_state[:, 7:] = 0.0
        self._asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self._asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


class reset_object_in_right_hand(ManagerTermBase):
    """Reset the object into a pre-grasp pose relative to the right hand."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: RigidObject = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("object")).name]
        self._robot: Articulation = env.scene[cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name]
        self._body_names = list(cfg.params.get("body_names", ["right_palm", "right_hand_base_link", "palm", "base_link"]))
        self._local_pos = torch.tensor(cfg.params.get("local_pos", (0.0, 0.0, 0.0)), device=env.device, dtype=torch.float32)
        self._world_pos_offset = torch.tensor(
            cfg.params.get("world_pos_offset", (0.0, 0.0, 0.10)), device=env.device, dtype=torch.float32
        )
        self._local_quat = torch.tensor(
            cfg.params.get("local_quat", (1.0, 0.0, 0.0, 0.0)), device=env.device, dtype=torch.float32
        )
        self._position_noise = float(cfg.params.get("position_noise", 0.0))
        self._body_idx = None
        for body_name in self._body_names:
            try:
                body_ids, _ = self._robot.find_bodies(body_name)
            except ValueError:
                continue
            if body_ids:
                self._body_idx = body_ids[0]
                break

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        body_names: list[str] | tuple[str, ...] = ("right_palm", "right_hand_base_link", "palm", "base_link"),
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        world_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.10),
        local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        position_noise: float = 0.0,
    ) -> None:
        # The default scene reset writes robot root/joint state back to PhysX but does not immediately
        # refresh articulated body poses. Force one sync so headless and GUI reset paths use the same
        # right-hand body pose when spawning the object into the grasp.
        env.scene.write_data_to_sim()
        env.sim.forward()
        self._robot.update(0.0)
        self._asset.update(0.0)

        root_state = self._asset.data.default_root_state[env_ids].clone()
        if self._body_idx is None:
            body_pos = self._robot.data.root_pos_w[env_ids]
            body_quat = self._robot.data.root_quat_w[env_ids]
        else:
            body_pos = self._robot.data.body_pos_w[env_ids, self._body_idx]
            body_quat = self._robot.data.body_quat_w[env_ids, self._body_idx]
        local_offset = math_utils.quat_apply(body_quat, self._local_pos.unsqueeze(0).expand(len(env_ids), -1))
        world_offset = self._world_pos_offset.unsqueeze(0).expand(len(env_ids), -1)
        root_state[:, :3] = body_pos + local_offset + world_offset
        if self._position_noise > 0.0:
            root_state[:, :3] += math_utils.sample_uniform(
                -self._position_noise, self._position_noise, root_state[:, :3].shape, root_state.device
            )
        local_quat = self._local_quat.unsqueeze(0).expand(len(env_ids), -1)
        root_state[:, 3:7] = math_utils.quat_mul(body_quat, local_quat)
        root_state[:, 7:] = 0.0
        self._asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self._asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


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
    right_body_names: tuple[str, ...] = RIGHT_PALM_BODY_NAMES,
    left_body_names: tuple[str, ...] = LEFT_PALM_BODY_NAMES,
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

    right_pos, right_quat = _get_body_pose(robot, env_ids, right_body_names)
    left_pos, left_quat = _get_body_pose(robot, env_ids, left_body_names)
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
    right_body_names: tuple[str, ...] = RIGHT_PALM_BODY_NAMES,
    left_body_names: tuple[str, ...] = LEFT_PALM_BODY_NAMES,
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

    right_pos, right_quat = _get_body_pose(robot, env_ids, right_body_names)
    left_pos, left_quat = _get_body_pose(robot, env_ids, left_body_names)

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
