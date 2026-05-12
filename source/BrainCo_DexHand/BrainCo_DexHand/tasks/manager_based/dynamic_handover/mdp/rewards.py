from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

RIGHT_PALM_BODY_NAMES = ("right_palm", "right_hand_base_link", "palm", "base_link")
LEFT_PALM_BODY_NAMES = ("left_palm", "left_hand_base_link", "palm", "base_link")


def _first_matching_body_pos_w(
    robot: Articulation,
    body_names: tuple[str, ...],
) -> torch.Tensor:
    for body_name in body_names:
        try:
            body_ids, _ = robot.find_bodies(body_name)
        except ValueError:
            continue
        if body_ids:
            return robot.data.body_pos_w[:, body_ids[0]]
    return robot.data.root_pos_w


def _first_matching_body_vel_w(
    robot: Articulation,
    body_names: tuple[str, ...],
) -> torch.Tensor:
    for body_name in body_names:
        try:
            body_ids, _ = robot.find_bodies(body_name)
        except ValueError:
            continue
        if body_ids:
            return robot.data.body_lin_vel_w[:, body_ids[0]]
    return robot.data.root_lin_vel_w


def _handover_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def _source_receiver_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Articulation]:
    command = _handover_command(env, command_name)
    source_is_left = command[:, 0] > 0.0
    throw_gate = (command[:, 1] > 0.0).float()
    hold_gate = 1.0 - throw_gate
    robot: Articulation = env.scene[robot_cfg.name]
    return command, source_is_left, throw_gate, hold_gate, robot


def _source_receiver_palm_state(
    env: ManagerBasedRLEnv,
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    command, source_is_left, throw_gate, hold_gate, robot = _source_receiver_state(env, command_name, robot_cfg)
    right_palm = _first_matching_body_pos_w(robot, RIGHT_PALM_BODY_NAMES)
    left_palm = _first_matching_body_pos_w(robot, LEFT_PALM_BODY_NAMES)
    source_palm = torch.where(source_is_left.unsqueeze(-1), left_palm, right_palm)
    receiver_palm = torch.where(source_is_left.unsqueeze(-1), right_palm, left_palm)
    return command, source_palm, receiver_palm, throw_gate, hold_gate


def _receiver_palm_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = _handover_command(env, command_name)
    source_is_left = command[:, 0] > 0.0
    robot: Articulation = env.scene[robot_cfg.name]
    right_vel = _first_matching_body_vel_w(robot, RIGHT_PALM_BODY_NAMES)
    left_vel = _first_matching_body_vel_w(robot, LEFT_PALM_BODY_NAMES)
    return torch.where(source_is_left.unsqueeze(-1), right_vel, left_vel)


def _initial_palm_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[robot_cfg.name]
    current_right = _first_matching_body_pos_w(robot, RIGHT_PALM_BODY_NAMES)
    current_left = _first_matching_body_pos_w(robot, LEFT_PALM_BODY_NAMES)
    initial_right = getattr(env, "_handover_initial_right_palm_pos_w", current_right)
    initial_left = getattr(env, "_handover_initial_left_palm_pos_w", current_left)
    initial_distance = getattr(env, "_handover_initial_hand_distance", torch.norm(current_right - current_left, p=2, dim=-1))
    return initial_right, initial_left, initial_distance


def goal_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    return torch.norm(goal.data.root_pos_w - obj.data.root_pos_w, p=2, dim=-1)


def _handover_throw_metrics(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    _, source_palm, receiver_palm, throw_gate, _ = _source_receiver_palm_state(env, command_name, robot_cfg)
    dist_to_source = torch.norm(obj.data.root_pos_w - source_palm, p=2, dim=-1)
    dist_to_receiver = torch.norm(obj.data.root_pos_w - receiver_palm, p=2, dim=-1)
    dist_to_goal = torch.norm(goal.data.root_pos_w - obj.data.root_pos_w, p=2, dim=-1)
    speed = torch.norm(obj.data.root_lin_vel_w, p=2, dim=-1)
    return throw_gate, dist_to_source, dist_to_receiver, dist_to_goal, speed


def _source_goal_progress_metrics(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    _, source_palm, _, throw_gate, _ = _source_receiver_palm_state(env, command_name, robot_cfg)
    source_to_goal = goal.data.root_pos_w - source_palm
    path_length = torch.norm(source_to_goal, p=2, dim=-1, keepdim=True).clamp_min(1.0e-6)
    path_dir = source_to_goal / path_length
    source_to_object = obj.data.root_pos_w - source_palm
    progress = torch.sum(source_to_object * path_dir, dim=-1)
    normalized_progress = torch.clamp(progress / path_length.squeeze(-1), min=0.0, max=1.2)
    projected_speed = torch.sum(obj.data.root_lin_vel_w * path_dir, dim=-1)
    return throw_gate, normalized_progress, projected_speed, path_length.squeeze(-1), torch.norm(source_to_object, p=2, dim=-1)


def exp_goal_tracking(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    sharpness: float = 12.0,
    command_name: str = "handover",
) -> torch.Tensor:
    dist = goal_distance(env, object_cfg=object_cfg, goal_cfg=goal_cfg)
    throw_gate = (_handover_command(env, command_name)[:, 1] > 0.0).float()
    return torch.exp(-sharpness * dist) * throw_gate


def source_release_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    release_distance: float = 0.10,
    catch_goal_distance: float = 0.08,
    speed_floor: float = 0.18,
) -> torch.Tensor:
    """Stage A: reward releasing the cube from the source hand, but stop after catch."""
    throw_gate, dist_to_source, _, dist_to_goal, speed = _handover_throw_metrics(env, object_cfg, goal_cfg, command_name, robot_cfg)
    released = torch.clamp(dist_to_source / max(release_distance, 1.0e-6), max=1.0)
    active = (dist_to_goal > catch_goal_distance) & (speed > speed_floor)
    return released * active.float() * throw_gate


def flight_velocity_band_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    release_distance: float = 0.10,
    goal_distance_window: float = 0.10,
    target_speed: float = 0.85,
    speed_tolerance: float = 0.35,
) -> torch.Tensor:
    """Stage B: reward forward flight toward the goal with a moderate projected speed band."""
    throw_gate, dist_to_source, _, dist_to_goal, speed = _handover_throw_metrics(env, object_cfg, goal_cfg, command_name, robot_cfg)
    _, normalized_progress, projected_speed, _, _ = _source_goal_progress_metrics(env, object_cfg, goal_cfg, command_name, robot_cfg)
    in_flight = (dist_to_source > release_distance) & (dist_to_goal > goal_distance_window)
    forward_speed = torch.clamp(projected_speed, min=0.0)
    band_reward = torch.exp(-torch.square((forward_speed - target_speed) / max(speed_tolerance, 1.0e-6)))
    progress_bonus = normalized_progress
    return (0.6 * band_reward + 0.4 * progress_bonus) * in_flight.float() * throw_gate


def catch_goal_proximity_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    sharpness: float = 14.0,
) -> torch.Tensor:
    """Stage C: reward approaching the blue goal point near the receiver."""
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    throw_gate = (_handover_command(env, command_name)[:, 1] > 0.0).float()
    dist = torch.norm(obj.data.root_pos_w - goal.data.root_pos_w, p=2, dim=-1)
    return torch.exp(-sharpness * dist) * throw_gate


def stalled_flight_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    release_distance: float = 0.10,
    goal_distance_window: float = 0.10,
    min_projected_speed: float = 0.25,
) -> torch.Tensor:
    """Penalize released cubes that stall before reaching the goal region."""
    throw_gate, dist_to_source, _, dist_to_goal, _ = _handover_throw_metrics(env, object_cfg, goal_cfg, command_name, robot_cfg)
    _, _, projected_speed, _, _ = _source_goal_progress_metrics(env, object_cfg, goal_cfg, command_name, robot_cfg)
    stalled = (dist_to_source > release_distance) & (dist_to_goal > goal_distance_window)
    speed_deficit = torch.clamp(min_projected_speed - projected_speed, min=0.0)
    return speed_deficit * stalled.float() * throw_gate


def catch_stability_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_distance_threshold: float = 0.06,
    receiver_distance_threshold: float = 0.06,
    stable_speed_threshold: float = 0.30,
) -> torch.Tensor:
    """Stage D: reward a stable low-speed catch near the goal and receiver hand."""
    throw_gate, _, dist_to_receiver, dist_to_goal, speed = _handover_throw_metrics(env, object_cfg, goal_cfg, command_name, robot_cfg)
    near_goal = dist_to_goal < goal_distance_threshold
    near_receiver = dist_to_receiver < receiver_distance_threshold
    low_speed = torch.clamp(1.0 - speed / max(stable_speed_threshold, 1.0e-6), min=0.0, max=1.0)
    stable = near_goal & near_receiver
    return low_speed * stable.float() * throw_gate


def object_transfer_velocity_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reward_clip: float = 0.2,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    _, source_palm, receiver_palm, throw_gate, _ = _source_receiver_palm_state(env, command_name, robot_cfg)
    direction = receiver_palm - source_palm
    direction = direction / torch.clamp(torch.norm(direction, p=2, dim=-1, keepdim=True), min=1e-6)
    projected_velocity = torch.sum(obj.data.root_lin_vel_w * direction, dim=-1)
    return torch.clamp(projected_velocity, -reward_clip, reward_clip) * throw_gate


def object_away_from_source_palm_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    palm_distance_threshold: float = 0.06,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    _, source_palm, _, throw_gate, _ = _source_receiver_palm_state(env, command_name, robot_cfg)
    dist = torch.norm(obj.data.root_pos_w - source_palm, p=2, dim=-1)
    return torch.clamp(palm_distance_threshold - dist, min=0.0) * throw_gate


def object_towards_receiver_palm_velocity_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    throw_gate = (_handover_command(env, command_name)[:, 1] > 0.0).float()
    direction = goal.data.root_pos_w - obj.data.root_pos_w
    direction = direction / torch.clamp(torch.norm(direction, p=2, dim=-1, keepdim=True), min=1e-6)
    return torch.sum(obj.data.root_lin_vel_w * direction, dim=-1) * throw_gate


def object_close_to_receiver_palm_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sharpness: float = 10.0,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    throw_gate = (_handover_command(env, command_name)[:, 1] > 0.0).float()
    dist = torch.norm(obj.data.root_pos_w - goal.data.root_pos_w, p=2, dim=-1)
    return torch.exp(-sharpness * dist) * throw_gate


def receiver_palm_towards_object_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    release_distance: float = 0.10,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    _, source_palm, receiver_palm, throw_gate, _ = _source_receiver_palm_state(env, command_name, robot_cfg)
    receiver_vel = _receiver_palm_velocity(env, command_name, robot_cfg)
    direction = obj.data.root_pos_w - receiver_palm
    direction = direction / torch.clamp(torch.norm(direction, p=2, dim=-1, keepdim=True), min=1e-6)
    released = torch.norm(obj.data.root_pos_w - source_palm, p=2, dim=-1) > release_distance
    return torch.sum(receiver_vel * direction, dim=-1) * released.float() * throw_gate


def hand_palm_separation_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    initial_distance: float | None = None,
    distance_margin: float = 0.05,
) -> torch.Tensor:
    """Penalize the hands when their palm distance is more than ``distance_margin`` below reset spacing."""
    robot: Articulation = env.scene[robot_cfg.name]
    right_palm = _first_matching_body_pos_w(robot, RIGHT_PALM_BODY_NAMES)
    left_palm = _first_matching_body_pos_w(robot, LEFT_PALM_BODY_NAMES)
    _, _, cached_initial_distance = _initial_palm_state(env, robot_cfg)
    distance = torch.norm(right_palm - left_palm, p=2, dim=-1)
    if initial_distance is None:
        minimum_distance = torch.clamp(cached_initial_distance - distance_margin, min=0.0)
    else:
        minimum_distance = torch.full_like(distance, max(initial_distance - distance_margin, 0.0))
    return torch.clamp(minimum_distance - distance, min=0.0)


def object_y_velocity_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    corridor_min_progress: float = 0.15,
    corridor_max_progress: float = 0.85,
    reward_clip: float = 0.1,
) -> torch.Tensor:
    """Reward cube flight along the initial source-to-receiver y corridor of the bimanual robot."""
    obj: RigidObject = env.scene[object_cfg.name]
    command = _handover_command(env, command_name)
    source_is_left = command[:, 0] > 0.0
    throw_gate = (command[:, 1] > 0.0).float()
    initial_right, initial_left, _ = _initial_palm_state(env, robot_cfg)
    source_init = torch.where(source_is_left.unsqueeze(-1), initial_left, initial_right)
    receiver_init = torch.where(source_is_left.unsqueeze(-1), initial_right, initial_left)
    segment_y = receiver_init[:, 1] - source_init[:, 1]
    corridor_start = source_init[:, 1] + corridor_min_progress * segment_y
    corridor_end = source_init[:, 1] + corridor_max_progress * segment_y
    corridor_low = torch.minimum(corridor_start, corridor_end)
    corridor_high = torch.maximum(corridor_start, corridor_end)
    direction_sign = torch.where(segment_y >= 0.0, torch.ones_like(segment_y), -torch.ones_like(segment_y))
    signed_y_velocity = direction_sign * obj.data.root_lin_vel_w[:, 1]
    active = (obj.data.root_pos_w[:, 1] > corridor_low) & (obj.data.root_pos_w[:, 1] < corridor_high)
    return torch.where(active, torch.clamp(signed_y_velocity, -reward_clip, reward_clip), torch.zeros_like(signed_y_velocity)) * throw_gate


def receiver_finger_surface_contacts_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    surface_threshold: float = 0.028,
    object_half_extents: tuple[float, float, float] = (0.03, 0.03, 0.025),
) -> torch.Tensor:
    """Reward the receiver fingertips for forming a catch pose around the cube surface."""
    obj: RigidObject = env.scene[object_cfg.name]
    command = _handover_command(env, command_name)
    source_is_left = command[:, 0] > 0.0
    throw_gate = (command[:, 1] > 0.0).float()
    robot: Articulation = env.scene[robot_cfg.name]

    right_tip_names = (
        "right_thumb_tip_Link",
        "right_index_tip_Link",
        "right_middle_tip_Link",
        "right_ring_tip_Link",
    )
    left_tip_names = (
        "left_thumb_tip_Link",
        "left_index_tip_Link",
        "left_middle_tip_Link",
        "left_ring_tip_Link",
    )

    right_tip_positions = [_first_matching_body_pos_w(robot, (body_name,)) for body_name in right_tip_names]
    left_tip_positions = [_first_matching_body_pos_w(robot, (body_name,)) for body_name in left_tip_names]
    right_tips = torch.stack(right_tip_positions, dim=1)
    left_tips = torch.stack(left_tip_positions, dim=1)
    receiver_tips = torch.where(source_is_left.view(-1, 1, 1), right_tips, left_tips)

    object_quat = obj.data.root_quat_w.unsqueeze(1).expand(-1, receiver_tips.shape[1], -1)
    tip_pos_obj = math_utils.quat_apply_inverse(object_quat, receiver_tips - obj.data.root_pos_w.unsqueeze(1))
    half_extents = torch.tensor(object_half_extents, device=env.device, dtype=receiver_tips.dtype).view(1, 1, 3)
    q = torch.abs(tip_pos_obj) - half_extents
    outside_distance = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    inside_distance = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
    surface_distance = torch.abs(outside_distance + inside_distance)
    surface_reward = torch.clamp(surface_threshold - surface_distance, min=0.0) / max(surface_threshold, 1.0e-6)
    return surface_reward.mean(dim=-1) * throw_gate


def receiver_multi_finger_surface_contacts_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    surface_threshold: float = 0.028,
    min_fingers: int = 2,
    object_half_extents: tuple[float, float, float] = (0.03, 0.03, 0.025),
) -> torch.Tensor:
    """Reward the receiver only after at least ``min_fingers`` fingertips reach the cube surface."""
    obj: RigidObject = env.scene[object_cfg.name]
    command = _handover_command(env, command_name)
    source_is_left = command[:, 0] > 0.0
    throw_gate = (command[:, 1] > 0.0).float()
    robot: Articulation = env.scene[robot_cfg.name]

    right_tip_names = (
        "right_thumb_tip_Link",
        "right_index_tip_Link",
        "right_middle_tip_Link",
        "right_ring_tip_Link",
    )
    left_tip_names = (
        "left_thumb_tip_Link",
        "left_index_tip_Link",
        "left_middle_tip_Link",
        "left_ring_tip_Link",
    )

    right_tip_positions = [_first_matching_body_pos_w(robot, (body_name,)) for body_name in right_tip_names]
    left_tip_positions = [_first_matching_body_pos_w(robot, (body_name,)) for body_name in left_tip_names]
    right_tips = torch.stack(right_tip_positions, dim=1)
    left_tips = torch.stack(left_tip_positions, dim=1)
    receiver_tips = torch.where(source_is_left.view(-1, 1, 1), right_tips, left_tips)

    object_quat = obj.data.root_quat_w.unsqueeze(1).expand(-1, receiver_tips.shape[1], -1)
    tip_pos_obj = math_utils.quat_apply_inverse(object_quat, receiver_tips - obj.data.root_pos_w.unsqueeze(1))
    half_extents = torch.tensor(object_half_extents, device=env.device, dtype=receiver_tips.dtype).view(1, 1, 3)
    q = torch.abs(tip_pos_obj) - half_extents
    outside_distance = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    inside_distance = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
    surface_distance = torch.abs(outside_distance + inside_distance)
    per_finger_reward = torch.clamp(surface_threshold - surface_distance, min=0.0) / max(surface_threshold, 1.0e-6)
    contact_mask = surface_distance < surface_threshold
    contact_count = contact_mask.sum(dim=-1)
    enough_contacts = contact_count >= min_fingers
    normalized_count = torch.clamp(contact_count.float() - (min_fingers - 1), min=0.0) / max(
        receiver_tips.shape[1] - (min_fingers - 1), 1
    )
    contact_quality = (per_finger_reward * contact_mask.float()).sum(dim=-1) / contact_count.clamp_min(1).float()
    return normalized_count * contact_quality * enough_contacts.float() * throw_gate


def source_arm_return_to_default_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    release_distance: float = 0.10,
    sharpness: float = 4.0,
    arm_weight: float = 4.0,
    finger_weight: float = 0.5,
    right_arm_joint_names: tuple[str, ...] = (
        "Joint1_R",
        "Joint2_R",
        "Joint3_R",
        "Joint4_R",
        "Joint5_R",
        "Joint6_R",
        "Joint7_R",
    ),
    left_arm_joint_names: tuple[str, ...] = (
        "Joint1_L",
        "Joint2_L",
        "Joint3_L",
        "Joint4_L",
        "Joint5_L",
        "Joint6_L",
        "Joint7_L",
    ),
    right_finger_joint_names: tuple[str, ...] = (
        "right_thumb_CMP_joint",
        "right_thumb_CMR_joint",
        "right_thumb_MCP_joint",
        "right_thumb_PIP_joint",
        "right_thumb_DIP_joint",
        "right_index_MPR_joint",
        "right_index_MCP_joint",
        "right_index_PIP_joint",
        "right_index_DIP_joint",
        "right_middle_MPR_joint",
        "right_middle_MCP_joint",
        "right_middle_PIP_joint",
        "right_middle_DIP_joint",
        "right_ring_MPR_joint",
        "right_ring_MCP_joint",
        "right_ring_PIP_joint",
        "right_ring_DIP_joint",
        "right_little_MPR_joint",
        "right_little_MCP_joint",
        "right_little_PIP_joint",
        "right_little_DIP_joint",
    ),
    left_finger_joint_names: tuple[str, ...] = (
        "left_thumb_CMP_joint",
        "left_thumb_CMR_joint",
        "left_thumb_MCP_joint",
        "left_thumb_PIP_joint",
        "left_thumb_DIP_joint",
        "left_index_MPR_joint",
        "left_index_MCP_joint",
        "left_index_PIP_joint",
        "left_index_DIP_joint",
        "left_middle_MPR_joint",
        "left_middle_MCP_joint",
        "left_middle_PIP_joint",
        "left_middle_DIP_joint",
        "left_ring_MPR_joint",
        "left_ring_MCP_joint",
        "left_ring_PIP_joint",
        "left_ring_DIP_joint",
        "left_little_MPR_joint",
        "left_little_MCP_joint",
        "left_little_PIP_joint",
        "left_little_DIP_joint",
    ),
) -> torch.Tensor:
    """Reward the throwing-side arm returning to default, with optional light finger regularization after release."""
    obj: RigidObject = env.scene[object_cfg.name]
    command, source_palm, _, throw_gate, _ = _source_receiver_palm_state(env, command_name, robot_cfg)
    source_is_left = command[:, 0] > 0.0
    robot: Articulation = env.scene[robot_cfg.name]

    right_joint_ids, _ = robot.find_joints(right_arm_joint_names, preserve_order=True)
    left_joint_ids, _ = robot.find_joints(left_arm_joint_names, preserve_order=True)
    right_finger_ids, _ = robot.find_joints(right_finger_joint_names, preserve_order=True)
    left_finger_ids, _ = robot.find_joints(left_finger_joint_names, preserve_order=True)

    right_error = robot.data.joint_pos[:, right_joint_ids] - robot.data.default_joint_pos[:, right_joint_ids]
    left_error = robot.data.joint_pos[:, left_joint_ids] - robot.data.default_joint_pos[:, left_joint_ids]
    right_finger_error = robot.data.joint_pos[:, right_finger_ids] - robot.data.default_joint_pos[:, right_finger_ids]
    left_finger_error = robot.data.joint_pos[:, left_finger_ids] - robot.data.default_joint_pos[:, left_finger_ids]

    source_arm_error = torch.where(source_is_left.unsqueeze(-1), left_error, right_error)
    source_finger_error = torch.where(source_is_left.unsqueeze(-1), left_finger_error, right_finger_error)
    arm_joint_error = torch.mean(torch.square(source_arm_error), dim=-1)
    finger_joint_error = torch.mean(torch.square(source_finger_error), dim=-1)
    total_weight = max(arm_weight + finger_weight, 1.0e-6)
    joint_error = (arm_weight * arm_joint_error + finger_weight * finger_joint_error) / total_weight

    dist_to_source = torch.norm(obj.data.root_pos_w - source_palm, p=2, dim=-1)
    released = dist_to_source > release_distance
    return torch.exp(-sharpness * joint_error) * released.float() * throw_gate


def object_close_to_source_palm_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sharpness: float = 15.0,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    _, source_palm, _, _, hold_gate = _source_receiver_palm_state(env, command_name, robot_cfg)
    dist = torch.norm(obj.data.root_pos_w - source_palm, p=2, dim=-1)
    return torch.exp(-sharpness * dist) * hold_gate


def object_low_speed_hold_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "handover",
    speed_scale: float = 8.0,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    hold_gate = (_handover_command(env, command_name)[:, 1] <= 0.0).float()
    speed = torch.norm(obj.data.root_lin_vel_w, p=2, dim=-1)
    return torch.exp(-speed_scale * speed) * hold_gate
