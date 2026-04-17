# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_ee_distance_delta(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_offset: tuple[float, float, float] | None = None,
    body_name_to_offset: str = "base",
    clamp_max: float = 10.0,
    history_key: str = "_closest_ee_distance",
) -> torch.Tensor:
    """Reward improvement over the historical closest end-effector distance.

    Computes the distance between the object and the end-effector bodies (max over bodies),
    then rewards only when the current distance is smaller than the historical best.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()
    asset_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]

    if body_offset is not None:
        offset_tensor = torch.tensor(body_offset, device=env.device, dtype=asset_pos.dtype)
        for i, body_id in enumerate(asset_cfg.body_ids):
            body_name = asset.body_names[body_id]
            if body_name == body_name_to_offset:
                offset_world = quat_apply(asset_quat[:, i], offset_tensor.unsqueeze(0).expand(env.num_envs, -1))
                asset_pos[:, i] = asset_pos[:, i] + offset_world
                break

    object_pos = object.data.root_pos_w
    curr_dist = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values

    buf_name = history_key
    closest = getattr(env, buf_name, None)
    if closest is None or closest.shape != curr_dist.shape:
        closest = curr_dist.clone()

    # Reset per-episode buffers on first step of each episode.
    # IsaacLab increments episode_length_buf before reward computation, so the
    # first reward step is typically 1 (not 0). Use <= 1 for compatibility.
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_mask = env.episode_length_buf <= 1
        if torch.any(reset_mask):
            closest = closest.clone()
            closest[reset_mask] = curr_dist[reset_mask]

    delta = (closest - curr_dist).clamp(min=0.0, max=clamp_max)
    new_closest = torch.minimum(closest, curr_dist)
    setattr(env, buf_name, new_closest)

    # normalize by std for scale compatibility
    return delta / (std + 1e-6)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_offset: tuple[float, float, float] | None = None,
    body_name_to_offset: str = "base",
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()
    asset_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]

    if body_offset is not None:
        offset_tensor = torch.tensor(body_offset, device=env.device, dtype=asset_pos.dtype)
        for i, body_id in enumerate(asset_cfg.body_ids):
            body_name = asset.body_names[body_id]
            if body_name == body_name_to_offset:
                offset_world = quat_apply(asset_quat[:, i], offset_tensor.unsqueeze(0).expand(env.num_envs, -1))
                asset_pos[:, i] = asset_pos[:, i] + offset_world
                break

    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)


def object_ee_distance_mean(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_offset: tuple[float, float, float] | None = None,
    body_name_to_offset: str = "base",
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on mean end-effector distance.

    The reward is close to 1 when the average distance between the object and end-effector bodies is small.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()
    asset_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]

    if body_offset is not None:
        offset_tensor = torch.tensor(body_offset, device=env.device, dtype=asset_pos.dtype)
        for i, body_id in enumerate(asset_cfg.body_ids):
            body_name = asset.body_names[body_id]
            if body_name == body_name_to_offset:
                offset_world = quat_apply(asset_quat[:, i], offset_tensor.unsqueeze(0).expand(env.num_envs, -1))
                asset_pos[:, i] = asset_pos[:, i] + offset_world
                break

    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).mean(dim=-1)
    return 1 - torch.tanh(object_ee_distance / std)


def any_finger_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
) -> torch.Tensor:
    """Reward any fingertip contact with the object above threshold.

    Supports both legacy LEAP sensor names and Arm-BrainCo `right_*` sensor names.
    """

    sensor_names = [
        # Legacy LEAP names
        "thumb_fingertip_object_s",
        "fingertip_object_s",
        "fingertip_2_object_s",
        "fingertip_3_object_s",
        "fingertip_4_object_s",
        # Arm-BrainCo names
        "right_thumb_DIP_Link_object_s",
        "right_index_DIP_Link_object_s",
        "right_middle_DIP_Link_object_s",
        "right_ring_DIP_Link_object_s",
        "right_little_DIP_Link_object_s",
    ]

    contact_magnitudes = []
    for sensor_name in sensor_names:
        if sensor_name in env.scene.sensors:
            contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
            contact_force = contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
            contact_mag = torch.norm(contact_force, dim=-1)
            contact_magnitudes.append(contact_mag)
            
            # Print when force is detected (only for env 0 to avoid too many prints)
            # if contact_mag[0].item() > threshold:
            #     force_value = contact_mag[0].item()
            #     finger_name = sensor_name.replace('_object_s', '').replace('_', ' ').title()
            #     print(f"[CONTACT] {finger_name} detected force: {force_value:.3f} N (threshold: {threshold:.3f} N)")

    if not contact_magnitudes:
        return torch.zeros(env.num_envs, device=env.device)

    all_contacts = torch.stack(contact_magnitudes, dim=1)
    any_contact = (all_contacts > threshold).any(dim=1)
    return any_contact.float()


def contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""

    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_link_3_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_link_3_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_link_3_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_link_3_object_s"]
    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    good_contact_cond1 = (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
    )

    return good_contact_cond1


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        # square is not necessary but this help to keep the final value between having rot_std or not roughly the same
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = torch.norm(rot_err, dim=1)
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, 1.0).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, 1.0).float()
