# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Revo3 hand dexsuite grasp environments."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from BrainCo_DexHand.assets.revotron_single_right_revo3 import REVOTRON_SINGLE_RIGHT_REVO3_CFG
from BrainCo_DexHand.assets.revotron_bimanual_revo3 import RIGHT_ARM_JOINT_NAMES, RIGHT_FINGER_JOINT_NAMES

from ... import dexsuite_env_cfg_grasp_tianji as dexsuite
from ... import mdp


TIANJI_PALM_BODY_NAME = dexsuite.TIANJI_PALM_BODY_NAME
TIANJI_HAND_DIP_BODIES = [
    "right_little_DIP_link",
    "right_ring_DIP_link",
    "right_middle_DIP_link",
    "right_index_DIP_link",
    "right_thumb_DIP_link",
]
TIANJI_HAND_TIP_BODIES = [
    TIANJI_PALM_BODY_NAME,
    "right_little_tip_Link",
    "right_ring_tip_Link",
    "right_middle_tip_Link",
    "right_index_tip_Link",
    "right_thumb_tip_Link",
]


@configclass
class Revo3RelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=RIGHT_ARM_JOINT_NAMES + RIGHT_FINGER_JOINT_NAMES,
        preserve_order=True, scale=0.1,
    )


def tianji_hand_contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Thumb plus at least one other fingertip contact for the Tianji hand."""
    thumb_contact_sensor: ContactSensor = env.scene.sensors["right_thumb_DIP_link_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["right_index_DIP_link_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["right_middle_DIP_link_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["right_ring_DIP_link_object_s"]
    little_contact_sensor: ContactSensor = env.scene.sensors["right_little_DIP_link_object_s"]

    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    little_contact = little_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    little_contact_mag = torch.norm(little_contact, dim=-1)

    return (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold)
        | (middle_contact_mag > threshold)
        | (ring_contact_mag > threshold)
        | (little_contact_mag > threshold)
    )


def tianji_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Position tracking reward gated by Tianji hand contact."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import combine_frame_transforms

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(obj.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * tianji_hand_contacts(env, 1.0).float()


def tianji_orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Orientation tracking reward gated by Tianji hand contact."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import quat_error_magnitude, quat_mul

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    quat_error = quat_error_magnitude(obj.data.root_quat_w, des_quat_w)
    return (1 - torch.tanh(quat_error / std)) * tianji_hand_contacts(env, 1.0).float()


@configclass
class Revo3ReorientRewardCfg(dexsuite.RewardsCfg):
    """Reward configuration dedicated to Revo3 lift/reorient tasks."""

    position_tracking = RewTerm(
        func=tianji_position_command_error_tanh,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    orientation_tracking = RewTerm(
        func=tianji_orientation_command_error_tanh,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class Revo3MixinCfg:
    """Mixin that attaches the public Revo3 robot asset."""

    rewards: Revo3ReorientRewardCfg = Revo3ReorientRewardCfg()
    actions: Revo3RelJointPosActionCfg = Revo3RelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()

        self.commands.object_pose.body_name = TIANJI_PALM_BODY_NAME
        self.commands.object_pose.debug_vis = True

        self.scene.robot = REVOTRON_SINGLE_RIGHT_REVO3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        base_joint_pos = dict(REVOTRON_SINGLE_RIGHT_REVO3_CFG.init_state.joint_pos) if (
            hasattr(REVOTRON_SINGLE_RIGHT_REVO3_CFG.init_state, "joint_pos")
            and REVOTRON_SINGLE_RIGHT_REVO3_CFG.init_state.joint_pos is not None
        ) else {}
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
            pos=(1, 0.0, 0.58),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=base_joint_pos,
        )

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["Joint[1-6]_R", "right_.*_joint"]),
                "position_range": [0.0, 0.0],
                "velocity_range": [0.0, 0.0],
            },
        )

        self.events.reset_robot_wrist_joint = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="Joint7_R"),
                "position_range": [0.0, 0.0],
                "velocity_range": [0.0, 0.0],
            },
        )

        for link_name in TIANJI_HAND_DIP_BODIES:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in TIANJI_HAND_DIP_BODIES]},
            clip=(-20.0, 20.0),
        )

        self.observations.proprio.hand_tips_state_b.params["frame_names"] = TIANJI_HAND_TIP_BODIES

        if hasattr(self.rewards, "fingers_to_object"):
            self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
                "robot",
            )
            self.rewards.fingers_to_object.params["frame_names"] = TIANJI_HAND_TIP_BODIES
        if hasattr(self.rewards, "fingers_to_object_delta"):
            self.rewards.fingers_to_object_delta.params["asset_cfg"] = SceneEntityCfg(
                "robot",
            )

            self.rewards.fingers_to_object_delta.params["frame_names"] = TIANJI_HAND_TIP_BODIES


@configclass
class DexsuiteRevo3LiftEnvCfg(Revo3MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Configuration for Revo3 lift environment (training)."""

    pass


@configclass
class DexsuiteRevo3LiftEnvCfg_PLAY(Revo3MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Configuration for Revo3 lift environment (evaluation/play)."""

    pass
