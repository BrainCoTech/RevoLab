# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from BrainCo_DexHand.assets import BRAINCO_CFG


@configclass
class BrainCoHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    action_space = 21
    observation_space = 147  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=2**20,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = BRAINCO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    actuated_joint_names = [
        # Thumb
        "right_thumb_CMP_joint",
        "right_thumb_CMR_joint",
        "right_thumb_MCP_joint",
        "right_thumb_PIP_joint",
        "right_thumb_DIP_joint",
        # Index
        "right_index_MPR_joint",
        "right_index_MCP_joint",
        "right_index_PIP_joint",
        "right_index_DIP_joint",
        # Middle
        "right_middle_MPR_joint",
        "right_middle_MCP_joint",
        "right_middle_PIP_joint",
        "right_middle_DIP_joint",
        # Ring
        "right_ring_MPR_joint",
        "right_ring_MCP_joint",
        "right_ring_PIP_joint",
        "right_ring_DIP_joint",
        # Little
        "right_little_MPR_joint",
        "right_little_MCP_joint",
        "right_little_PIP_joint",
        "right_little_DIP_joint",
    ]
    fingertip_body_names = [
        'right_little_DIP_Link',
        'right_ring_DIP_Link',
        'right_middle_DIP_Link',
        'right_index_DIP_Link',
        'right_thumb_DIP_Link',
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1, 1, 1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.11, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1, 1, 1),
            ),
            "target_dot": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
