# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

BRAINCO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(__file__).resolve().parents[4] / "assets" / "usd" / "inhand" / "right_hand.usd"),
        activate_contact_sensors=True,  # Enable contact sensors for fingertips
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.00, 0.5),
        rot=(0.5, 0.5, -0.5, 0.5),
        joint_pos={
            # Thumb
            "right_thumb_CMP_joint": 0.43,
            "right_thumb_CMR_joint": 1.57,
            "right_thumb_MCP_joint": 0.0,
            "right_thumb_PIP_joint": 0.13,
            "right_thumb_DIP_joint": 0.18,
            # Index
            "right_index_MPR_joint": 0.0,
            "right_index_MCP_joint": 0.1,
            "right_index_PIP_joint": 0.15,
            "right_index_DIP_joint": 0.11,
            # Middle
            "right_middle_MPR_joint": 0.0,
            "right_middle_MCP_joint": 0.1,
            "right_middle_PIP_joint": 0.15,
            "right_middle_DIP_joint": 0.11,
            # Ring
            "right_ring_MPR_joint": 0,
            "right_ring_MCP_joint": 0.1,
            "right_ring_PIP_joint": 0.15,
            "right_ring_DIP_joint": 0.11,
            # Little
            "right_little_MPR_joint": 0.0,
            "right_little_MCP_joint": 0.1,
            "right_little_PIP_joint": 0.15,
            "right_little_DIP_joint": 0.11,
        },
    ),
    actuators={
        "brainco_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_.*_joint"],
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
