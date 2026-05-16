from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


REPO_ROOT = Path(__file__).resolve().parents[4]
ASSETS_DIR = REPO_ROOT / "assets"

REVOTRON_BIMANUAL_REVO3_USD = ASSETS_DIR / "usd/dynamic_handover/revotron_bimanual_revo3.usd"

RIGHT_ARM_JOINT_NAMES = [
    "Joint1_R",
    "Joint2_R",
    "Joint3_R",
    "Joint4_R",
    "Joint5_R",
    "Joint6_R",
    "Joint7_R",
]
LEFT_ARM_JOINT_NAMES = [
    "Joint1_L",
    "Joint2_L",
    "Joint3_L",
    "Joint4_L",
    "Joint5_L",
    "Joint6_L",
    "Joint7_L",
]
RIGHT_FINGER_JOINT_NAMES = [
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
]
LEFT_FINGER_JOINT_NAMES = [
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
]

RIGHT_ARM_DEFAULT_JOINT_POS = {
    "Joint1_R": -0.60,
    "Joint2_R": -0.80,
    "Joint3_R": -0.80,
    "Joint4_R": -1.20,
    "Joint5_R": 1.0,
    "Joint6_R": 0.00,
    "Joint7_R": 0.00,
}
LEFT_ARM_DEFAULT_JOINT_POS = {
    "Joint1_L": -0.60,
    "Joint2_L": -0.80,
    "Joint3_L": -0.80,
    "Joint4_L": -1.20,
    "Joint5_L": 1.0,
    "Joint6_L": 0.00,
    "Joint7_L": 0.00,
}
'''RIGHT_DEFAULT_FINGER_JOINT_POS = {
    "right_thumb_CMP_joint": 0.0,
    "right_thumb_CMR_joint": 1.3495253790945758,
    "right_thumb_MCP_joint": 0.8659920759388671,
    "right_thumb_PIP_joint": 0.780414711365591,
    "right_thumb_DIP_joint": 0.9655586519308622,
    "right_index_MPR_joint": -0.03989830748810656,
    "right_index_MCP_joint": 1.0139016439397597,
    "right_index_PIP_joint": 0.8501943208059994,
    "right_index_DIP_joint": 1.3264760744914152,
    "right_middle_MPR_joint": 0.0,
    "right_middle_MCP_joint": 1.347864170294202,
    "right_middle_PIP_joint": 0.6030536585610538,
    "right_middle_DIP_joint": 0.9181400800651911,
    "right_ring_MPR_joint": -0.20482272250532974,
    "right_ring_MCP_joint": 1.48,
    "right_ring_PIP_joint": 1.2686849760515697,
    "right_ring_DIP_joint": 0.8245164874462315,
    "right_little_MPR_joint": -0.21341465375119012,
    "right_little_MCP_joint": 0.0,
    "right_little_PIP_joint": 0.0,
    "right_little_DIP_joint": 0.0,
}
LEFT_DEFAULT_FINGER_JOINT_POS = {
    "left_thumb_CMP_joint": 0.0,
    "left_thumb_CMR_joint": 1.3495253790945758,
    "left_thumb_MCP_joint": 0.8659920759388671,
    "left_thumb_PIP_joint": 0.780414711365591,
    "left_thumb_DIP_joint": 0.9655586519308622,
    "left_index_MPR_joint": -0.03989830748810656,
    "left_index_MCP_joint": 1.0139016439397597,
    "left_index_PIP_joint": 0.8501943208059994,
    "left_index_DIP_joint": 1.3264760744914152,
    "left_middle_MPR_joint": 0.0,
    "left_middle_MCP_joint": 1.347864170294202,
    "left_middle_PIP_joint": 0.6030536585610538,
    "left_middle_DIP_joint": 0.9181400800651911,
    "left_ring_MPR_joint": -0.20482272250532974,
    "left_ring_MCP_joint": 1.48,
    "left_ring_PIP_joint": 1.2686849760515697,
    "left_ring_DIP_joint": 0.8245164874462315,
    "left_little_MPR_joint": -0.21341465375119012,
    "left_little_MCP_joint": 0.0,
    "left_little_PIP_joint": 0.0,
    "left_little_DIP_joint": 0.0,
}'''
RIGHT_DEFAULT_FINGER_JOINT_POS = {
    "right_thumb_CMP_joint": 1.2,
    "right_thumb_CMR_joint": 0.0,
    "right_thumb_MCP_joint": 0.5,
    "right_thumb_PIP_joint": 0.6,
    "right_thumb_DIP_joint": 0.3,
    "right_index_MPR_joint": 0.26,
    "right_index_MCP_joint": 1.0,
    "right_index_PIP_joint": 0.6,
    "right_index_DIP_joint": 0.3,
    "right_middle_MPR_joint": 0.26,
    "right_middle_MCP_joint": 1.0,
    "right_middle_PIP_joint": 0.6,
    "right_middle_DIP_joint": 0.3,
    "right_ring_MPR_joint": 0.26,
    "right_ring_MCP_joint": 1.0,
    "right_ring_PIP_joint": 0.6,
    "right_ring_DIP_joint": 0.3,
    "right_little_MPR_joint": 0.26,
    "right_little_MCP_joint": 1.0,
    "right_little_PIP_joint": 0.6,
    "right_little_DIP_joint": 0.3,
}
LEFT_DEFAULT_FINGER_JOINT_POS = {
    "left_thumb_CMP_joint": 1.2,
    "left_thumb_CMR_joint": 0.0,
    "left_thumb_MCP_joint": 0.5,
    "left_thumb_PIP_joint": 0.6,
    "left_thumb_DIP_joint": 0.3,
    "left_index_MPR_joint": 0.26,
    "left_index_MCP_joint": 1.0,
    "left_index_PIP_joint": 0.6,
    "left_index_DIP_joint": 0.3,
    "left_middle_MPR_joint": 0.26,
    "left_middle_MCP_joint": 1.0,
    "left_middle_PIP_joint": 0.6,
    "left_middle_DIP_joint": 0.3,
    "left_ring_MPR_joint": 0.26,
    "left_ring_MCP_joint": 1.0,
    "left_ring_PIP_joint": 0.6,
    "left_ring_DIP_joint": 0.3,
    "left_little_MPR_joint": 0.26,
    "left_little_MCP_joint": 1.0,
    "left_little_PIP_joint": 0.6,
    "left_little_DIP_joint": 0.3,
}
RIGHT_DEFAULT_JOINT_POS = {**RIGHT_ARM_DEFAULT_JOINT_POS, **RIGHT_DEFAULT_FINGER_JOINT_POS}
LEFT_DEFAULT_JOINT_POS = {**LEFT_ARM_DEFAULT_JOINT_POS, **LEFT_DEFAULT_FINGER_JOINT_POS}
ROBOT_DEFAULT_JOINT_POS = {**RIGHT_DEFAULT_JOINT_POS, **LEFT_DEFAULT_JOINT_POS}

RIGHT_CUBE_GRASP_FINGER_JOINT_POS = {
    "right_thumb_CMP_joint": -0.70,
    "right_thumb_CMR_joint": 1.00,
    "right_thumb_MCP_joint": 0.70,
    "right_thumb_PIP_joint": 0.90,
    "right_thumb_DIP_joint": 0.80,
    "right_index_MPR_joint": -0.05,
    "right_index_MCP_joint": 1.20,
    "right_index_PIP_joint": 1.25,
    "right_index_DIP_joint": 0.90,
    "right_middle_MPR_joint": 0.0,
    "right_middle_MCP_joint": 1.20,
    "right_middle_PIP_joint": 1.25,
    "right_middle_DIP_joint": 0.90,
    "right_ring_MPR_joint": 0.05,
    "right_ring_MCP_joint": 1.20,
    "right_ring_PIP_joint": 1.25,
    "right_ring_DIP_joint": 0.90,
    "right_little_MPR_joint": 0.10,
    "right_little_MCP_joint": 1.15,
    "right_little_PIP_joint": 1.20,
    "right_little_DIP_joint": 0.85,
}
LEFT_CUBE_GRASP_FINGER_JOINT_POS = {
    name.replace("right_", "left_", 1): value for name, value in RIGHT_CUBE_GRASP_FINGER_JOINT_POS.items()
}
RIGHT_PREGRASP_JOINT_POS = {**RIGHT_DEFAULT_JOINT_POS, **RIGHT_CUBE_GRASP_FINGER_JOINT_POS}
LEFT_PREGRASP_JOINT_POS = {**LEFT_DEFAULT_JOINT_POS, **LEFT_CUBE_GRASP_FINGER_JOINT_POS}
RIGHT_CUBE_GRASP_FINGER_JOINT_POS_TUPLE = tuple(
    RIGHT_CUBE_GRASP_FINGER_JOINT_POS[name] for name in RIGHT_FINGER_JOINT_NAMES
)
LEFT_CUBE_GRASP_FINGER_JOINT_POS_TUPLE = tuple(
    LEFT_CUBE_GRASP_FINGER_JOINT_POS[name] for name in LEFT_FINGER_JOINT_NAMES
)

RIGHT_ARM_SCALE = {name: 0.1 for name in RIGHT_ARM_JOINT_NAMES}
LEFT_ARM_SCALE = {name: 0.1 for name in LEFT_ARM_JOINT_NAMES}
RIGHT_ARM_EFFORT_LIMITS = {
    "Joint1_R": 300.0,
    "Joint2_R": 300.0,
    "Joint3_R": 300.0,
    "Joint4_R": 300.0,
    "Joint5_R": 100.0,
    "Joint6_R": 100.0,
    "Joint7_R": 100.0,
}
LEFT_ARM_EFFORT_LIMITS = {
    "Joint1_L": 300.0,
    "Joint2_L": 300.0,
    "Joint3_L": 300.0,
    "Joint4_L": 300.0,
    "Joint5_L": 100.0,
    "Joint6_L": 100.0,
    "Joint7_L": 100.0,
}
ARM_VELOCITY_LIMITS = {name: 3.1416 for name in RIGHT_ARM_JOINT_NAMES + LEFT_ARM_JOINT_NAMES}
RIGHT_ARM_STIFFNESS = {
    "Joint1_R": 100.0,
    "Joint2_R": 100.0,
    "Joint3_R": 64.0,
    "Joint4_R": 64.0,
    "Joint5_R": 48.0,
    "Joint6_R": 40.0,
    "Joint7_R": 40.0,
}
LEFT_ARM_STIFFNESS = {
    "Joint1_L": 100.0,
    "Joint2_L": 100.0,
    "Joint3_L": 64.0,
    "Joint4_L": 64.0,
    "Joint5_L": 48.0,
    "Joint6_L": 40.0,
    "Joint7_L": 40.0,
}
RIGHT_FINGER_EFFORT_LIMITS = {name: 5.0 for name in RIGHT_FINGER_JOINT_NAMES}
LEFT_FINGER_EFFORT_LIMITS = {name: 5.0 for name in LEFT_FINGER_JOINT_NAMES}
RIGHT_FINGER_VELOCITY_LIMITS = {name: 10.0 for name in RIGHT_FINGER_JOINT_NAMES}
LEFT_FINGER_VELOCITY_LIMITS = {name: 10.0 for name in LEFT_FINGER_JOINT_NAMES}

RIGHT_PALM_BODY_NAMES = ["right_palm", "right_hand_base_link", "palm", "base_link"]
LEFT_PALM_BODY_NAMES = ["left_palm", "left_hand_base_link", "palm", "base_link"]

DEFAULT_ROBOT_INIT_POS = (0.0, -0.575, -0.19)
DEFAULT_ROBOT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


def make_revotron_bimanual_revo3_cfg(
    prim_path: str,
    init_pos: tuple[float, float, float] = DEFAULT_ROBOT_INIT_POS,
    init_rot: tuple[float, float, float, float] = DEFAULT_ROBOT_INIT_ROT,
    usd_path: Path = REVOTRON_BIMANUAL_REVO3_USD,
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=ROBOT_DEFAULT_JOINT_POS,
        ),
        actuators={
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=RIGHT_ARM_JOINT_NAMES,
                stiffness=RIGHT_ARM_STIFFNESS,
                damping=1.0,
                effort_limit_sim=RIGHT_ARM_EFFORT_LIMITS,
                velocity_limit_sim={name: ARM_VELOCITY_LIMITS[name] for name in RIGHT_ARM_JOINT_NAMES},
            ),
            "right_fingers": ImplicitActuatorCfg(
                joint_names_expr=RIGHT_FINGER_JOINT_NAMES,
                stiffness=30.0,
                damping=1.0,
                effort_limit_sim=RIGHT_FINGER_EFFORT_LIMITS,
                velocity_limit_sim=RIGHT_FINGER_VELOCITY_LIMITS,
            ),
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=LEFT_ARM_JOINT_NAMES,
                stiffness=LEFT_ARM_STIFFNESS,
                damping=1.0,
                effort_limit_sim=LEFT_ARM_EFFORT_LIMITS,
                velocity_limit_sim={name: ARM_VELOCITY_LIMITS[name] for name in LEFT_ARM_JOINT_NAMES},
            ),
            "left_fingers": ImplicitActuatorCfg(
                joint_names_expr=LEFT_FINGER_JOINT_NAMES,
                stiffness=30.0,
                damping=1.0,
                effort_limit_sim=LEFT_FINGER_EFFORT_LIMITS,
                velocity_limit_sim=LEFT_FINGER_VELOCITY_LIMITS,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
