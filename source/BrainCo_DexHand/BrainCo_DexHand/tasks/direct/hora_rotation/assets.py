"""Asset configs for Revo3 right hand in-hand rotation."""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

_REPO_ROOT = Path(__file__).resolve().parents[6]
_REVO3_USD = str(_REPO_ROOT / "usd" / "hora" / "revo3_right.usd")

# Object initial positions in env-local coordinates.
OBJECT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
# Hand initial pose (-25 deg around world X-axis)
HAND_INIT_POS = (0.0, 0.0, 1.5)
HAND_INIT_ROT = (0.59636781, 0.37992820, -0.37992820, 0.59636781)

CYLINDER_INIT_POS = (0.000, -0.08, 1.635)
BALL_INIT_POS = (0.000, -0.08, 1.645)

REVO3_HAND_CYLINDER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/hand",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_REVO3_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=HAND_INIT_POS,
        rot=HAND_INIT_ROT,
        joint_pos={
            "right_thumb_CMP_joint":  1.65, "right_thumb_CMR_joint":  1.35,
            "right_thumb_MCP_joint":  0.35, "right_thumb_PIP_joint":  0.20,
            "right_thumb_DIP_joint":  0.00,
            "right_index_MPR_joint": -0.25, "right_index_MCP_joint":  1.20,
            "right_index_PIP_joint":  0.30, "right_index_DIP_joint":  0.00,
            "right_middle_MPR_joint": 0.00, "right_middle_MCP_joint": 0.95,
            "right_middle_PIP_joint": 0.20, "right_middle_DIP_joint": 0.00,
            "right_ring_MPR_joint":   0.20, "right_ring_MCP_joint":   0.95,
            "right_ring_PIP_joint":   0.20, "right_ring_DIP_joint":   0.00,
            "right_little_MPR_joint": 0.25, "right_little_MCP_joint": 1.20,
            "right_little_PIP_joint": 0.30, "right_little_DIP_joint": 0.00,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_.*"],
            effort_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.01,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

REVO3_HAND_BALL_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/hand",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_REVO3_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=HAND_INIT_POS,
        rot=HAND_INIT_ROT,
        joint_pos={
            "right_thumb_CMP_joint":  1.65, "right_thumb_CMR_joint":  1.35,
            "right_thumb_MCP_joint":  0.50, "right_thumb_PIP_joint":  0.00,
            "right_thumb_DIP_joint":  0.00,
            "right_index_MPR_joint": -0.25, "right_index_MCP_joint":  1.40,
            "right_index_PIP_joint":  0.00, "right_index_DIP_joint":  0.00,
            "right_middle_MPR_joint": 0.00, "right_middle_MCP_joint": 1.05,
            "right_middle_PIP_joint": 0.05, "right_middle_DIP_joint": 0.00,
            "right_ring_MPR_joint":   0.20, "right_ring_MCP_joint":   1.10,
            "right_ring_PIP_joint":   0.00, "right_ring_DIP_joint":   0.00,
            "right_little_MPR_joint": 0.20, "right_little_MCP_joint": 1.40,
            "right_little_PIP_joint": 0.00, "right_little_DIP_joint": 0.00,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_.*"],
            effort_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.01,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

_COMMON_RIGID = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=0,
    sleep_threshold=0.005,
    stabilization_threshold=0.0025,
    max_depenetration_velocity=1000.0,
)
_COMMON_MASS = sim_utils.MassPropertiesCfg(mass=0.10)
_COMMON_COLLISION = sim_utils.CollisionPropertiesCfg(
    collision_enabled=True, contact_offset=0.002, rest_offset=0.0)
_COMMON_MATERIAL = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)


BALL_OBJECT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/object",
    spawn=sim_utils.SphereCfg(
        radius=0.030,
        rigid_props=_COMMON_RIGID,
        mass_props=_COMMON_MASS,
        collision_props=_COMMON_COLLISION,
        physics_material=_COMMON_MATERIAL,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_INIT_POS, rot=OBJECT_INIT_ROT),
)

CYLINDER_OBJECT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/object",
    spawn=sim_utils.CylinderCfg(
        radius=0.03, height=0.070,
        rigid_props=_COMMON_RIGID,
        mass_props=_COMMON_MASS,
        collision_props=_COMMON_COLLISION,
        physics_material=_COMMON_MATERIAL,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=CYLINDER_INIT_POS, rot=OBJECT_INIT_ROT),
)
