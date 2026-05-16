"""Environment config for Revo3 right hand in-hand rotation."""
from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

_REPO_ROOT = Path(__file__).resolve().parents[6]
_REVO3_USD = str(_REPO_ROOT / "assets" / "usd" / "hora" / "revo3_right.usd")


@configclass
class Revo3HandHoraEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    action_space = 21
    observation_space = 141  # 3 frames x 47 dims (21 joint_pos + 21 targets + 5 contacts)
    prop_hist_len = 30
    priv_info_dim = 8
    state_space = 0
    asymmetric_obs = False
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    torque_control = True
    pgain: float = 2.0
    dgain: float = 0.2

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240, render_interval=2,
        gravity=(0.0, 0.0, -0.05),
        physx=PhysxCfg(
            solver_type=1, max_position_iteration_count=8, max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608, gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )

    hand_init_pose = ((0.0, 0.0, 1.5), (0.59636781, 0.37992820, -0.37992820, 0.59636781))
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_REVO3_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0, max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                sleep_threshold=0.005, stabilization_threshold=0.0005, fix_root_link=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=hand_init_pose[0], rot=hand_init_pose[1],
        ),
        actuators={
            "fingers": IdealPDActuatorCfg(
                joint_names_expr=["right_.*"], stiffness=None, damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    actuated_joint_names = [
        "right_thumb_CMP_joint", "right_thumb_CMR_joint",
        "right_thumb_MCP_joint", "right_thumb_PIP_joint", "right_thumb_DIP_joint",
        "right_index_MPR_joint", "right_index_MCP_joint",
        "right_index_PIP_joint", "right_index_DIP_joint",
        "right_middle_MPR_joint", "right_middle_MCP_joint",
        "right_middle_PIP_joint", "right_middle_DIP_joint",
        "right_ring_MPR_joint", "right_ring_MCP_joint",
        "right_ring_PIP_joint", "right_ring_DIP_joint",
        "right_little_MPR_joint", "right_little_MCP_joint",
        "right_little_PIP_joint", "right_little_DIP_joint",
    ]
    fingertip_body_names = [
        "right_thumb_DIP_Link", "right_index_DIP_Link",
        "right_middle_DIP_Link", "right_ring_DIP_Link", "right_little_DIP_Link",
    ]
    elastomer_body_names = [
        "right_thumb_DIP_Link", "right_index_DIP_Link",
        "right_middle_DIP_Link", "right_ring_DIP_Link", "right_little_DIP_Link",
    ]
    contact_sensor = []

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.CylinderCfg(
            radius=0.03, height=0.070,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                sleep_threshold=0.005, stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.000, -0.08, 1.635), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16384, env_spacing=0.75, replicate_physics=False)

    reset_height_lower = 1.615
    reset_height_upper = 1.655
    reset_angle_diff = 45 / 180 * math.pi
    reset_random_quat = False

    rot_axis = (0, 0, 1)
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5
    rotate_reward_scale = 2.5
    object_linvel_penalty_scale = -0.3
    pos_diff_penalty_scale = -0.4
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5
    object_pos_reward_scale = 0.003

    grasp_cache_path = 'assets/grasp_cache/hora/revo3_right_grasp_cylinder'

    joint_noise_scale = 0.02
    enable_tactile = True
    enable_contact_in_obs = True   # Stage2 sets False: actor sees zero contact, adapt_tconv retains contact history
    binary_contact = False
    enable_contact_pos = False
    disable_tactile_ids = []
    contact_smooth = 0.5
    contact_threshold = 0.05
    contact_latency = 0.005
    contact_sensor_noise = 0.01
    dof_limits_scale = 0.9

    randomize_pd_gains = True
    randomize_p_gain_scale_lower = 0.5
    randomize_p_gain_scale_upper = 2
    randomize_d_gain_scale_lower = 0.5
    randomize_d_gain_scale_upper = 2
    randomize_friction = True
    randomize_friction_scale_lower = 0.5
    randomize_friction_scale_upper = 2.0
    elastomer_base_friction = 0.8
    metal_base_friction = 0.1
    object_base_friction = 0.5
    randomize_com = True
    randomize_com_lower = -0.01
    randomize_com_upper = 0.01
    randomize_mass = True
    randomize_mass_lower = 0.01
    randomize_mass_upper = 0.20

    force_scale = 2
    random_force_prob_scalar = 0.25
    force_decay = 0.9
    force_decay_interval = 0.08

    gravity_curriculum = True
    debug_show_axes = False

    def __post_init__(self):
        super().__post_init__()
        self.contact_sensor = []
        for name in self.elastomer_body_names:
            self.contact_sensor.append(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/hand/{name}",
                history_length=3,
                track_contact_points=True,
                filter_prim_paths_expr=["/World/envs/env_.*/object"],
            ))


@configclass
class Revo3HandHoraCylinderEnvCfg(Revo3HandHoraEnvCfg):
    """Cylinder variant for Gym registration."""

    def __post_init__(self):
        super().__post_init__()
        from .assets import CYLINDER_OBJECT_CFG, REVO3_HAND_CYLINDER_CFG

        self.robot_cfg = REVO3_HAND_CYLINDER_CFG
        self.object_cfg = CYLINDER_OBJECT_CFG
        self.grasp_cache_path = "assets/grasp_cache/hora/revo3_right_grasp_cylinder"


@configclass
class Revo3HandHoraBallEnvCfg(Revo3HandHoraEnvCfg):
    """Ball variant for Gym registration."""

    def __post_init__(self):
        super().__post_init__()
        from .assets import BALL_OBJECT_CFG, REVO3_HAND_BALL_CFG

        self.robot_cfg = REVO3_HAND_BALL_CFG
        self.object_cfg = BALL_OBJECT_CFG
        self.grasp_cache_path = "assets/grasp_cache/hora/revo3_right_grasp_ball"
