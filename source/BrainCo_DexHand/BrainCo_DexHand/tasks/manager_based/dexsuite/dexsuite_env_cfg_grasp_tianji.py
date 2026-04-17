# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .adr_curriculum import CurriculumCfg as BaseCurriculumCfg


TIANJI_PALM_BODY_NAME = "palm"


def reset_object_above_palm(
    env,
    env_ids: torch.Tensor | None,
    height: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=TIANJI_PALM_BODY_NAME),
):
    """Reset object root pose to be `height` meters above the Tianji palm (world +Z)."""
    obj = env.scene[object_cfg.name]
    robot = env.scene[palm_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=obj.device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.tensor(env_ids, device=obj.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=obj.device)

    body_ids, body_names = robot.find_bodies(palm_cfg.body_names, preserve_order=True)
    if len(body_ids) != 1:
        raise ValueError(
            f"Expected a single palm body for {palm_cfg.body_names}, but got {len(body_ids)}: {body_names}"
        )

    palm_pos_w = robot.data.body_pos_w[env_ids, body_ids[0], :]
    root_state = obj.data.default_root_state[env_ids].clone()
    offset = root_state.new_tensor([0.0, 0.0, height])
    root_state[:, 0:3] = palm_pos_w + offset

    obj.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:13]), env_ids=env_ids)


def object_palm_distance_exceeded(
    env,
    env_ids: torch.Tensor | None = None,
    threshold: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=TIANJI_PALM_BODY_NAME),
) -> torch.Tensor:
    """Terminate if object is farther than `threshold` meters from the Tianji palm."""
    obj = env.scene[object_cfg.name]
    robot = env.scene[palm_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=obj.device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.tensor(env_ids, device=obj.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=obj.device)

    body_ids, body_names = robot.find_bodies(palm_cfg.body_names, preserve_order=True)
    if len(body_ids) != 1:
        raise ValueError(
            f"Expected a single palm body for {palm_cfg.body_names}, but got {len(body_ids)}: {body_names}"
        )

    palm_pos_w = robot.data.body_pos_w[env_ids, body_ids[0], :]
    obj_pos_w = obj.data.root_pos_w[env_ids]
    dist = torch.linalg.norm(obj_pos_w - palm_pos_w, dim=-1)
    return dist > threshold


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dexsuite Scene for Tianji multi-object lifting."""

    robot: ArticulationCfg = MISSING

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                CuboidCfg(size=(0.06, 0.06, 0.06), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CapsuleCfg(radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # ConeCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                # ConeCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.1, 0.95)),
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.6, 0.76),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.98, 0.92, 0.95), metallic=0.8),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    marker_helper: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/marker_helper",
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command terms for the Tianji MDP."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.5),
            pos_y=(-0.15, 0.15),
            pos_z=(0.43, 0.55),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="marker_helper",
        success_visualizer_cfg=mdp.VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={},
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the Tianji MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        target_object_pose_b = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class StudentProprioObsCfg(ObsGroup):
        """Student proprioception observation group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PerceptionObsCfg(ObsGroup):
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
            params={"num_points": 64, "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()
    student_proprio: StudentProprioObsCfg = StudentProprioObsCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"scale_range": (1, 1), "asset_cfg": SceneEntityCfg("object")},
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.5, 2.0],
            "damping_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.0, 5.0],
            "operation": "scale",
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.2, 2.0],
            "operation": "scale",
        },
    )

    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.4, 0.5],
                "y": [-0.1, 0.1],
                "z": [0.0, 0.02],
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [-0.50, 0.50],
            "velocity_range": [0.0, 0.0],
        },
    )

    reset_robot_wrist_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="iiwa7_joint_7"),
            "position_range": [-3, 3],
            "velocity_range": [0.0, 0.0],
        },
    )

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -0], [0.0, 0.0, 0]),
            "operation": "abs",
        },
    )


@configclass
class TianjiGraspCurriculumCfg(BaseCurriculumCfg):
    """Curriculum terms scoped to the Tianji grasp/lift task."""

    gravity_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.variable_gravity.params.gravity_distribution_params",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                "difficulty_term_str": "adr",
            },
        },
    )


@configclass
class ActionsCfg:
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    fingers_to_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.4}, weight=1)
    fingers_to_object_delta = RewTerm(func=mdp.object_ee_distance_delta, params={"std": 0.4}, weight=200)

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.1,
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    # early_termination = RewTerm(func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.5, 1), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)

    non_finite = DoneTerm(func=mdp.non_finite_state)


@configclass
class DexsuiteReorientEnvCfg(ManagerBasedEnvCfg):
    """Tianji dexsuite reorientation task base definition."""

    viewer: ViewerCfg = ViewerCfg(eye=(-2.25, 0.0, 1.35), lookat=(0.0, 0.0, 0.95), origin_type="env")
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: TianjiGraspCurriculumCfg | None = TianjiGraspCurriculumCfg()

    def __post_init__(self):
        self.decimation = 2

        self.commands.object_pose.resampling_time_range = (10.0, 10.0)
        self.commands.object_pose.position_only = False
        self.commands.object_pose.success_visualizer_cfg.markers["failure"] = sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1), opacity=0.6),
        )
        self.commands.object_pose.success_visualizer_cfg.markers["success"] = sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1), opacity=0.6),
        )

        self.episode_length_s = 4.0
        self.is_finite_horizon = True

        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15

        if self.curriculum is not None:
            self.curriculum.adr.params["pos_tol"] = self.rewards.success.params["pos_std"] / 2
            self.curriculum.adr.params["rot_tol"] = self.rewards.success.params["rot_std"] / 2


class DexsuiteLiftEnvCfg(DexsuiteReorientEnvCfg):
    """Tianji dexsuite lift task definition."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.orientation_tracking = None
        self.commands.object_pose.position_only = True
        if self.curriculum is not None:
            self.rewards.success.params["rot_std"] = None
            self.curriculum.adr.params["rot_tol"] = None


class DexsuiteReorientEnvCfg_PLAY(DexsuiteReorientEnvCfg):
    """Tianji dexsuite reorientation evaluation environment definition."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]


class DexsuiteLiftEnvCfg_PLAY(DexsuiteLiftEnvCfg):
    """Tianji dexsuite lift evaluation environment definition."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.position_only = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]
