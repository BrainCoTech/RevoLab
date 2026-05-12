from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from BrainCo_DexHand.assets.revotron_bimanual_revo3 import (
    ASSETS_DIR,
    DEFAULT_ROBOT_INIT_POS,
    DEFAULT_ROBOT_INIT_ROT,
    LEFT_FINGER_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    LEFT_ARM_SCALE,
    RIGHT_FINGER_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_ARM_SCALE,
    make_revotron_bimanual_revo3_cfg,
)
from . import mdp

TRAINING_OBJECTS = [
    ASSETS_DIR / "urdf/objects/cube_multicolor.urdf",
    ASSETS_DIR / "urdf/objects/cube_multicolor1.urdf",
    ASSETS_DIR / "urdf/objects/cube_goal_multicolor.urdf",
]


def _robot_relative_pos(offset: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(DEFAULT_ROBOT_INIT_POS[i] + offset[i] for i in range(3))


# The actual initial palm spacing is cached from the live single-USD articulation during reset.
INITIAL_HAND_DISTANCE = None
OBJECT_BASE_POS = _robot_relative_pos((0.04, 0.0, 1.34))
GOAL_BASE_POS = _robot_relative_pos((0.10, 0.0, 1.33))
HANDOVER_SOURCE_DISTANCE_THRESHOLD = 0.06
HANDOVER_CATCH_DISTANCE_THRESHOLD = 0.10
HANDOVER_RELEASE_DISTANCE = 0.10
HANDOVER_RETURN_RELEASE_DISTANCE = 0.12
HANDOVER_GOAL_DISTANCE_THRESHOLD = 0.08
HANDOVER_PALM_DISTANCE_MARGIN = 0.05
HANDOVER_Y_CORRIDOR_MIN_PROGRESS = 0.15
HANDOVER_Y_CORRIDOR_MAX_PROGRESS = 0.85


@configclass
class DynamicHandoverSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = make_revotron_bimanual_revo3_cfg(
        "{ENV_REGEX_NS}/Robot",
        DEFAULT_ROBOT_INIT_POS,
        DEFAULT_ROBOT_INIT_ROT,
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.UrdfFileCfg(
                    asset_path=str(asset_path),
                    fix_base=False,
                    merge_fixed_joints=True,
                    make_instanceable=False,
                    joint_drive=None,
                    collision_from_visuals=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        enable_gyroscopic_forces=True,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                        max_depenetration_velocity=1000.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(density=500.0),
                )
                for asset_path in TRAINING_OBJECTS
            ],
            random_choice=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_BASE_POS),
    )

    goal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/goal",
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.45, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=GOAL_BASE_POS),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


@configclass
class DynamicHandoverCubeSceneCfg(DynamicHandoverSceneCfg):
    robot: ArticulationCfg = make_revotron_bimanual_revo3_cfg(
        "{ENV_REGEX_NS}/Robot",
        DEFAULT_ROBOT_INIT_POS,
        DEFAULT_ROBOT_INIT_ROT,
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_BASE_POS),
    )


@configclass
class ActionsCfg:
    right_arm = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        scale=RIGHT_ARM_SCALE,
        preserve_order=True,
    )
    right_fingers = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=RIGHT_FINGER_JOINT_NAMES,
        rescale_to_limits=True,
    )
    left_arm = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEFT_ARM_JOINT_NAMES,
        scale=LEFT_ARM_SCALE,
        preserve_order=True,
    )
    left_fingers = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=LEFT_FINGER_JOINT_NAMES,
        rescale_to_limits=True,
    )


@configclass
class CommandsCfg:
    handover = mdp.BimanualHandoverCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
    )


@configclass
class CubeCommandsCfg(CommandsCfg):
    handover = mdp.BimanualHandoverCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        rel_throw_envs=1.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        right_joint_pos = ObsTerm(func=mdp.joint_pos_subset_rel, params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": tuple(RIGHT_ARM_JOINT_NAMES + RIGHT_FINGER_JOINT_NAMES)})
        right_joint_vel = ObsTerm(func=mdp.joint_vel_subset_rel, scale=0.2, params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": tuple(RIGHT_ARM_JOINT_NAMES + RIGHT_FINGER_JOINT_NAMES)})
        left_joint_pos = ObsTerm(func=mdp.joint_pos_subset_rel, params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": tuple(LEFT_ARM_JOINT_NAMES + LEFT_FINGER_JOINT_NAMES)})
        left_joint_vel = ObsTerm(func=mdp.joint_vel_subset_rel, scale=0.2, params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": tuple(LEFT_ARM_JOINT_NAMES + LEFT_FINGER_JOINT_NAMES)})
        object_pos = ObsTerm(func=mdp.object_pos_in_handover_frame, params={"command_name": "handover"})
        goal_pos = ObsTerm(func=mdp.goal_pos_in_handover_frame, params={"command_name": "handover"})
        object_lin_vel = ObsTerm(func=mdp.object_lin_vel_in_handover_frame, scale=0.2, params={"command_name": "handover"})
        object_ang_vel = ObsTerm(func=mdp.object_ang_vel_in_handover_frame, scale=0.2, params={"command_name": "handover"})
        right_palm_pos = ObsTerm(func=mdp.right_palm_pos_in_robot_root)
        left_palm_pos = ObsTerm(func=mdp.left_palm_pos_in_robot_root)
        handover_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "handover"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    prepare_handover = EventTerm(func=mdp.prepare_handover_reset_command, mode="reset")
    reset_fingers = EventTerm(
        func=mdp.reset_fingers_by_handover_command,
        mode="reset",
        params={
            "command_name": "handover",
            "right_finger_joint_names": tuple(RIGHT_FINGER_JOINT_NAMES),
            "left_finger_joint_names": tuple(LEFT_FINGER_JOINT_NAMES),
        },
    )
    reset_object = EventTerm(
        func=mdp.reset_object_by_handover_command,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "command_name": "handover",
            "world_pos_offset": (0.04, 0.0, 0.10),
            "mirror_world_x_by_source": False,
        },
    )
    reset_goal = EventTerm(
        func=mdp.reset_goal_by_handover_command,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("goal"),
            "command_name": "handover",
            "local_pos": (0.0, 0.0, 0.0),
            "world_pos_offset": (0.10, 0.0, 0.04),
            "world_pos_offset_in_body_frame": False,
            "mirror_world_x_by_source": False,
            "local_quat": (1.0, 0.0, 0.0, 0.0),
            "position_noise": 0.0,
        },
    )


@configclass
class CubeEventCfg(EventCfg):
    reset_object = EventTerm(
        func=mdp.reset_object_by_handover_command,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "command_name": "handover",
            "local_pos": (0.0, 0.0, 0.0),
            "world_pos_offset": (0.04, 0.0, 0.010),
            "mirror_world_x_by_source": False,
            "local_quat": (1.0, 0.0, 0.0, 0.0),
            "position_noise": 0.0,
        },
    )


@configclass
class RewardsCfg:
    goal_tracking = RewTerm(func=mdp.exp_goal_tracking, weight=1.0, params={"sharpness": 12.0, "command_name": "handover"})
    leave_source_palm = RewTerm(
        func=mdp.object_away_from_source_palm_penalty,
        weight=-2.0,
        params={"palm_distance_threshold": HANDOVER_SOURCE_DISTANCE_THRESHOLD, "command_name": "handover"},
    )
    towards_receiver_palm_velocity = RewTerm(
        func=mdp.object_towards_receiver_palm_velocity_bonus, weight=0.7, params={"command_name": "handover"}
    )
    close_to_receiver_palm = RewTerm(
        func=mdp.object_close_to_receiver_palm_reward, weight=0.75, params={"sharpness": 10.0, "command_name": "handover"}
    )
    receiver_palm_towards_object = RewTerm(
        func=mdp.receiver_palm_towards_object_bonus,
        weight=0.25,
        params={"command_name": "handover", "release_distance": HANDOVER_RELEASE_DISTANCE},
    )
    receiver_finger_contacts = RewTerm(
        func=mdp.receiver_multi_finger_surface_contacts_reward,
        weight=0.7,
        params={"command_name": "handover", "surface_threshold": 0.028, "min_fingers": 2},
    )
    object_y_velocity = RewTerm(
        func=mdp.object_y_velocity_bonus,
        weight=1.0,
        params={
            "command_name": "handover",
            "corridor_min_progress": HANDOVER_Y_CORRIDOR_MIN_PROGRESS,
            "corridor_max_progress": HANDOVER_Y_CORRIDOR_MAX_PROGRESS,
        },
    )
    source_arm_return = RewTerm(
        func=mdp.source_arm_return_to_default_reward,
        weight=1.0,
        params={
            "command_name": "handover",
            "release_distance": HANDOVER_RETURN_RELEASE_DISTANCE,
            "sharpness": 4.0,
            "arm_weight": 4.0,
            "finger_weight": 1.5,
        },
    )
    hand_separation = RewTerm(
        func=mdp.hand_palm_separation_penalty,
        weight=-5.0,
        params={"initial_distance": INITIAL_HAND_DISTANCE, "distance_margin": HANDOVER_PALM_DISTANCE_MARGIN},
    )
    
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0002)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_drop = DoneTerm(func=mdp.object_below_height, params={"minimum_height": 0.15})


@configclass
class DynamicHandoverEnvCfg(ManagerBasedRLEnvCfg):
    scene: DynamicHandoverSceneCfg = DynamicHandoverSceneCfg(
        num_envs=2048,
        env_spacing=3.0,
        clone_in_fabric=True,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 3
        self.episode_length_s = 0.75
        self.viewer.eye = (2.4, -2.7, 1.9)
        self.viewer.lookat = (0.0, -0.68, 1.1)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 8
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.enable_ccd = True
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2**23


@configclass
class DynamicHandoverEnvCfg_PLAY(DynamicHandoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class DynamicHandoverCubeEnvCfg(DynamicHandoverEnvCfg):
    scene: DynamicHandoverCubeSceneCfg = DynamicHandoverCubeSceneCfg(
        num_envs=2048,
        env_spacing=3.0,
        clone_in_fabric=False,
        replicate_physics=False,
    )
    commands: CubeCommandsCfg = CubeCommandsCfg()
    events: CubeEventCfg = CubeEventCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class DynamicHandoverCubeEnvCfg_PLAY(DynamicHandoverCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
