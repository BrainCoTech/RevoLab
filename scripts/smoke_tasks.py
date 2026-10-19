#!/usr/bin/env python3
"""Reset and step a v1 task without loading a policy or training."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "BrainCo_DexHand"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--check_rsl_runner", action="store_true", help="Construct the configured RSL runner and run inference, without learning.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import BrainCo_DexHand.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from BrainCo_DexHand.assets.description import LOCKED_REVOTRON_JOINTS
from BrainCo_DexHand.assets.frames import frame_state_w


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "Non-finite environment output"


def observation_shapes(value, space):
    if isinstance(space, gym.spaces.Dict):
        assert set(value) == set(space.spaces), "Observation groups disagree with the declared space"
        for key, item in value.items():
            observation_shapes(item, space[key])
    else:
        expected = (args.num_envs, *space.shape)
        assert tuple(value.shape) == expected, f"Observation shape {tuple(value.shape)} != declared {expected}"


def check_reset_scene(base, task):
    """Catch frame changes that preserve joint names but rotate the task workspace."""
    from isaaclab.utils.math import quat_apply
    robot = base.hand if hasattr(base, "hand") else base.scene["robot"]
    right = frame_state_w(robot, "right_palm")
    x_axis = right.new_tensor([1.0, 0.0, 0.0]).expand(base.num_envs, -1)
    normal = quat_apply(right[:, 3:7], x_axis)
    palm = right[:, :3] - base.scene.env_origins
    if "Handover" in task:
        left = frame_state_w(robot, "left_palm")
        left_normal = quat_apply(left[:, 3:7], x_axis)
        assert (normal[:, 2] > 0.85).all() and (left_normal[:, 2] > 0.85).all(), "Handover palms must face up"
        expected = palm.new_tensor([0.609, -0.923, 1.125])
        assert ((palm - expected).norm(dim=-1) < 0.02).all(), "Handover shoulder/palm placement changed"
        expected_left = palm.new_tensor([0.609, -0.226, 1.124])
        assert ((left[:, :3] - base.scene.env_origins - expected_left).norm(dim=-1) < 0.02).all()
        assert (robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2] >= 1.149).all(), "Support base below ground"
    elif "Lift" in task:
        expected = palm.new_tensor([0.575, 0.0, 1.296])
        assert ((palm - expected).norm(dim=-1) < 0.025).all(), "Lift arm must reach toward the table"
        assert (normal[:, 2] < -0.95).all(), "Lift palm must face down"
    else:
        expected_z = 0.9063 if "HoraRotate" in task else 1.0
        assert ((normal[:, 2] - expected_z).abs() < 0.01).all(), "In-hand task palm orientation changed"
    print("SCENE_POSE_PASS " + task, flush=True)


try:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    print("SMOKE_CREATE " + args.task, flush=True)
    env = gym.make(args.task, cfg=cfg)
    print("SMOKE_RESET", flush=True)
    base = env.unwrapped
    obs, _ = env.reset(seed=42)
    observation_shapes(obs, base.single_observation_space)
    robot = base.hand if hasattr(base, "hand") else base.scene["robot"]
    assert torch.isfinite(robot.data.joint_pos).all()
    limits = robot.data.joint_pos_limits
    default = robot.data.default_joint_pos
    assert ((default >= limits[..., 0] - 1e-5) & (default <= limits[..., 1] + 1e-5)).all(), "Default pose exceeds limits"
    finite(frame_state_w(robot, "right_palm"))
    check_reset_scene(base, args.task)
    if hasattr(base, "hand_dof_lower_limits"):
        assert ((default >= base.hand_dof_lower_limits - 1e-5) &
                (default <= base.hand_dof_upper_limits + 1e-5)).all(), "Default pose exceeds control limits"
    actions = torch.zeros((args.num_envs, base.action_space.shape[-1]), device=base.device)
    for _ in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(actions)
        finite(obs)
        finite(reward)
        finite(robot.data.joint_pos)
    observation_shapes(obs, base.single_observation_space)
    if "Handover" in args.task:
        from pxr import Usd, UsdPhysics
        from isaaclab.sim import find_matching_prims
        assert not set(LOCKED_REVOTRON_JOINTS).intersection(robot.joint_names)
        assert robot.num_joints == 56
        for root in find_matching_prims(robot.cfg.prim_path):
            locked = [prim for prim in Usd.PrimRange(root) if prim.GetName() in LOCKED_REVOTRON_JOINTS]
            assert len(locked) == 3 and all(prim.IsA(UsdPhysics.FixedJoint) for prim in locked)
        finite(frame_state_w(robot, "left_palm"))
    if "Lift" in args.task:
        # Zero relative actions must not make the wrist's own housings kick the arm away.
        from pxr import UsdPhysics
        from isaaclab.sim import find_matching_prims
        for root in find_matching_prims(robot.cfg.prim_path):
            excluded = UsdPhysics.FilteredPairsAPI(root.GetChild("Link5_R")).GetFilteredPairsRel().GetTargets()
            assert root.GetChild("Link7_R").GetPath() in excluded
        palm = frame_state_w(robot, "right_palm")[:, :3] - base.scene.env_origins
        assert ((palm - palm.new_tensor([0.575, 0.0, 1.296])).norm(dim=-1) < 0.05).all(), "Lift arm drifts under zero actions"
    from BrainCo_DexHand.artifacts import asset_contract, write_asset_contract, validate_asset_contract
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as run_dir:
        checkpoint = Path(run_dir) / "nn" / "test.pth"
        checkpoint.parent.mkdir()
        checkpoint.touch()  # metadata check only; no policy is loaded
        write_asset_contract(base, args.task, run_dir)
        validate_asset_contract(base, args.task, str(checkpoint))
        metadata_path = Path(run_dir) / "params" / "asset_contract.json"
        metadata = json.loads(metadata_path.read_text())
        if "robot_initial_state" in metadata:
            metadata["robot_initial_state"]["position"][2] -= 1.0
        else:
            metadata["asset_revision"] = "wrong-revision"
        metadata_path.write_text(json.dumps(metadata))
        try:
            validate_asset_contract(base, args.task, str(checkpoint))
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched checkpoint was accepted")
    contract = asset_contract(base, args.task)
    if args.check_rsl_runner:
        import importlib.metadata
        from isaaclab_tasks.utils import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        agent_cfg.device = base.device
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=base.device)
        with torch.inference_mode():
            prediction = runner.get_inference_policy(device=base.device)(wrapped.get_observations())
        finite(prediction)
        assert prediction.shape == actions.shape and prediction.device == actions.device
        print("RSL_RUNNER_PASS " + args.task, flush=True)
    print("ASSET_CONTRACT " + json.dumps(contract), flush=True)
    print("SMOKE_PASS " + json.dumps({"task": args.task, "steps": args.steps, "envs": args.num_envs,
                                    "joints": robot.num_joints, "actions": actions.shape[-1]}), flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # Kit shutdown can swallow an exception; preserve the smoke-check failure exit status.
    import os
    os._exit(1)
finally:
    app.close(wait_for_replicator=False)
