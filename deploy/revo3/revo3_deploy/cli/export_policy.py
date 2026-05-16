from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an Isaac Lab/RSL-RL checkpoint to ONNX plus deploy policy.yaml."
    )
    parser.add_argument("--task", required=True, help="Isaac Lab task name.")
    parser.add_argument("--checkpoint", required=True, help="Path to RSL-RL .pt checkpoint.")
    parser.add_argument("--profile", default="config/revo3_right.yaml", help="Robot profile with policy joint order.")
    parser.add_argument("--output-dir", required=True, help="Directory for policy.onnx and policy.yaml.")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_with_isaaclab(args)
    return 0


def export_with_isaaclab(args: argparse.Namespace) -> None:
    """Run the Isaac Lab export path without entering the play loop."""

    from isaaclab.app import AppLauncher

    app_args = argparse.Namespace(
        headless=args.headless,
        enable_cameras=False,
        device=args.device,
        livestream=0,
        offscreen_render=False,
        render_mode=None,
        experience="",
        kit_args="",
    )
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
        from isaaclab_tasks.utils.hydra import hydra_task_config
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        import BrainCo_DexHand  # noqa: F401

        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args.task, args.agent)
        env_cfg.scene.num_envs = args.num_envs
        if args.device is not None:
            env_cfg.sim.device = args.device
            agent_cfg.device = args.device

        env_cfg.log_dir = str(Path(args.output_dir).resolve())
        env = gym.make(args.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        runner.load(str(Path(args.checkpoint).resolve()))
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        if normalizer is None:
            normalizer = getattr(policy_nn, "student_obs_normalizer", None)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(output_dir), filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(output_dir), filename="policy.onnx")
        write_policy_yaml(args, output_dir / "policy.yaml")
        env.close()
    finally:
        simulation_app.close()


def write_policy_yaml(args: argparse.Namespace, output_path: Path) -> None:
    profile_path = Path(args.profile)
    with profile_path.open("r", encoding="utf-8") as f:
        profile = yaml.safe_load(f) or {}

    policy_order = list(profile.get("policy_joint_order") or [])
    cfg = {
        "export": {
            "stage": "stage2",
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "task": args.task,
        },
        "artifacts": {"onnx": "policy.onnx"},
        "io_contract": {
            "inputs": [
                {"name": "obs", "shape": ["B", 126], "dtype": "float32"},
                {"name": "proprio_hist", "shape": ["B", 30, 42], "dtype": "float32"},
            ],
            "outputs": [{"name": "action", "shape": ["B", 21], "dtype": "float32"}],
            "action_semantics": "delta",
            "action_formula": "cur_targets = prev_targets + action_scale * action, then clamp to joint limits",
            "action_clip": [-1.0, 1.0],
            "policy_rate_hz": float(profile.get("default_rate_hz", 20.0)),
            "joint_order_right_hand": policy_order,
        },
        "normalization": {"baked_in_onnx": True},
    }
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    raise SystemExit(main())
