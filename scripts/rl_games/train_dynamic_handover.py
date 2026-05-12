#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path


def _extend_pythonpath(isaaclab_root: Path, repo_root: Path) -> None:
    extra_paths = [
        repo_root / "source" / "BrainCo_DexHand",
        repo_root / "source",
        repo_root,
        isaaclab_root / "source" / "isaaclab",
        isaaclab_root / "source" / "isaaclab_tasks",
        isaaclab_root / "source" / "isaaclab_assets",
        isaaclab_root / "source" / "isaaclab_rl",
    ]
    for path in reversed(extra_paths):
        if path.exists():
            sys.path.insert(0, str(path))


def _resolve_isaaclab_root() -> Path:
    env_root = os.environ.get("ISAACLAB_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path.home() / "IsaacLab").resolve()


def _ensure_default_task(argv: list[str]) -> list[str]:
    if any(arg == "--task" or arg.startswith("--task=") for arg in argv):
        return argv
    return [*argv, "--task", "BrainCo-Dynamic-Handover-Revo3-Cube-v0"]


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = _resolve_isaaclab_root()
_extend_pythonpath(ISAACLAB_ROOT, REPO_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Train dynamic handover cube with RL-Games plus debug-friendly prints.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BrainCo-Dynamic-Handover-Revo3-Cube-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL-Games config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="The wandb project name.")
parser.add_argument("--wandb-entity", type=str, default=None, help="The wandb entity.")
parser.add_argument("--wandb-name", type=str, default=None, help="The wandb run name.")
parser.add_argument("--track", action="store_true", default=False, help="Enable Weights and Biases tracking.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(_ensure_default_task(sys.argv[1:]))
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import omni  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import BrainCo_DexHand  # noqa: F401, E402
import BrainCo_DexHand.tasks.manager_based.dynamic_handover  # noqa: F401, E402


class DebugIsaacAlgoObserver(IsaacAlgoObserver):
    """Print per-epoch episode statistics to stdout in addition to TensorBoard logging."""

    def _summarize_episode_infos(self) -> dict[str, float]:
        if not self.ep_infos:
            return {}

        summary: dict[str, float] = {}
        for key in self.ep_infos[0]:
            info_tensor = torch.tensor([], device=self.algo.device)
            for ep_info in self.ep_infos:
                value = ep_info[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor([value], dtype=torch.float32, device=self.algo.device)
                else:
                    value = value.to(self.algo.device)
                if value.ndim == 0:
                    value = value.unsqueeze(0)
                info_tensor = torch.cat((info_tensor, value))
            summary[key] = torch.mean(info_tensor).item()
        return summary

    def after_print_stats(self, frame, epoch_num, total_time):
        ep_summary = self._summarize_episode_infos()
        direct_summary: dict[str, float] = {}
        for key, value in self.direct_info.items():
            if isinstance(value, torch.Tensor):
                direct_summary[key] = value.item()
            else:
                direct_summary[key] = float(value)

        super().after_print_stats(frame, epoch_num, total_time)

        print("\n" + "*" * 100)
        print(f"[EPOCH {epoch_num}] frame={frame} total_time={total_time:.2f}s")
        if ep_summary:
            print("[EPISODE]")
            for key in sorted(ep_summary):
                print(f"  {key}: {ep_summary[key]:.6f}")
        else:
            print("[EPISODE] no episodic stats collected this epoch")
        if direct_summary:
            print("[DIRECT]")
            for key in sorted(direct_summary):
                print(f"  {key}: {direct_summary[key]:.6f}")
        print("*" * 100)


def _print_env_debug(env, env_cfg, agent_cfg, log_root_path: str, log_dir: str) -> None:
    obs_space = env.observation_space
    action_space = env.action_space
    rl_device = agent_cfg["params"]["config"]["device"]
    print(f"\n{'=' * 84}")
    print("[DEBUG] Dynamic Handover RL-Games Train")
    print(f"{'=' * 84}")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Num envs: {env.unwrapped.num_envs}")
    print(f"[INFO] Sim device: {env_cfg.sim.device}")
    print(f"[INFO] RL device: {rl_device}")
    print(f"[INFO] Seed: {agent_cfg['params']['seed']}")
    print(f"[INFO] Max epochs: {agent_cfg['params']['config']['max_epochs']}")
    print(f"[INFO] Log root: {log_root_path}")
    print(f"[INFO] Log run: {log_dir}")
    print(f"[INFO] Observation space: {obs_space}")
    print(f"[INFO] Action space: {action_space}")
    if hasattr(obs_space, 'shape'):
        print(f"[INFO] Observation shape: {obs_space.shape}")
    if hasattr(action_space, 'shape'):
        print(f"[INFO] Action shape: {action_space.shape}")
    print(f"[INFO] Episode length (s): {env_cfg.episode_length_s}")
    print(f"[INFO] Step dt: {env.unwrapped.step_dt}")
    print(f"{'=' * 84}\n")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO] Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    env_cfg.seed = agent_cfg["params"]["seed"]

    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", config_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = os.path.join(log_root_path, log_dir)
    else:
        omni.log.warn("IO descriptors are only supported for manager based RL environments.")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    _print_env_debug(env, env_cfg, agent_cfg, log_root_path, log_dir)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner(DebugIsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    print(f"[INFO] Runner ready. num_actors={agent_cfg['params']['config']['num_actors']}, rl_device={rl_device}")
    print("[INFO] Starting training loop...\n")

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    print("\n[INFO] Training loop completed.")
    print(f"[INFO] Outputs saved under: {os.path.join(log_root_path, log_dir)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
