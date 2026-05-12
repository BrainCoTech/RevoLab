#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

HANDOVER_SOURCE_DISTANCE_THRESHOLD = 0.06
HANDOVER_CATCH_DISTANCE_THRESHOLD = 0.10


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
    return [*argv, "--task", "BrainCo-Dynamic-Handover-Revo3-Cube-Play-v0"]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "dynamic_handover" / "BrainCo_allegro.pth"
ISAACLAB_ROOT = _resolve_isaaclab_root()
_extend_pythonpath(ISAACLAB_ROOT, REPO_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Play dynamic handover cube with real-time handover command overrides.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BrainCo-Dynamic-Handover-Revo3-Cube-Play-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="RL-Games config entry.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the published pretrained checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use the last saved checkpoint when omitted.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--auto-cycle", action="store_true", default=True, help="Automatically cycle hold/throw phases.")
parser.add_argument("--hold-steps", type=int, default=18, help="Minimum hold steps before auto-throw.")
parser.add_argument("--throw-min-steps", type=int, default=10, help="Minimum throw steps before catch detection.")
parser.add_argument("--catch-stable-steps", type=int, default=6, help="Stable steps required to confirm a catch.")
parser.add_argument("--hold-dist-thresh", type=float, default=HANDOVER_SOURCE_DISTANCE_THRESHOLD, help="Distance threshold for source-hand hold.")
parser.add_argument("--catch-dist-thresh", type=float, default=HANDOVER_CATCH_DISTANCE_THRESHOLD, help="Distance threshold for receiver-hand catch.")
parser.add_argument("--hold-speed-thresh", type=float, default=0.35, help="Object speed threshold for hold stability.")
parser.add_argument("--catch-speed-thresh", type=float, default=0.45, help="Object speed threshold for catch stability.")
parser.add_argument(
    "--command-file",
    type=Path,
    default=None,
    help="Optional text file polled every step. Supported values: left_throw, right_throw, left_hold, right_hold, or two floats.",
)
parser.add_argument(
    "--initial-command",
    type=str,
    default="right_throw",
    help="Fallback handover command before the command file is written.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(_ensure_default_task(sys.argv[1:]))
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import BrainCo_DexHand  # noqa: F401, E402
import BrainCo_DexHand.tasks.manager_based.dynamic_handover  # noqa: F401, E402

from BrainCo_DexHand.assets.revotron_bimanual_revo3 import (  # noqa: E402
    LEFT_PALM_BODY_NAMES,
    RIGHT_PALM_BODY_NAMES,
)


def _parse_handover_command(command_text: str) -> torch.Tensor:
    value = command_text.strip().lower()
    presets = {
        "right_throw": torch.tensor([-1.0, 1.0], dtype=torch.float32),
        "left_throw": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "right_hold": torch.tensor([-1.0, -1.0], dtype=torch.float32),
        "left_hold": torch.tensor([1.0, -1.0], dtype=torch.float32),
    }
    if value in presets:
        return presets[value].clone()
    parts = value.replace(",", " ").split()
    if len(parts) == 2:
        return torch.tensor([float(parts[0]), float(parts[1])], dtype=torch.float32)
    raise ValueError(f"Unsupported handover command: {command_text!r}")


def _read_command_file(path: Path | None, fallback: torch.Tensor, last_mtime: float | None) -> tuple[torch.Tensor, float | None]:
    if path is None or not path.exists():
        return fallback, last_mtime
    stat = path.stat()
    if last_mtime is not None and math.isclose(stat.st_mtime, last_mtime):
        return fallback, last_mtime
    command = _parse_handover_command(path.read_text(encoding="utf-8"))
    return command, stat.st_mtime


def _command_to_phase(command: torch.Tensor) -> str:
    source_is_left = command[0].item() > 0.0
    throw_mode = command[1].item() > 0.0
    if source_is_left:
        return "left_throw" if throw_mode else "left_hold"
    return "right_throw" if throw_mode else "right_hold"


def _phase_to_command(phase: str, device: torch.device) -> torch.Tensor:
    return _parse_handover_command(phase).to(device=device)


def _first_body_index(robot, body_names: tuple[str, ...]) -> int | None:
    for body_name in body_names:
        try:
            body_ids, _ = robot.find_bodies(body_name)
        except ValueError:
            continue
        if body_ids:
            return body_ids[0]
    return None


def _body_pos(robot, body_idx: int | None) -> torch.Tensor:
    if body_idx is None:
        return robot.data.root_pos_w
    return robot.data.body_pos_w[:, body_idx]


def _handover_metrics(base_env) -> dict[str, float]:
    robot = base_env.scene["robot"]
    obj = base_env.scene["object"]

    right_idx = _first_body_index(robot, tuple(RIGHT_PALM_BODY_NAMES))
    left_idx = _first_body_index(robot, tuple(LEFT_PALM_BODY_NAMES))
    right_palm = _body_pos(robot, right_idx)
    left_palm = _body_pos(robot, left_idx)
    obj_pos = obj.data.root_pos_w
    obj_speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)

    env_id = 0
    return {
        "dist_to_right": torch.norm(obj_pos[env_id] - right_palm[env_id], p=2).item(),
        "dist_to_left": torch.norm(obj_pos[env_id] - left_palm[env_id], p=2).item(),
        "obj_speed": obj_speed[env_id].item(),
    }


def _next_phase(current_phase: str) -> str:
    if current_phase == "right_hold":
        return "right_throw"
    if current_phase == "right_throw":
        return "left_hold"
    if current_phase == "left_hold":
        return "left_throw"
    return "right_hold"


def _auto_cycle_phase(
    current_phase: str,
    phase_step_count: int,
    stable_counter: int,
    metrics: dict[str, float],
) -> tuple[str, int]:
    if current_phase == "right_hold":
        stable = metrics["dist_to_right"] < args_cli.hold_dist_thresh and metrics["obj_speed"] < args_cli.hold_speed_thresh
        stable_counter = stable_counter + 1 if stable else 0
        if phase_step_count >= args_cli.hold_steps and stable_counter >= args_cli.catch_stable_steps:
            return "right_throw", 0
        return current_phase, stable_counter
    if current_phase == "left_hold":
        stable = metrics["dist_to_left"] < args_cli.hold_dist_thresh and metrics["obj_speed"] < args_cli.hold_speed_thresh
        stable_counter = stable_counter + 1 if stable else 0
        if phase_step_count >= args_cli.hold_steps and stable_counter >= args_cli.catch_stable_steps:
            return "left_throw", 0
        return current_phase, stable_counter
    if current_phase == "right_throw":
        caught = metrics["dist_to_left"] < args_cli.catch_dist_thresh and metrics["obj_speed"] < args_cli.catch_speed_thresh
        stable_counter = stable_counter + 1 if caught else 0
        if phase_step_count >= args_cli.throw_min_steps and stable_counter >= args_cli.catch_stable_steps:
            return "left_hold", 0
        return current_phase, stable_counter
    caught = metrics["dist_to_right"] < args_cli.catch_dist_thresh and metrics["obj_speed"] < args_cli.catch_speed_thresh
    stable_counter = stable_counter + 1 if caught else 0
    if phase_step_count >= args_cli.throw_min_steps and stable_counter >= args_cli.catch_stable_steps:
        return "right_hold", 0
    return current_phase, stable_counter


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None and not args_cli.use_last_checkpoint and DEFAULT_CHECKPOINT.exists():
        resume_path = str(DEFAULT_CHECKPOINT)
    elif args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    base_env = env.unwrapped
    handover_term = base_env.command_manager.get_term("handover")
    current_command = _parse_handover_command(args_cli.initial_command)
    handover_term.set_manual_command(current_command)
    current_phase = _command_to_phase(current_command)
    phase_step_count = 0
    stable_counter = 0
    command_file_mtime = None

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for state in agent.states:
                    state[:, dones, :] = 0.0

        phase_step_count += 1
        metrics = _handover_metrics(base_env)
        next_phase, stable_counter = _auto_cycle_phase(current_phase, phase_step_count, stable_counter, metrics) if args_cli.auto_cycle else (current_phase, stable_counter)
        current_command, command_file_mtime = _read_command_file(args_cli.command_file, current_command, command_file_mtime)
        manual_phase = _command_to_phase(current_command)
        if args_cli.command_file is not None and command_file_mtime is not None:
            if manual_phase != current_phase:
                current_phase = manual_phase
                phase_step_count = 0
                stable_counter = 0
        elif next_phase != current_phase:
            current_phase = next_phase
            current_command = _phase_to_command(current_phase, base_env.device)
            phase_step_count = 0
            stable_counter = 0

        handover_term.set_manual_command(_phase_to_command(current_phase, base_env.device))
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
