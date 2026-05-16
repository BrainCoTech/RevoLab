#!/usr/bin/env python3
"""Training entry point for Stage1 (PPO) and Stage2 (ProprioAdapt).

Task selection: --task ball|cylinder selects robot_cfg, object_cfg, and grasp cache.
  robot_cfg and object_cfg are chosen from assets.py (not env_cfg.py class defaults).

Cache path: {grasp_cache_path}.npy under assets/grasp_cache/hora/.
Override with --cache_file.

Gotcha — min envs: num_envs × horizon_length must be >= minibatch_size (32768) and
  exactly divisible. Minimum num_envs = 4096. Violation raises ValueError.

Gotcha — Stage2 enable_contact_in_obs=False: actor obs contacts zeroed, but
  proprio_hist retains real contact history for adapt_tconv distillation.
"""

import argparse
import copy
import datetime
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HORA_SKIP_SIM_CLOSE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_PATH = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXTENSION_PATH) not in sys.path:
    sys.path.insert(0, str(EXTENSION_PATH))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='cylinder', choices=['ball', 'cylinder'])
parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'ProprioAdapt'])
parser.add_argument('--train_cfg', type=str, default='Revo3HandHora')
parser.add_argument('--output_name', type=str, default='debug')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--cache_file', type=str, default='', help='Override grasp cache filename under assets/grasp_cache/hora/.')
parser.add_argument('--usd', type=str, default='', help='Override hand USD path.')
parser.add_argument('--num_envs', type=int, default=16384)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--test', action='store_true')
parser.add_argument('--force_overwrite', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def _is_stage2_checkpoint(path: str) -> bool:
    if not path:
        return False
    return path.endswith('.ckpt') or 'stage2_nn' in path


def _default_output_name() -> str:
    if args.algo == 'PPO':
        return 'run1_continue' if args.checkpoint else f'run_{args.task}'
    # Stage2: output to Stage1's run dir
    if not _is_stage2_checkpoint(args.checkpoint):
        return f'run_{args.task}'
    # Stage2 resume: output to same directory as checkpoint
    return 'run2_continue'


if not args.test and args.output_name == 'debug':
    args.output_name = _default_output_name()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from omegaconf import OmegaConf
from termcolor import cprint

from BrainCo_DexHand.algo.hora.padapt.padapt import ProprioAdapt
from BrainCo_DexHand.algo.hora.ppo.ppo import PPO
from BrainCo_DexHand.algo.hora.utils.misc import set_np_formatting, set_seed
from BrainCo_DexHand.tasks.direct.hora_rotation.assets import (
    BALL_OBJECT_CFG, CYLINDER_OBJECT_CFG,
    REVO3_HAND_BALL_CFG, REVO3_HAND_CYLINDER_CFG,
)
from BrainCo_DexHand.tasks.direct.hora_rotation.hora_compat_wrapper import HoraCompatWrapper
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_hora_env import Revo3HandHoraEnv
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_hora_env_cfg import Revo3HandHoraEnvCfg


_ALGO_MAP = {
    'PPO': PPO,
    'ProprioAdapt': ProprioAdapt,
}

_TASK_ROBOT_CFG = {'ball': REVO3_HAND_BALL_CFG, 'cylinder': REVO3_HAND_CYLINDER_CFG}
_TASK_OBJECT_CFG = {'ball': BALL_OBJECT_CFG, 'cylinder': CYLINDER_OBJECT_CFG}
_TASK_CACHE = {
    'ball': 'assets/grasp_cache/hora/revo3_right_grasp_ball',
    'cylinder': 'assets/grasp_cache/hora/revo3_right_grasp_cylinder',
}

def _build_full_config(seed: int):
    cfg_path = (
        REPO_ROOT
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "direct"
        / "hora_rotation"
        / "agents"
        / f"{args.train_cfg}.yaml"
    )
    train_cfg = OmegaConf.load(str(cfg_path))
    train_cfg.algo = args.algo
    train_cfg.load_path = os.path.abspath(args.checkpoint) if args.checkpoint else ''
    train_cfg.ppo.output_name = args.output_name
    minibatch = train_cfg.ppo.minibatch_size
    min_envs = minibatch // train_cfg.ppo.horizon_length
    if not args.test and (args.num_envs < min_envs or (args.num_envs * train_cfg.ppo.horizon_length) % minibatch != 0):
        raise ValueError(
            f"num_envs ({args.num_envs}) must be >= {min_envs} and num_envs*horizon must be divisible "
            f"by minibatch_size ({minibatch}). Valid num_envs: {', '.join(str(i) for i in range(min_envs, 20000, min_envs))}..."
        )
    train_cfg.ppo.num_actors = args.num_envs
    train_cfg.ppo.priv_info = True
    train_cfg.ppo.proprio_adapt = args.algo == 'ProprioAdapt'

    rl_device = getattr(args, 'device', None) or 'cuda:0'
    return OmegaConf.create({
        'rl_device': rl_device,
        'test': args.test,
        'seed': seed,
        'train': train_cfg,
    })


def _build_env_cfg(seed: int):
    env_cfg = Revo3HandHoraEnvCfg()
    env_cfg.robot_cfg = _TASK_ROBOT_CFG.get(args.task, REVO3_HAND_CYLINDER_CFG)
    env_cfg.object_cfg = _TASK_OBJECT_CFG.get(args.task, CYLINDER_OBJECT_CFG)
    env_cfg.grasp_cache_path = _TASK_CACHE.get(args.task, 'assets/grasp_cache/hora/revo3_right_grasp_cylinder')
    if args.cache_file:
        env_cfg.grasp_cache_path = f"assets/grasp_cache/hora/{args.cache_file.replace('.npy', '')}"
    if args.usd:
        usd_path = os.path.abspath(args.usd)
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"--usd path not found: {usd_path}")
        env_cfg.robot_cfg = copy.deepcopy(env_cfg.robot_cfg)
        if env_cfg.robot_cfg.spawn is None or not hasattr(env_cfg.robot_cfg.spawn, "usd_path"):
            raise RuntimeError("env_cfg.robot_cfg.spawn has no usd_path to override.")
        env_cfg.robot_cfg.spawn.usd_path = usd_path

    env_cfg.scene.num_envs = args.num_envs


    if hasattr(env_cfg, 'seed'):
        env_cfg.seed = seed
    if hasattr(env_cfg.sim, 'device') and getattr(args, 'device', None):
        env_cfg.sim.device = args.device
    return env_cfg


def _save_run_metadata(output_dif: str, full_config) -> None:
    date = str(datetime.datetime.now().strftime('%m%d%H'))
    with open(os.path.join(output_dif, 'gitdiff.patch'), 'w', encoding='utf-8') as f:
        f.write('')
    config_name = f'config_{date}.yaml'

    with open(os.path.join(output_dif, config_name), 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(full_config))


def _attach_env_runtime_to_config(full_config, env_cfg) -> None:
    full_config.env_runtime = OmegaConf.create(
        {
            'grasp_cache_path': str(env_cfg.grasp_cache_path),
        }
    )


def main():
    if args.test and not args.checkpoint:
        raise ValueError('--test requires --checkpoint')
    if args.algo == 'ProprioAdapt' and not args.checkpoint:
        raise ValueError('ProprioAdapt training requires --checkpoint')

    set_np_formatting()
    seed = set_seed(args.seed)
    full_config = _build_full_config(seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env_cfg = _build_env_cfg(seed)
    if args.algo == 'ProprioAdapt':
        env_cfg.enable_contact_in_obs = False  # Stage2: actor sees zero contact, adapt_tconv still sees contact history
    if args.test:
        env_cfg.gravity_curriculum = False
        env_cfg.sim.gravity = (0.0, 0.0, -9.81)  # full gravity for test/play
    env = Revo3HandHoraEnv(
        cfg=env_cfg,
        render_mode=None if getattr(args, 'headless', False) else 'human',
    )
    env = HoraCompatWrapper(env)

    # Output to Stage1's run directory
    if args.algo == 'ProprioAdapt' and not _is_stage2_checkpoint(args.checkpoint):
        output_dif = os.path.dirname(os.path.dirname(args.checkpoint))
    else:
        output_dif = os.path.join('outputs', 'hora', 'revo3_right', args.output_name)
    os.makedirs(output_dif, exist_ok=True)
    algo_name = str(full_config.train.algo)
    if algo_name not in _ALGO_MAP:
        raise ValueError(f"Unsupported algo: {algo_name}. Available: {list(_ALGO_MAP.keys())}")
    agent = _ALGO_MAP[algo_name](env, output_dif, full_config=full_config)

    if args.test:
        agent.restore_test(full_config.train.load_path)
        agent.test()
    else:
        best_ckpt_path = os.path.join(
            output_dif,
            'stage1_nn' if full_config.train.algo == 'PPO' else 'stage2_nn',
            'best.pth' if full_config.train.algo == 'PPO' else 'model_best.ckpt',
        )
        if os.path.exists(best_ckpt_path):
            if args.force_overwrite:
                print(f"[INFO] --force_overwrite enabled, continue and overwrite in {output_dif}", flush=True)
            else:
                user_input = input(
                    f'are you intentionally going to overwrite files in {output_dif}, type yes to continue \n'
                )
                if user_input != 'yes':
                    return

        _attach_env_runtime_to_config(full_config, env_cfg)
        _save_run_metadata(output_dif, full_config)
        agent.restore_train(full_config.train.load_path)
        agent.train()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\n[ERROR] Training terminated with an exception. Full traceback:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if os.getenv("HORA_SKIP_SIM_CLOSE", "0") == "1":
            print("[INFO] Skip simulation_app.close() due to HORA_SKIP_SIM_CLOSE=1", flush=True)
        else:
            simulation_app.close()
