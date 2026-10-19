#!/usr/bin/env python3
"""Sample new-asset grasp resets and retain states that survive a static hold.

This validates initialization for learning, not rotation or a trained policy.
Only object-filtered fingertip contacts count; resets invalidate a trial.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "BrainCo_DexHand"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("ball", "cylinder"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--batches", type=int, default=4)
parser.add_argument("--hold_steps", type=int, default=120)
parser.add_argument("--min_states", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.suffix != ".npz":
    parser.error("Output must have an .npz extension")
if args.output.exists():
    parser.error("Output already exists; choose a new cache path")
if min(args.num_envs, args.batches, args.hold_steps, args.min_states) <= 0:
    parser.error("Counts must be positive")
app = AppLauncher(args).app

import numpy as np
import torch
from BrainCo_DexHand.assets.description import DESCRIPTION_REVISION
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_hora_env import Revo3HandHoraEnv
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_hora_env_cfg import (
    Revo3HandHoraBallEnvCfg, Revo3HandHoraCylinderEnvCfg,
)


def hold(env):
    """Return exact sampled reset rows and a mask of uninterrupted stable holds."""
    env.reset()
    initial = torch.cat((env.hand.data.joint_pos, env.object_pos, env.object_rot), dim=-1).clone()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    contact_steps = torch.zeros(env.num_envs, device=env.device)
    actions = torch.zeros((env.num_envs, env.num_hand_dofs), device=env.device)
    contact_window = min(20, args.hold_steps)
    for step in range(args.hold_steps):
        obs, rewards, terminated, truncated, _ = env.step(actions)
        alive &= ~(terminated | truncated)
        alive &= torch.isfinite(rewards)
        for value in obs.values():
            alive &= torch.isfinite(value).flatten(1).all(1)
        if step >= args.hold_steps - contact_window:
            contacts = torch.stack([
                sensor.data.force_matrix_w.flatten(1, -2).norm(dim=-1).sum(-1) > 0.05
                for sensor in env._contact_sensor
            ], dim=-1)
            contact_steps += (contacts.sum(-1) >= 2)
    alive &= contact_steps >= 0.8 * contact_window
    alive &= (env.object_pos - initial[:, -7:-4]).norm(dim=-1) < 0.03
    alive &= env.object.data.root_lin_vel_w.norm(dim=-1) < 0.1
    return initial, alive


try:
    torch.manual_seed(args.seed)
    cfg = (Revo3HandHoraBallEnvCfg if args.task == "ball" else Revo3HandHoraCylinderEnvCfg)()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.sim.gravity = (0.0, 0.0, -9.81)
    cfg.gravity_curriculum = False
    cfg.force_scale = 0.0
    env = Revo3HandHoraEnv(cfg, render_mode=None)
    accepted = []
    counts = []
    with torch.inference_mode():
        for batch in range(args.batches):
            q = env.init_joint_pos + (torch.rand_like(env.hand.data.joint_pos) * 2 - 1) * 0.12
            q = q.clamp(env.hand_dof_lower_limits, env.hand_dof_upper_limits)
            pose = env.object.data.default_root_state[:, :7].clone()
            pose[:, :3] += (torch.rand_like(pose[:, :3]) * 2 - 1) * 0.004
            env.saved_grasping_states = torch.cat((q, pose), dim=-1)
            initial, valid = hold(env)
            accepted.append(initial[valid])
            counts.append(int(valid.sum()))
            print(f"CACHE_BATCH {batch + 1}: {counts[-1]}/{args.num_envs} held", flush=True)
        candidates = torch.cat(accepted)
        if len(candidates) < args.min_states:
            raise RuntimeError(f"Only {len(candidates)} held grasps; need {args.min_states}")
        # Fresh randomized PD gains and environments: independently replay sampled candidates.
        env.saved_grasping_states = candidates
        initial, valid = hold(env)
        states = np.unique(initial[valid].cpu().numpy(), axis=0)
        if len(states) < args.min_states:
            raise RuntimeError(f"Only {len(states)} unique replayed grasps; need {args.min_states}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, states=states, joint_names=np.array(env.hand.joint_names),
                        asset_revision=np.array(DESCRIPTION_REVISION))
    report = {
        "task": args.task, "asset_revision": DESCRIPTION_REVISION, "seed": args.seed,
        "num_envs": args.num_envs, "batch_survivors": counts, "states": len(states),
        "hold_steps": args.hold_steps, "hold_seconds": args.hold_steps * env.step_dt,
        "gravity": -9.81, "external_force_scale": 0.0,
        "minimum_object_contact_fingers": 2, "final_contact_fraction": 0.8,
        "max_position_drift_m": 0.03, "max_final_linear_speed_m_s": 0.1,
        "replay_survivors": int(valid.sum()), "replay_trials": args.num_envs,
        "scope": "Static initialization hold under randomized task properties; no rotation-policy validation",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("CACHE_PASS " + json.dumps(report), flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(1)
finally:
    app.close(wait_for_replicator=False)
