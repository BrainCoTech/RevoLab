# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dexsuite Revo3 environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteRevo3PPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DexsuiteRevo3DistillationRunnerCfg",
    },
)

gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftEnvCfg_PLAY",
        "play_env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteRevo3PPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DexsuiteRevo3DistillationRunnerCfg",
    },
)

gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Lift-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftTactileEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftTactileEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteRevo3PPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DexsuiteRevo3DistillationRunnerCfg",
    },
)

gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Lift-Tactile-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftTactileEnvCfg_PLAY",
        "play_env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftTactileEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteRevo3PPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DexsuiteRevo3DistillationRunnerCfg",
    },
)
