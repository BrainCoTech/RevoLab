"""Manager-based dynamic handover task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="BrainCo-Dynamic-Handover-Revo3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_handover_env_cfg:DynamicHandoverEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.dynamic_handover_env_cfg:DynamicHandoverEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="BrainCo-Dynamic-Handover-Revo3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_handover_env_cfg:DynamicHandoverEnvCfg_PLAY",
        "play_env_cfg_entry_point": f"{__name__}.dynamic_handover_env_cfg:DynamicHandoverEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="BrainCo-Dynamic-Handover-Revo3-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_handover_env_cfg:DynamicHandoverCubeEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.dynamic_handover_env_cfg:DynamicHandoverCubeEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="BrainCo-Dynamic-Handover-Revo3-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_handover_env_cfg:DynamicHandoverCubeEnvCfg_PLAY",
        "play_env_cfg_entry_point": f"{__name__}.dynamic_handover_env_cfg:DynamicHandoverCubeEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
