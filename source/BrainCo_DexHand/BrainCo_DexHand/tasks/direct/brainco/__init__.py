# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Brain Inhand Manipulation environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

inhand_task_entry = "BrainCo_DexHand.tasks.direct"

gym.register(
    id="BrainCo-Direct-Revo3-Repose-Cube-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.brainco_hand_env_cfg:BrainCoHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BrainCoHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-Reorient-Cylinder-v0",
    entry_point=f"{inhand_task_entry}.reorient:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.brainco_hand_reorient_env_cfg:BrainCoHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BrainCoHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
