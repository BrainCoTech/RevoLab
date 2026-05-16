# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HORA-style Revo3 in-hand rotation tasks."""

import gymnasium as gym


gym.register(
    id="BrainCo-Direct-Revo3-HoraRotate-Ball-v0",
    entry_point=f"{__name__}.revo3_hand_hora_env:Revo3HandHoraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_hora_env_cfg:Revo3HandHoraBallEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraRotate-Cylinder-v0",
    entry_point=f"{__name__}.revo3_hand_hora_env:Revo3HandHoraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_hora_env_cfg:Revo3HandHoraCylinderEnvCfg",
    },
)
