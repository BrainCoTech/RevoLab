"""Adapt DirectRLEnv API to the 4-tuple API expected by Hora PPO/ProprioAdapt.

Gotcha — infos/extras reference: DirectRLEnv.step() returns self.extras as the info dict
  (same object reference). The wrapper must iterate extras BEFORE setting time_outs,
  otherwise time_outs gets overwritten by extras loop's .float().mean().

Gotcha — Bool tensors: extras may contain Bool tensors (e.g. height_reset_upper from
  _get_dones before .float()). The loop uses v.float().mean() to safely handle these.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from .revo3_hand_hora_env import Revo3HandHoraEnv


class HoraCompatWrapper:
    """Wrap Revo3HandHoraEnv for Hora PPO-compatible reset/step signatures."""

    def __init__(self, env: Revo3HandHoraEnv):
        self._env = env

    @property
    def observation_space(self) -> gym.spaces.Box:
        obs_dim = self._env.cfg.observation_space
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        n = self._env.cfg.action_space
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

    @property
    def prop_hist_len(self) -> int:
        """Return proprio history length used by ProprioAdapt."""
        return self._env.cfg.prop_hist_len

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset all envs and return obs_dict only."""
        obs_dict, _ = self._env.reset()
        return obs_dict

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """Step env and return old-style (obs_dict, rewards, dones, infos)."""
        obs_dict, rewards, terminated, truncated, infos = self._env.step(actions)
        dones = (terminated | truncated).to(torch.uint8)
        for k, v in getattr(self._env, "extras", {}).items():
            if isinstance(v, torch.Tensor):
                infos[k] = v.float().mean()
        infos["time_outs"] = truncated
        return obs_dict, rewards, dones, infos

    def __getattr__(self, name: str):
        """Forward unknown attributes to wrapped env."""
        return getattr(self._env, name)
