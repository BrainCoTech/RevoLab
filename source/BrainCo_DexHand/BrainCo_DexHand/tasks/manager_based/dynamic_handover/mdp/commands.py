from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class BimanualHandoverCommand(CommandTerm):
    """Command term for bidirectional bimanual handover.

    The command is a 2D tensor per environment:
    - command[:, 0]: source-side selector. ``-1`` means the right hand starts with the object,
      ``+1`` means the left hand starts with the object.
    - command[:, 1]: mode selector. ``-1`` means hold in the source hand, ``+1`` means throw to
      the opposite hand.
    """

    cfg: "BimanualHandoverCommandCfg"

    def __init__(self, cfg: "BimanualHandoverCommandCfg", env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._pending_reset_command = torch.zeros_like(self._command)
        self._has_pending_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._manual_command: torch.Tensor | None = None
        self.metrics["source_side_abs"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.metrics["source_side_signed"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.metrics["source_is_left"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.metrics["source_is_right"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.metrics["throw_mode"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def __str__(self) -> str:
        msg = "BimanualHandoverCommand:\n"
        msg += "\tCommand dimension: 2\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def set_manual_command(self, command: torch.Tensor | Sequence[float] | None) -> None:
        if command is None:
            self._manual_command = None
            return
        command_tensor = torch.as_tensor(command, device=self.device, dtype=torch.float32).flatten()
        if command_tensor.numel() != 2:
            raise ValueError(f"Expected a 2D handover command, got shape {tuple(command_tensor.shape)}.")
        command_tensor[0] = torch.where(command_tensor[0] >= 0.0, torch.ones_like(command_tensor[0]), -torch.ones_like(command_tensor[0]))
        command_tensor[1] = torch.where(command_tensor[1] >= 0.0, torch.ones_like(command_tensor[1]), -torch.ones_like(command_tensor[1]))
        self._manual_command = command_tensor

    def prepare_reset(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return self._command[env_ids]
        command = self._sample_command(env_ids)
        self._pending_reset_command[env_ids] = command
        self._has_pending_reset[env_ids] = True
        self._command[env_ids] = command
        return command

    def get_command_for_envs(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        env_ids = self._resolve_env_ids(env_ids)
        return self._command[env_ids]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return super().reset(env_ids)

    def _update_metrics(self):
        self.metrics["source_side_abs"][:] = torch.abs(self._command[:, 0])
        self.metrics["source_side_signed"][:] = self._command[:, 0]
        self.metrics["source_is_left"][:] = (self._command[:, 0] > 0.0).float()
        self.metrics["source_is_right"][:] = (self._command[:, 0] < 0.0).float()
        self.metrics["throw_mode"][:] = 0.5 * (self._command[:, 1] + 1.0)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        pending_mask = self._has_pending_reset[env_ids]
        if torch.any(pending_mask):
            pending_env_ids = env_ids[pending_mask]
            self._command[pending_env_ids] = self._pending_reset_command[pending_env_ids]
            self._has_pending_reset[pending_env_ids] = False
        remaining_env_ids = env_ids[~pending_mask]
        if remaining_env_ids.numel() > 0:
            self._command[remaining_env_ids] = self._sample_command(remaining_env_ids)

    def _update_command(self):
        if self._manual_command is not None:
            self._command[:] = self._manual_command.unsqueeze(0)

    def _sample_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self._manual_command is not None:
            return self._manual_command.unsqueeze(0).expand(env_ids.numel(), -1).clone()

        source_side = torch.where(
            torch.rand(env_ids.numel(), device=self.device) < 0.5,
            -torch.ones(env_ids.numel(), device=self.device),
            torch.ones(env_ids.numel(), device=self.device),
        )
        throw_mode = torch.where(
            torch.rand(env_ids.numel(), device=self.device) < self.cfg.rel_throw_envs,
            torch.ones(env_ids.numel(), device=self.device),
            -torch.ones(env_ids.numel(), device=self.device),
        )
        return torch.stack((source_side, throw_mode), dim=-1)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)


@configclass
class BimanualHandoverCommandCfg(CommandTermCfg):
    class_type: type = BimanualHandoverCommand
    rel_throw_envs: float = 0.8
