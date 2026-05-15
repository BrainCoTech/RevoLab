from __future__ import annotations

from collections import deque

import numpy as np

OBS_DIM = 126
HIST_LEN = 30
OBS_PER_STEP = 42
JOINT_DIM = 21
ACTION_SCALE = 1.0 / 24.0


class Stage2InputBuilder:
    """Build Stage-2 policy inputs from real robot joint positions."""

    def __init__(
        self,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        joint_names: tuple[str, ...] | list[str],
        action_scale: float = ACTION_SCALE,
    ) -> None:
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != JOINT_DIM:
            raise ValueError(f"joint_names must have {JOINT_DIM} entries.")
        if len(set(self.joint_names)) != JOINT_DIM:
            raise ValueError("joint_names contains duplicates.")

        self.joint_lower = self._as_vector(joint_lower, "joint_lower")
        self.joint_upper = self._as_vector(joint_upper, "joint_upper")
        if np.any(self.joint_upper <= self.joint_lower):
            raise ValueError("Every joint upper limit must be greater than lower.")

        self.action_scale = float(action_scale)
        self.current_target: np.ndarray | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=HIST_LEN)

    def reset(self, joint_pos: np.ndarray, target: np.ndarray | None = None) -> dict[str, np.ndarray]:
        joint_pos = self._as_vector(joint_pos, "joint_pos")
        if target is None:
            target = joint_pos
        target = self._clip_target(self._as_vector(target, "target"))
        frame = self._build_frame(joint_pos, target)

        self.current_target = target.copy()
        self._frames.clear()
        for _ in range(HIST_LEN):
            self._frames.append(frame.copy())
        return self._get_policy_inputs()

    def observe(self, joint_pos: np.ndarray) -> dict[str, np.ndarray]:
        joint_pos = self._as_vector(joint_pos, "joint_pos")
        if self.current_target is None:
            return self.reset(joint_pos)
        self._frames.append(self._build_frame(joint_pos, self.current_target))
        return self._get_policy_inputs()

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        if self.current_target is None:
            raise RuntimeError("Call reset() before action_to_target().")
        action = np.clip(self._as_vector(action, "action"), -1.0, 1.0)
        self.current_target = self._clip_target(self.current_target + self.action_scale * action)
        return self.current_target.copy()

    def _get_policy_inputs(self) -> dict[str, np.ndarray]:
        if len(self._frames) != HIST_LEN:
            raise RuntimeError(f"History not ready: expected {HIST_LEN} frames.")
        hist = np.stack(list(self._frames), axis=0).astype(np.float32)
        return {
            "obs": hist[-3:].reshape(1, OBS_DIM),
            "proprio_hist": hist.reshape(1, HIST_LEN, OBS_PER_STEP),
        }

    def _build_frame(self, joint_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        q_norm = (2.0 * joint_pos - self.joint_upper - self.joint_lower) / (
            self.joint_upper - self.joint_lower
        )
        return np.concatenate([q_norm, target], axis=0).astype(np.float32)

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        return np.clip(target, self.joint_lower, self.joint_upper).astype(np.float32)

    @staticmethod
    def _as_vector(value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (JOINT_DIM,):
            raise ValueError(f"{name} must have shape ({JOINT_DIM},), got {vector.shape}.")
        return vector
