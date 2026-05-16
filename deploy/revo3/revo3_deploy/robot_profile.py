from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

JOINT_DIM = 21


@dataclass(frozen=True)
class Revo3Profile:
    path: Path
    hand: str
    policy_joint_order: tuple[str, ...]
    sdk_joint_order: tuple[str, ...]
    joint_lower_policy: np.ndarray
    joint_upper_policy: np.ndarray
    policy_to_sdk_perm: np.ndarray
    sdk_to_policy_perm: np.ndarray
    sdk_offset_rad: np.ndarray
    policy_offset_rad: np.ndarray
    action_scale: float
    default_rate_hz: float
    sdk: dict
    mit: dict

    @classmethod
    def load(cls, path: str | Path, policy_joint_order: list[str] | tuple[str, ...] | None = None) -> "Revo3Profile":
        profile_path = Path(path)
        with profile_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        profile_policy_order = tuple(cfg.get("policy_joint_order") or [])
        if policy_joint_order is None:
            policy_order = profile_policy_order
        else:
            policy_order = tuple(policy_joint_order)
            if profile_policy_order and policy_order != profile_policy_order:
                raise ValueError("policy.yaml joint order differs from profile policy_joint_order.")

        sdk_order = tuple(cfg.get("sdk_joint_order") or cfg.get("controller_joint_order") or [])
        _validate_order(policy_order, "policy_joint_order")
        _validate_order(sdk_order, "sdk_joint_order")
        if set(policy_order) != set(sdk_order):
            raise ValueError("policy_joint_order and sdk_joint_order must contain the same joints.")

        limits = cfg.get("joint_limits") or {}
        lower = []
        upper = []
        for joint in policy_order:
            joint_limits = limits.get(joint)
            if not isinstance(joint_limits, dict):
                raise ValueError(f"Missing joint_limits for {joint}.")
            lower.append(float(joint_limits["lower"]))
            upper.append(float(joint_limits["upper"]))

        policy_to_sdk = np.asarray([policy_order.index(name) for name in sdk_order], dtype=np.int64)
        sdk_to_policy = np.asarray([sdk_order.index(name) for name in policy_order], dtype=np.int64)
        sdk_offset = _load_offset(cfg)
        policy_offset = np.zeros(JOINT_DIM, dtype=np.float32)
        for sdk_index, policy_index in enumerate(policy_to_sdk):
            policy_offset[policy_index] = sdk_offset[sdk_index]

        return cls(
            path=profile_path,
            hand=str(cfg.get("hand", "right")),
            policy_joint_order=policy_order,
            sdk_joint_order=sdk_order,
            joint_lower_policy=np.asarray(lower, dtype=np.float32),
            joint_upper_policy=np.asarray(upper, dtype=np.float32),
            policy_to_sdk_perm=policy_to_sdk,
            sdk_to_policy_perm=sdk_to_policy,
            sdk_offset_rad=sdk_offset,
            policy_offset_rad=policy_offset,
            action_scale=float(cfg.get("action_scale", 1.0 / 24.0)),
            default_rate_hz=float(cfg.get("default_rate_hz", 20.0)),
            sdk=dict(cfg.get("sdk") or {}),
            mit=dict(cfg.get("mit") or {}),
        )

    def measured_sdk_to_policy(self, sdk_pos_rad: np.ndarray) -> np.ndarray:
        sdk_pos_rad = np.asarray(sdk_pos_rad, dtype=np.float32).reshape(JOINT_DIM)
        return sdk_pos_rad[self.sdk_to_policy_perm] - self.policy_offset_rad

    def target_policy_to_sdk(self, policy_target_rad: np.ndarray) -> np.ndarray:
        policy_target_rad = np.asarray(policy_target_rad, dtype=np.float32).reshape(JOINT_DIM)
        return policy_target_rad[self.policy_to_sdk_perm] + self.sdk_offset_rad


def _validate_order(order: tuple[str, ...], name: str) -> None:
    if len(order) != JOINT_DIM:
        raise ValueError(f"{name} must have {JOINT_DIM} entries, got {len(order)}.")
    if len(set(order)) != JOINT_DIM:
        raise ValueError(f"{name} contains duplicates.")


def _load_offset(cfg: dict) -> np.ndarray:
    offset_cfg = cfg.get("sim2real_joint_offset") or {}
    if not offset_cfg:
        return np.zeros(JOINT_DIM, dtype=np.float32)
    if offset_cfg.get("order") not in {"sdk_joint_order", "controller_joint_order"}:
        raise ValueError("sim2real_joint_offset.order must be sdk_joint_order.")
    values = np.asarray(offset_cfg.get("values") or [], dtype=np.float32).reshape(-1)
    if values.shape != (JOINT_DIM,):
        raise ValueError(f"sim2real_joint_offset.values must have {JOINT_DIM} entries.")
    return values
