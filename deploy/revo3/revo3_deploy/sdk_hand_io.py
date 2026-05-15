from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from revo3_deploy.robot_profile import JOINT_DIM

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
RPM_TO_RAD_S = 2.0 * math.pi / 60.0
RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


def _load_sdk():
    try:
        from bc_stark_sdk import main_mod as sdk
    except ImportError:
        try:
            from bc_stark_sdk import bc_stark_sdk as sdk
        except ImportError as exc:
            raise RuntimeError(
                "bc-stark-sdk is not installed. Install the hardware extra with "
                '`pip install -e ".[hardware]"` from deploy/revo3, or install the '
                "bc-stark-sdk wheel provided for your platform."
            ) from exc
    return sdk


@dataclass
class Revo3SdkConfig:
    port: str | None = None
    baudrate: int = 5000000
    slave_id: int = 126
    auto_detect: bool = True


class Revo3SdkHandIO:
    """Thin async adapter around bc-stark-sdk using radians internally."""

    def __init__(self, config: Revo3SdkConfig) -> None:
        self.config = config
        self.sdk = _load_sdk()
        self.ctx: Any | None = None
        self.slave_id = int(config.slave_id)

    async def open(self) -> None:
        self.sdk.init_logging()
        port = self.config.port
        baudrate = self.config.baudrate
        slave_id = self.config.slave_id

        if self.config.auto_detect and port is None:
            _, port, baudrate, slave_id = await self.sdk.auto_detect_modbus_revo3()

        self.slave_id = int(slave_id)
        self.ctx = await self.sdk.modbus_open(port, self._baudrate_enum(int(baudrate)))

    def close(self) -> None:
        if self.ctx is not None:
            self.sdk.modbus_close(self.ctx)
            self.ctx = None

    async def read_position_rad(self) -> np.ndarray:
        status = await self._ctx.v3_get_motor_status_data(self.slave_id)
        return np.asarray(status.positions[:JOINT_DIM], dtype=np.float32) * DEG_TO_RAD

    async def read_state_rad(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        status = await self._ctx.v3_get_motor_status_data(self.slave_id)
        pos = np.asarray(status.positions[:JOINT_DIM], dtype=np.float32) * DEG_TO_RAD
        vel = np.asarray(status.velocities[:JOINT_DIM], dtype=np.float32) * RPM_TO_RAD_S
        cur = np.asarray(status.currents[:JOINT_DIM], dtype=np.float32)
        return pos, vel, cur

    async def send_mit_command_rad(
        self,
        position_rad: np.ndarray,
        velocity_rad_s: np.ndarray | None = None,
        kp: float | list[float] | np.ndarray = 1.0,
        kd: float | list[float] | np.ndarray = 0.1,
        effort_ma: float | list[float] | np.ndarray = 0.0,
    ) -> None:
        pos_deg = self._vector(position_rad, "position_rad") * RAD_TO_DEG
        vel_rpm = self._vector(
            np.zeros(JOINT_DIM, dtype=np.float32) if velocity_rad_s is None else velocity_rad_s,
            "velocity_rad_s",
        ) * RAD_S_TO_RPM
        await self._ctx.revo3_multi_mit_set_all(
            self.slave_id,
            self._command_values(kp, "kp"),
            self._command_values(kd, "kd"),
            pos_deg.tolist(),
            vel_rpm.tolist(),
            self._command_values(effort_ma, "effort_ma"),
        )

    @property
    def _ctx(self):
        if self.ctx is None:
            raise RuntimeError("Revo3SdkHandIO is not open.")
        return self.ctx

    def _baudrate_enum(self, value: int):
        baudrate_type = self.sdk.Baudrate
        if hasattr(baudrate_type, "from_int"):
            return baudrate_type.from_int(value)
        mapping = {
            115200: baudrate_type.Baud115200,
            57600: baudrate_type.Baud57600,
            19200: baudrate_type.Baud19200,
            460800: baudrate_type.Baud460800,
            1000000: baudrate_type.Baud1Mbps,
            2000000: baudrate_type.Baud2Mbps,
            5000000: baudrate_type.Baud5Mbps,
        }
        return mapping[value]

    @staticmethod
    def _vector(value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (JOINT_DIM,):
            raise ValueError(f"{name} must have {JOINT_DIM} values.")
        return vector

    @staticmethod
    def _command_values(value, name: str) -> list[float]:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape == (1,):
            return [float(vector[0])] * JOINT_DIM
        if vector.shape != (JOINT_DIM,):
            raise ValueError(f"{name} must be scalar or {JOINT_DIM} values.")
        return [float(v) for v in vector]
