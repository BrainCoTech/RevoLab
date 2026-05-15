from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from revo3_deploy.policy_runner import Revo3PolicyRunner
from revo3_deploy.robot_profile import Revo3Profile
from revo3_deploy.sdk_hand_io import Revo3SdkConfig, Revo3SdkHandIO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Revo3 ONNX policy through bc-stark-sdk.")
    parser.add_argument("--onnx", required=True, help="Path to policy.onnx.")
    parser.add_argument("--policy", required=True, help="Path to policy.yaml.")
    parser.add_argument("--profile", default="config/revo3_right.yaml", help="Path to robot profile YAML.")
    parser.add_argument("--port", default=None, help="Serial port. Omit for SDK auto-detect.")
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--slave-id", type=int, default=None)
    parser.add_argument("--rate", type=float, default=None, help="Override policy rate in Hz.")
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--effort-ma", type=float, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run inference without sending commands.")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    profile = Revo3Profile.load(args.profile)
    runner = Revo3PolicyRunner(args.onnx, args.policy, profile, use_gpu=args.use_gpu)

    sdk_cfg = profile.sdk
    io = Revo3SdkHandIO(
        Revo3SdkConfig(
            port=args.port,
            baudrate=int(args.baudrate or sdk_cfg.get("baudrate", 5000000)),
            slave_id=int(args.slave_id or sdk_cfg.get("slave_id", 126)),
            auto_detect=args.port is None and bool(sdk_cfg.get("auto_detect", True)),
        )
    )

    mit = profile.mit
    kp = float(args.kp if args.kp is not None else mit.get("kp", 1.0))
    kd = float(args.kd if args.kd is not None else mit.get("kd", 0.1))
    effort_ma = float(args.effort_ma if args.effort_ma is not None else mit.get("effort_ma", 0.0))
    rate_hz = float(args.rate or runner.rate_hz)
    period = 1.0 / rate_hz

    await io.open()
    print(f"Connected Revo3 slave_id={io.slave_id}; policy rate={rate_hz:.2f} Hz")
    try:
        next_tick = time.monotonic()
        while True:
            sdk_pos = await io.read_position_rad()
            policy_pos = profile.measured_sdk_to_policy(sdk_pos)
            policy_target = runner.step(policy_pos)
            sdk_target = profile.target_policy_to_sdk(policy_target)
            if args.dry_run:
                print(np.array2string(sdk_target, precision=3, suppress_small=True))
            else:
                await io.send_mit_command_rad(sdk_target, kp=kp, kd=kd, effort_ma=effort_ma)

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        return 130
    finally:
        io.close()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
