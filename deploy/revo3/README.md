# Revo3 Sim-to-Real Deploy

Lightweight, ROS-free deployment runtime for Revo3 ONNX policies exported from
RevoLab / Isaac Lab training.

This package keeps the training policy contract separate from the real robot
transport:

- `revo3_deploy.input_builder`: builds Stage-2 policy observations.
- `revo3_deploy.policy_runner`: loads ONNX and advances delta-action targets.
- `revo3_deploy.robot_profile`: validates joint limits, order mappings, and offsets.
- `revo3_deploy.sdk_hand_io`: talks to the Revo3 Python SDK (`bc-stark-sdk`).
- `scripts/export_policy.py`: exports a RSL-RL checkpoint to ONNX plus deploy config.
- `scripts/run_policy.py`: reads SDK joint state, runs ONNX, sends MIT commands.

## Install

```bash
cd deploy/revo3
pip install -e .
```

The hardware runtime uses the Revo3/Stark Python SDK (`bc-stark-sdk==1.4.5`).
It is optional so export-only workflows can be installed without the hardware
SDK:

```bash
pip install -e ".[hardware]"
```

If `bc-stark-sdk` is distributed to you as a wheel instead of through your
Python package index, install that wheel first, then install this package.

## Joint Orders

`config/revo3_right.yaml` stores both orders:

- `policy_joint_order`: Isaac Lab / ONNX action order.
- `sdk_joint_order`: Revo3 SDK motor order.

Runtime commands are generated in policy order and then permuted to SDK order by
joint name. SDK positions are degrees; deploy internals use radians.

## Export

The first version of `scripts/export_policy.py` delegates to Isaac Lab's RSL-RL
export helpers. Run it inside an Isaac Lab Python environment.

Install this package in that same Python environment first, or run the script
directly from this source tree as shown below.

```bash
python scripts/export_policy.py \
  --task BrainCo-Direct-Revo3-Repose-Cube-v0 \
  --checkpoint ../../checkpoints/BrainCo-Direct-Revo3-Repose-Cube-v0.pt \
  --output-dir artifacts/repose_cube
```

## Run

```bash
python scripts/run_policy.py \
  --onnx artifacts/repose_cube/policy.onnx \
  --policy artifacts/repose_cube/policy.yaml \
  --profile config/revo3_right.yaml \
  --port /dev/ttyUSB0 \
  --slave-id 126
```
