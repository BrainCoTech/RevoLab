# Remote environment validation — 2026-10-16

Three independent Ubuntu 24.04.3 x86_64 hosts were provisioned in user-owned
workspaces. Each exposes eight RTX 5090 GPUs (32 GB) with driver 580.178.04.
SSH endpoints, account names and GPU UUIDs are kept in private server manifests,
outside this public repository. No training or simulator execution ran locally.

## Runtime and provenance

- Isaac Sim pip distribution `5.1.0.0`, Python `3.11.15`, PyTorch `2.7.0+cu128`.
- Isaac Lab `b4c321024792976150ca55fddb26fa34480d974e` (`isaaclab` package `0.54.4`).
- Robot submodule `6db7e2ed1d864ee0a3ab6ff62d43a303787137f6`; both LFS objects passed `git lfs fsck` on every host.
- RSL-RL `5.0.1`, RL-Games `1.6.1` from the pinned Isaac Sim fork, ONNX Runtime `1.27.0`.
- Runtime NumPy `1.26.0`; standalone asset checks use a separate environment with `usd-core 26.8` and NumPy `1.26.4`.
- Ray `2.45.0`, packaging `23.0`, Click `8.1.7`, W&B `0.26.1`, TensorDict `0.14.3`, Warp `1.18.0`.

RL-Games was built from the official source archive at commit
`6b3534f29568158e9e29ec8bf83cc88fce5f0cae` and transferred as a small pure-Python
wheel to avoid repeated slow GitHub downloads. Server manifests record the source
URL, archive SHA-256 and wheel SHA-256; the wheel digest was checked on each host.
All installed Python versions are recorded per host in `manifests/requirements-installed.txt`.

The known upstream FastAPI/Starlette metadata conflict and the headless-only scope
are described in [remote setup](remote-training.md). The dependency check accepts
only that exact conflict; other incompatibilities were resolved with version pins.

## Results

Task code was validated at RevoLab `d0c28c2` on nodes A/B and `9393d63` on node C.
The latter adds the private renderer-library setup; task implementation is unchanged.
Subsequent documentation and constraint pins record the already-tested versions.

| Task | Runtime joints / actions | Node A, GPU 2 | Node B, GPU 1 | Node C, GPU 3 |
| --- | ---: | --- | --- | --- |
| Repose Cube v1 | 21 / 21 | Passed | Passed | Passed |
| Reorient Cylinder v1 | 21 / 21 | Passed | Passed | Passed |
| Right Lift v1 | 28 / 28 | Passed | Passed | Passed |
| HORA Ball v1 | 21 / 21 | Passed | Passed | Passed |
| HORA Cylinder v1 | 21 / 21 | Passed | Passed | Passed |
| Dynamic Handover Cube v1 | 56 / 56 | Passed | Passed | Passed |

Each case used **2 environments × 32 zero-action steps**, including reset, finite
observations/rewards, declared observation shapes, default-pose physical/control
limits, named palm frames, asset-contract round-trip and revision mismatch
rejection. Handover checked all cloned waist/neck/head joints as native fixed joints.
The three RSL cases also constructed their actual configured runners and produced
finite correctly shaped outputs from an untrained policy on the selected GPU.
CUDA operations, backend imports, the offline USD checks and both deployment-input
unit tests passed on all three hosts.

Raw logs and success manifests remain on each host under `logs/validation-<UTC>/`
and `manifests/validation-<UTC>.txt`; private local copies are kept outside Git.
GPU indices describe the checks, not a resource reservation.

## Issues resolved before retraining

- Simulation and policy now follow the same explicit GPU selection in RSL-RL and RL-Games.
- RSL configuration uses Isaac Lab's version adapter and explicit `actor` observation groups, preserving Lift's proprioceptive and perception inputs.
- Repose/Reorient declare their actual 152-dimensional observations. HORA declares its `obs`, `priv_info` and `proprio_hist` tensors so checkpoint metadata describes the returned data.
- Workspace `TMPDIR` avoids another account's `/tmp/isaaclab/logs` permissions. A missing `libGLU.so.1` on one host was supplied from Ubuntu packages in a user-owned sysroot.

No optimizer updates, policy training, grasp-cache generation, task-success
assessment or hardware commands were performed. New HORA grasp caches and the
six retrained policies remain subsequent work.
