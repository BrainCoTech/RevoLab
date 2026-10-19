# Remote GPU environment

Run simulation, validation and training on the GPU servers. The local checkout is
for editing, Git and SSH transfers. Server disks are independent; copying code
does not copy environments, caches or experiment outputs.

## Pinned stack

| Component | Revision |
| --- | --- |
| Python | 3.11.15 |
| Isaac Sim | 5.1.0, pip `all,extscache` distribution |
| Isaac Lab | `b4c321024792976150ca55fddb26fa34480d974e` |
| PyTorch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| NumPy (simulation environment) | 1.26.0, required by the Isaac Sim pip package |
| RSL-RL | 5.0.1 |
| RL-Games | `6b3534f29568158e9e29ec8bf83cc88fce5f0cae` (1.6.1, Isaac Sim fork) |
| Ray / packaging / Click | 2.45.0 / 23.0 / 8.1.7 |
| Robot descriptions | RevoLab's pinned submodule gitlink |

Isaac Lab's `VERSION` file says 2.3.2 at this commit, but the checkout is newer than
the 2.3.2 release tag. Use the commit above to reproduce this environment.
Installation follows the official [Isaac Sim pip instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
and [Isaac Lab pip instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

The upstream packages have one incompatible metadata constraint: Isaac Sim pins
FastAPI 0.115.7 (Starlette <0.46), while this Isaac Lab checkout pins Starlette
0.49.1. This profile keeps the newer Starlette and supports headless tasks only;
it does not configure or validate a FastAPI service. The bootstrap records this
exact exception in `manifests/pip-check.txt` and rejects any other dependency
conflict. GPU validation is required separately; a completed install is not a
passing runtime check.

## Server layout and installation

Each server uses its own copy of this layout (override the root with `REVOLAB_ROOT`):

```text
~/revolab/
  repos/RevoLab/            # Working branch, including the asset submodule
  repos/IsaacLab/           # Detached at the pinned commit
  envs/isaaclab/            # Simulation and RL Python environment
  envs/assets/              # Standalone USD checks; separate pxr installation
  sysroot/                  # Optional user-scoped GLU/OpenGL libraries
  artifacts/<server-id>/    # Working directory for runs: logs/, outputs/, caches
  logs/                    # Installation and validation console logs
  tmp/                     # User-owned simulator temporary files and default logs
  manifests/               # Server ID, revisions, installed packages, validation
```

Install `uv` and Git LFS in the server user's environment, and put `uv`/`git-lfs`
on `PATH`. Check out the desired RevoLab revision under `~/revolab/repos/RevoLab`.
On the **server**, run:

```bash
cd ~/revolab/repos/RevoLab
mkdir -p ~/revolab/manifests
printf '%s\n' gpu-a > ~/revolab/manifests/server-id  # Unique to this server
bash scripts/remote/bootstrap.sh
```

The bootstrap targets Ubuntu 24.04 x86_64. It installs in the user's workspace, pins Isaac Lab and the RL-Games
fork, initializes the submodule/LFS payloads, and records installed versions.
If `libGLU.so.1` is missing, it downloads Ubuntu's `libglu1-mesa` and `libopengl0`
packages into `sysroot/` without changing system packages or requiring sudo.
It does not train. Do not install standalone `usd-core` into the simulation
environment; the asset-check environment keeps those USD bindings separate.
For restricted networks, `REVOLAB_RL_GAMES_SOURCE` may point to a `file://` wheel
built from the pinned fork commit. Record the original source URL and the source
archive/wheel SHA-256 hashes alongside the server manifest when using this route.
Keep SSH addresses and account names in local SSH configuration, outside Git.

## Validation and GPU selection

Check `nvidia-smi` immediately before starting a process; an idle GPU observed
earlier is not a reservation. From the server checkout:

```bash
# GPU 1 is an example; choose an available physical device.
bash scripts/remote/validate.sh 1
```

Validation checks offline assets, deployment input tests, backend imports, a small
CUDA operation, and all six tasks with two environments and 32 zero-action steps.
For the three RSL-RL tasks it also constructs the configured runner and checks one
forward pass using random initial weights, without calling the learning loop.
Logs are stored under `~/revolab/logs/validation-<UTC timestamp>/`. It performs no
optimizer steps and does not assess policy quality or grasp-cache quality.

`run.sh` selects one physical GPU for simulation and policy computation, disables
renderer multi-GPU use, defaults CPU thread pools to eight, and runs headlessly.
It sets `TMPDIR` inside the workspace: the default `/tmp/isaaclab/logs` can belong
to another account on a shared host and prevent simulation startup.
Leave `CUDA_VISIBLE_DEVICES` unset so physical device numbering remains explicit.
Isaac Sim initialization can still create CUDA contexts on other visible GPUs
(about 0.5 GiB per other card in the observed eight-GPU setup); selection does not
reserve the device or guarantee exclusive memory use.
The launcher sets `OMNI_KIT_ACCEPT_EULA=YES` for the installed Isaac Sim runtime.

## Subsequent training and artifacts

Start long runs inside a remote `tmux` session so SSH disconnects do not end them.
For example, **when retraining is authorized and the environment is validated**:

```bash
tmux new -s revo-repose
cd ~/revolab/repos/RevoLab
bash scripts/remote/run.sh 1 scripts/rsl_rl/train.py \
  --task BrainCo-Direct-Revo3-Repose-Cube-v1 --run_name seed42 --seed 42
```

RSL-RL and RL-Games write under `artifacts/<server-id>/logs/`; HORA writes under
`artifacts/<server-id>/outputs/`. Use distinct HORA `--output_name` values and
absolute paths for `--cache_file` and `--checkpoint`. The two HORA tasks still
require new revisioned grasp caches before the planned training campaign.

Keep a run's checkpoints, `params/asset_contract.json`, configuration files and
server manifest together. Transfer whole run directories with `rsync --partial`
via SSH; preserve the source server ID in the destination path. Inspect the copied
metadata before resume. Updating one server does not update any other server:
record each server's RevoLab commit, submodule commit and validation result after
every synchronization. Never infer server identity from hostname alone.
