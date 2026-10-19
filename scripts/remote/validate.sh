#!/usr/bin/env bash
# Six small GPU environment checks; no policies, optimizer steps or training.
set -euo pipefail
GPU="${1:?Pass the physical GPU index}"
ROOT="${REVOLAB_ROOT:-$HOME/revolab}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT/logs/validation-$STAMP"
mkdir -p "$LOG_DIR"
cd "$REPO"
"$ROOT/envs/assets/bin/python" scripts/check_assets.py > "$LOG_DIR/assets.log" 2>&1
"$ROOT/envs/assets/bin/python" -m unittest discover -s tests > "$LOG_DIR/offline-tests.log" 2>&1
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
"$ROOT/envs/isaaclab/bin/python" - "$GPU" > "$LOG_DIR/backends.log" 2>&1 <<'PY'
import sys
import importlib.metadata as metadata
import torch
from rsl_rl.runners import OnPolicyRunner
from rl_games.torch_runner import Runner
from BrainCo_DexHand.algo.hora.ppo.ppo import PPO
from BrainCo_DexHand.algo.hora.padapt.padapt import ProprioAdapt
import onnxruntime

device = f"cuda:{int(sys.argv[1])}"
x = torch.ones((32, 32), device=device)
assert (x @ x).eq(32).all().item()
torch.cuda.synchronize(device)
for package in ("isaacsim", "isaaclab", "torch", "rsl-rl-lib", "rl-games", "onnxruntime"):
    print(package, metadata.version(package))
print("BACKENDS_PASS", device, torch.cuda.get_device_name(device))
PY
tasks=(
    BrainCo-Direct-Revo3-Repose-Cube-v1
    BrainCo-Direct-Revo3-Reorient-Cylinder-v1
    BrainCo-Dexsuite-Revo3-Right-Lift-v1
    BrainCo-Direct-Revo3-HoraRotate-Ball-v1
    BrainCo-Direct-Revo3-HoraRotate-Cylinder-v1
    BrainCo-Dynamic-Handover-Revo3-Cube-v1
)
for task in "${tasks[@]}"; do
    echo "Checking $task on GPU $GPU ($LOG_DIR)"
    extra=()
    case "$task" in
        *Repose*|*Reorient*|*Lift*) extra=(--check_rsl_runner) ;;
    esac
    bash "$REPO/scripts/remote/run.sh" "$GPU" scripts/smoke_tasks.py \
        --task "$task" --num_envs 2 --steps 32 "${extra[@]}" > "$LOG_DIR/$task.log" 2>&1
    grep '^SMOKE_PASS ' "$LOG_DIR/$task.log"
done
{
    date -u --iso-8601=seconds
    git rev-parse HEAD
    git submodule status
    echo "GPU=$GPU"
    echo "LOG_DIR=$LOG_DIR"
    echo 'REMOTE_VALIDATION_PASS: backend imports, CUDA, offline assets, six 2-env/32-step checks; no training'
} | tee "$ROOT/manifests/validation-$STAMP.txt"
