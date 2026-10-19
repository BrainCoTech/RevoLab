#!/usr/bin/env bash
# Run from an SSH/tmux session on a prepared GPU server.
set -euo pipefail
if [[ $# -lt 2 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash scripts/remote/run.sh GPU_INDEX SCRIPT.py [arguments...]" >&2
    exit 2
fi
GPU="$1"
SCRIPT="$2"
shift 2
ROOT="${REVOLAB_ROOT:-$HOME/revolab}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_ID="${REVOLAB_SERVER_ID:-$(cat "$ROOT/manifests/server-id")}"
[[ "$SERVER_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Unset CUDA_VISIBLE_DEVICES; pass the physical GPU index to this launcher.' >&2
    exit 2
fi
for arg in "$@"; do
    case "$arg" in
        --device|--device=*|--distributed|--kit_args|--kit_args=*)
            echo 'This launcher selects one GPU; do not override --device/--kit_args or use --distributed.' >&2
            exit 2 ;;
    esac
done
[[ -f "$REPO/$SCRIPT" ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PXR_WORK_THREAD_LIMIT="$OMP_NUM_THREADS"
export TMPDIR="$ROOT/tmp"
export LD_LIBRARY_PATH="$ROOT/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$TMPDIR"
RUN_ROOT="$ROOT/artifacts/$SERVER_ID"
mkdir -p "$RUN_ROOT"
cd "$RUN_ROOT"
nvidia-smi --id="$GPU" --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
exec "$ROOT/envs/isaaclab/bin/python" "$REPO/$SCRIPT" --device "cuda:$GPU" --headless \
    --kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=$OMP_NUM_THREADS --/plugins/omni.tbb.globalcontrol/maxThreadCount=$OMP_NUM_THREADS" "$@"
