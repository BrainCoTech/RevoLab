#!/usr/bin/env bash
# Run ON the GPU server. Installs dependencies; does not start training.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="${REVOLAB_ROOT:-$HOME/revolab}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB="$ROOT/repos/IsaacLab"
ENV="$ROOT/envs/isaaclab"
LAB_COMMIT=b4c321024792976150ca55fddb26fa34480d974e
RL_GAMES_COMMIT=6b3534f29568158e9e29ec8bf83cc88fce5f0cae
RL_GAMES_SOURCE="${REVOLAB_RL_GAMES_SOURCE:-https://codeload.github.com/isaac-sim/rl_games/tar.gz/$RL_GAMES_COMMIT}"
CONSTRAINTS="$REPO/scripts/remote/constraints.txt"
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=4
export UV_LINK_MODE=copy
mkdir -p "$ROOT"/{repos,envs,logs,manifests,artifacts}
command -v uv >/dev/null
git lfs version
if [[ ! -d "$LAB/.git" ]]; then
    git init "$LAB"
    git -C "$LAB" remote add origin https://github.com/isaac-sim/IsaacLab.git
fi
if ! git -C "$LAB" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$LAB" fetch --depth 1 origin "$LAB_COMMIT"
    git -C "$LAB" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$LAB" rev-parse HEAD)" == "$LAB_COMMIT" ]]
if [[ ! -x "$ENV/bin/python" ]]; then
    uv venv --python 3.11.15 --seed "$ENV"
fi
PY="$ENV/bin/python"
export LD_LIBRARY_PATH="$ROOT/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ! "$PY" -c 'import ctypes; ctypes.CDLL("libGLU.so.1")' >/dev/null 2>&1; then
    # Minimal Ubuntu servers may lack this renderer library. Install privately.
    mkdir -p "$ROOT/bootstrap/system-libs" "$ROOT/sysroot"
    (
        cd "$ROOT/bootstrap/system-libs"
        apt-get download libglu1-mesa libopengl0
        for package in ./*.deb; do dpkg-deb -x "$package" "$ROOT/sysroot"; done
    )
    "$PY" -c 'import ctypes; ctypes.CDLL("libGLU.so.1")'
fi
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128'
uv pip install --python "$PY" --constraint "$CONSTRAINTS" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \
    'isaacsim[all,extscache]==5.1.0' 'setuptools<81' wheel toml
uv pip install --python "$PY" --constraint "$CONSTRAINTS" "rl-games @ $RL_GAMES_SOURCE"
uv pip install --python "$PY" --constraint "$CONSTRAINTS" --no-build-isolation \
    -e "$LAB/source/isaaclab" -e "$LAB/source/isaaclab_assets" \
    -e "$LAB/source/isaaclab_tasks" -e "$LAB/source/isaaclab_rl[rsl-rl]" \
    -e "$REPO/source/BrainCo_DexHand" onnxruntime
(cd "$REPO" && bash scripts/setup_assets.sh)
# These upstream releases disagree only on the unused web API dependency.
# Preserve the diagnostic and reject every other metadata inconsistency.
if ! "$PY" -m pip check > "$ROOT/manifests/pip-check.txt"; then
    expected='fastapi 0.115.7 has requirement starlette<0.46.0,>=0.40.0, but you have starlette 0.49.1.'
    cat "$ROOT/manifests/pip-check.txt"
    [[ "$(cat "$ROOT/manifests/pip-check.txt")" == "$expected" ]]
    echo 'Known upstream FastAPI/Starlette metadata conflict; this profile supports headless tasks only.'
fi
# Standalone USD must not shadow Isaac Sim's own pxr bindings.
if [[ ! -x "$ROOT/envs/assets/bin/python" ]]; then
    uv venv --python 3.11.15 "$ROOT/envs/assets"
fi
uv pip install --python "$ROOT/envs/assets/bin/python" usd-core==26.8 numpy==1.26.4 PyYAML
uv pip freeze --python "$PY" > "$ROOT/manifests/requirements-installed.txt"
{
    date -u --iso-8601=seconds
    "$PY" --version
    git -C "$REPO" rev-parse HEAD
    git -C "$LAB" rev-parse HEAD
    git -C "$REPO" submodule status
    echo "RL_GAMES_SOURCE=$RL_GAMES_SOURCE"
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv
} > "$ROOT/manifests/environment.txt"
echo 'BOOTSTRAP_PASS (installation only; run remote validation separately)'
