#!/bin/bash
set -e

# Get the directory where the script is located
SCRIPT_DIR="$(dirname $0)"
cd "$SCRIPT_DIR/../"

# Set the target directory for downloaded checkpoints
mkdir -p checkpoints/
echo "Downloading pretrained models from OSS to the checkpoints/ directory..."

# Note: This path must align with the CLOUD_FOLDER setting in upload-checkpoints.sh
BASE_URL="https://focus-resource.oss-cn-beijing.aliyuncs.com/universal/RevoLab/checkpoints"

CHECKPOINT_FILES=(
    "BrainCo-Direct-Revo3-Repose-Cube-v0.pt"
    "BrainCo-Direct-Revo3-Reorient-Cylinder-v0.pt"
    "BrainCo-Dexsuite-Revo3-Right-Lift-v0.pt"
)

for file in "${CHECKPOINT_FILES[@]}"; do
    echo "Downloading: $file"
    # Add timestamp parameter to bypass Aliyun OSS/CDN cache
    curl -L -C - -o "checkpoints/$file" "$BASE_URL/$file?t=$(date +%s)"
done

echo "All checkpoints downloaded successfully!"
