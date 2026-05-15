#!/bin/bash
set -e

# Get the directory where the script is located
SCRIPT_DIR="$(dirname $0)"

# Navigate back to the project root directory
cd "$SCRIPT_DIR/../"

# Set the target OSS directory for uploading
export CLOUD_FOLDER="universal/RevoLab/checkpoints"

echo "Uploading checkpoints to OSS directory: $CLOUD_FOLDER"

# Core pretrained checkpoint files to be uploaded
CHECKPOINT_FILES=(
    "checkpoints/BrainCo-Direct-Revo3-Repose-Cube-v0.pt"
    "checkpoints/BrainCo-Direct-Revo3-Reorient-Cylinder-v0.pt"
    "checkpoints/BrainCo-Dexsuite-Revo3-Right-Lift-v0.pt"
)

# Upload files using the base OSS upload script
./scripts/upload-oss.sh "${CHECKPOINT_FILES[@]}"

echo "All checkpoints uploaded successfully!"
