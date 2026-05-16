#!/bin/bash

ISAACLAB=${ISAACLAB_PATH:-"$HOME/IsaacLab/isaaclab.sh"}
OUTPUT_NAME="debug"

if [ "$#" -gt 0 ] && [[ "$1" != --* ]] && [[ "$1" != *.pth ]] && [[ "$1" != *.ckpt ]]; then
    OUTPUT_NAME=$1
    shift
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: scripts/hora/train_s2.sh [OUTPUT_NAME] STAGE_CKPT [extra train.py args]" >&2
    exit 1
fi

CHECKPOINT=$1
shift

"${ISAACLAB}" -p scripts/hora/train.py \
    --algo ProprioAdapt \
    --output_name "${OUTPUT_NAME}" \
    --checkpoint "${CHECKPOINT}" \
    "$@"
