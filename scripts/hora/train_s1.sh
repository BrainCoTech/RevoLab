#!/bin/bash

ISAACLAB=${ISAACLAB_PATH:-"$HOME/IsaacLab/isaaclab.sh"}
OUTPUT_NAME="debug"

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    OUTPUT_NAME=$1
    shift
fi

"${ISAACLAB}" -p scripts/hora/train.py \
    --algo PPO \
    --output_name "${OUTPUT_NAME}" \
    "$@"
