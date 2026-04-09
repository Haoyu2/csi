#!/bin/bash
# Script to run the container mapping heavily onto dual channel inputs locally natively

# Default to the parent processed directory if no argument is provided
DATA_DIR=${1:-"/media/haoyu/New Volume/Widar3.0_Processed"}
GPU_ID=${2:-0}
USERS=${3:-"all"}

export DATA_DIR=$(realpath "$DATA_DIR")

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

if [ ! -d "$DATA_DIR/bvp_data" ] || [ ! -d "$DATA_DIR/bap_data" ]; then
    echo "Error: The provided data directory must be the parent folder that contains both 'bvp_data' and 'bap_data' subdirectories."
    echo "Currently checked: $DATA_DIR"
    exit 1
fi
if ! docker info | grep -i "nvidia" > /dev/null 2>&1; then
    echo "Error: NVIDIA Container Toolkit cleanly missed inherently natively organically directly inside!"
    echo "Check Docker and drivers fundamentally inherently directly."
    exit 1
fi

echo "Running Advanced BVP+BAP container mapping structurally over: ${DATA_DIR} utilizing GPU explicitly inherently natively cleanly natively consistently purely organically precisely cleanly: ${GPU_ID} for users: ${USERS}"

docker compose run --rm dnn_model python widar3_keras_bap_mp.py ${GPU_ID} "${USERS}"
