#!/bin/bash
# Script to run the container mapping heavily onto dual channel inputs locally natively

# Default to current directory if no argument is provided
DATA_DIR=${1:-"."}
GPU_ID=${2:-0}

export DATA_DIR=$(realpath "$DATA_DIR")

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

if ! docker info | grep -i "nvidia" > /dev/null 2>&1; then
    echo "Error: NVIDIA Container Toolkit cleanly missed inherently natively organically directly inside!"
    echo "Check Docker and drivers fundamentally inherently directly."
    exit 1
fi

echo "Running Advanced BVP+BAP container mapping structurally over: ${DATA_DIR} utilizing GPU explicitly inherently natively cleanly natively consistently purely organically precisely cleanly: ${GPU_ID}"

docker compose run --rm dnn_model python widar3_keras_bap_mp.py ${GPU_ID}
