#!/bin/bash
# Script to run the container with the data folder and GPU

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 /path/to/data/folder [gpu_id]"
    echo "Example: $0 ./Data 0"
    exit 1
fi

# Convert the provided data directory to an absolute path
export DATA_DIR=$(realpath "$1")
GPU_ID=${2:-0}

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

# Check if Docker supports NVIDIA runtime
if ! docker info | grep -i "nvidia" > /dev/null 2>&1; then
    echo "Error: NVIDIA Container Toolkit does not appear to be installed or configured in Docker."
    echo "Docker cannot access the GPU. Please check your Docker and NVIDIA Drivers installation."
    exit 1
fi

echo "Running container with data folder: ${DATA_DIR} on GPU: ${GPU_ID}"

# Use docker-compose to run the container
# This sets up the volumes automatically from docker-compose.yml
docker compose run --rm dnn_model python widar3_keras_mp.py ${GPU_ID}
