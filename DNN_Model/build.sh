#!/bin/bash
# Script to build the Docker image when dependencies change
# Usage: ./build.sh

IMAGE_NAME="widar3-dnn-model"
IMAGE_TAG="latest"

echo "Building Docker image ${IMAGE_NAME}:${IMAGE_TAG}..."

# Build the docker image
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -eq 0 ]; then
    echo "Successfully built ${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo "Failed to build Docker image."
    exit 1
fi
