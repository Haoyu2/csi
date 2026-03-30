# Use NVIDIA's TensorFlow 1.15 image which includes CUDA 11+ for Ampere GPUs (sm_86)
FROM nvcr.io/nvidia/tensorflow:23.01-tf1-py3

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# The code will be mounted at runtime
