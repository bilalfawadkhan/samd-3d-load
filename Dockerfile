# Base Image: NVIDIA CUDA 12.1 compatible
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Environment Setup & System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    software-properties-common \
    openssh-server \
    nginx \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m ensurepip --upgrade

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory for base app
WORKDIR /app

# 2. Python Dependencies (Pre-install)
# Install runpod (for the serverless handler)
RUN pip install runpod

# Set environment variables for PyTorch / NVIDIA wheels
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# 3. Project Install (Clone from GitHub)
# Clone the repository
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git sam-3d-objects

# Set working directory to the repo
WORKDIR /app/sam-3d-objects

# Remove nvidia-pyindex as it is redundant (we use ENV vars) and fails to build
RUN sed -i '/nvidia-pyindex/d' requirements.txt

# Run pip install -e '.[dev]' first
RUN pip install -e '.[dev]'

# Run pip install -e '.[p3d]' explicitly to handle the PyTorch3D dependency issue
RUN pip install -e '.[p3d]'

# Run pip install -e '.[inference]'
RUN pip install -e '.[inference]'

# 4. Patching
# Execute the patching script located at ./patching/hydra inside the container
RUN chmod +x ./patching/hydra && ./patching/hydra

# 5. Runtime / Handler Setup
# Copy handler and start script from build context to the repo directory
COPY handler.py .
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Set Workspace Dir for start.sh
ENV WORKSPACE_DIR="/app/sam-3d-objects"

# Entrypoint
CMD [ "./start.sh" ]
