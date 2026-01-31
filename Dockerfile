# Base Image: NVIDIA CUDA 12.1 compatible
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    software-properties-common \
    openssh-server \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniforge (Mamba)
# Installs Miniforge3 to /root/miniforge3 and adds to PATH temporarily for setup
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O miniforge.sh \
    && bash miniforge.sh -b -p /root/miniforge3 \
    && rm miniforge.sh

# Add Miniforge to PATH so we can use mamba
ENV PATH="/root/miniforge3/bin:${PATH}"

# 3. Project Setup (Clone from GitHub)
WORKDIR /app
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git sam-3d-objects
WORKDIR /app/sam-3d-objects

# 4. Create Conda/Mamba Environment
# This reads environments/default.yml and creates the 'sam3d-objects' environment
RUN mamba env create -f environments/default.yml

# 5. Set Environment as Default
# We PREPEND the environment bin to PATH so that 'python', 'pip', etc. point to the mamba env
ENV PATH="/root/miniforge3/envs/sam3d-objects/bin:${PATH}"
# Set CONDA_DEFAULT_ENV just in case
ENV CONDA_DEFAULT_ENV=sam3d-objects

# 6. Python Dependencies (pip steps from setup.md)

# Set environment variables for PyTorch / NVIDIA wheels
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Set CUDA_HOME and Arch List for building extensions (like gsplat, diff-gaussian-rasterization)
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
# Ensure TORCH_CUDA_ARCH_LIST is respected
ENV IBR_NET_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Install dev dependencies
RUN pip install -e '.[dev]'

# Install p3d dependencies
# The setup.md mentions this handles the broken pytorch3d dependency
RUN pip install -e '.[p3d]'

# Install inference dependencies
RUN pip install -e '.[inference]'

# 7. Patching
# Execute the patching script
RUN chmod +x ./patching/hydra && ./patching/hydra

# 8. Runtime / Handler Setup
COPY handler.py .
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Set Workspace Dir for start.sh
ENV WORKSPACE_DIR="/app/sam-3d-objects"

# Entrypoint
CMD [ "./start.sh" ]
