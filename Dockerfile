FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# ----------------------------
# System deps (split into build + runtime)
# ----------------------------
# Runtime libs first (keep)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Build deps (remove later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Miniforge (mamba)
# ----------------------------
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    -O /tmp/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /workspace/mamba && \
    rm -f /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/workspace/mamba/bin:$PATH
SHELL ["/bin/bash", "-lc"]
ENV CONDA_AUTO_ACTIVATE_BASE=false

# ----------------------------
# Clone repo
# ----------------------------
RUN git clone https://github.com/bilalfawadkhan/sam-3d-objects.git /workspace/sam-3d-objects
WORKDIR /workspace/sam-3d-objects

# ----------------------------
# Create conda env + install deps
# ----------------------------
RUN mamba env create -f environments/default.yml
RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    loguru seaborn
RUN mamba run -n sam3d-objects pip install --no-cache-dir seaborn

ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://pypi.ngc.nvidia.com"

RUN rm -f /workspace/mamba/envs/sam3d-objects/etc/conda/activate.d/activate-binutils_linux-64.sh 2>/dev/null || true

RUN mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel
RUN mamba run -n sam3d-objects pip install --no-cache-dir "hydra-core>=1.3,<1.4"

RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN mamba run -n sam3d-objects pip install --no-cache-dir -e . --no-deps
RUN mamba run -n sam3d-objects ./patching/hydra

RUN mamba run -n sam3d-objects pip uninstall -y utils3d || true && \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    runpod numpy pillow opencv-python-headless imageio tqdm \
    "huggingface-hub[cli]<1.0"

RUN mamba run -n sam3d-objects python -c "import utils3d; import torch; print('utils3d ok | torch', torch.__version__)"

# ----------------------------
# Cleanup to shrink image
# ----------------------------
# 1) Remove build tools
# 2) Clear apt cache
# 3) Clear conda/mamba package caches
RUN apt-get purge -y --auto-remove \
      git wget curl build-essential cmake \
    && rm -rf /var/lib/apt/lists/* \
    && mamba clean -a -y \
    && rm -rf /workspace/mamba/pkgs

# ----------------------------
# Copy handler
# ----------------------------
COPY handler.py /workspace/sam-3d-objects/handler.py

# ... after: RUN mamba env create -f environments/default.yml

ENV PATH=/workspace/mamba/envs/sam3d-objects/bin:$PATH

# keep your CMD (it’s fine)
CMD ["bash", "-lc", "cd /workspace/sam-3d-objects && mamba run -n sam3d-objects python -u handler.py"]




