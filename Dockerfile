FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ----------------------------
# System deps
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates bash \
    build-essential cmake \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
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

# ----------------------------
# Clone repo
# ----------------------------
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects
WORKDIR /workspace/sam-3d-objects

# ----------------------------
# Create conda env
# ----------------------------
RUN mamba env create -f environments/default.yml

# Use CUDA/PyTorch extra indices (keep for torch/torchvision wheels if needed)
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://pypi.ngc.nvidia.com"
# Keep Kaolin link for later if you add kaolin
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# ----------------------------
# Install only what we need (avoid repo pinned dev/p3d deps)
# ----------------------------
RUN mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel

# Install the project itself, but DO NOT pull its pinned dependencies (avoids torchaudio==...+cu121, flash-attn, etc.)
RUN mamba run -n sam3d-objects pip install --no-cache-dir -e . --no-deps



# API + core runtime deps
RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    fastapi uvicorn pillow numpy pydantic \
    huggingface-hub[cli]<1.0

# Common inference deps often required by this repo (safe additions)
RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    opencv-python-headless imageio tqdm

# Optional: gsplat as wheel-only (won’t compile during build). If no wheel exists for your python, this will fail.
# Uncomment if your inference requires it:
# RUN mamba run -n sam3d-objects pip install --no-cache-dir --only-binary=:all: gsplat

# Build-time sanity check: ensure Inference import works
RUN mamba run -n sam3d-objects python -c "import sys; sys.path.append('notebook'); from inference import Inference; print('Inference import OK')"

# ----------------------------
# Copy API + entrypoint
# ----------------------------
COPY api.py /workspace/sam-3d-objects/api.py
COPY start.sh /workspace/sam-3d-objects/start.sh
RUN chmod +x /workspace/sam-3d-objects/start.sh

EXPOSE 8000
CMD ["/workspace/sam-3d-objects/start.sh"]

