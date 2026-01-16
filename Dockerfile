FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates bash \
    build-essential cmake \
    libglm-dev libgl1-mesa-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
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

# ----------------------------
# Pip indices / links (for NGC + cu121 + kaolin)
# ----------------------------
RUN mamba run -n sam3d-objects mamba install -y -c pytorch3d -c conda-forge pytorch3d
RUN mamba run -n sam3d-objects pip install --no-cache-dir kaolin


# ----------------------------
# Install Python deps INSIDE env
# ----------------------------
RUN mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel

# Fix nvidia-pyindex build issues (needs appdirs)
RUN mamba run -n sam3d-objects pip install --no-cache-dir appdirs && \
    mamba run -n sam3d-objects pip install --no-cache-dir --no-build-isolation "nvidia-pyindex==1.0.9"

# Repo extras
# Repo extras
RUN mamba run -n sam3d-objects pip install --no-cache-dir -e ".[dev]"

# p3d extras WITHOUT deps to avoid flash-attn build issues
RUN mamba run -n sam3d-objects pip install --no-cache-dir -e ".[p3d]" --no-deps

# Install core p3d deps explicitly (wheels)
RUN mamba run -n sam3d-objects pip install --no-cache-dir pytorch3d kaolin

# gsplat: force wheel-only to avoid CUDA compilation during docker build
RUN mamba run -n sam3d-objects pip install --no-cache-dir --only-binary=:all: "gsplat"

# Inference extras (this was missing in your last file)
RUN mamba run -n sam3d-objects pip install --no-cache-dir -e ".[inference]"

# Repo patch step
RUN mamba run -n sam3d-objects ./patching/hydra

# HF CLI + API runtime deps
RUN mamba run -n sam3d-objects pip install --no-cache-dir "huggingface-hub[cli]<1.0" && \
    mamba run -n sam3d-objects pip install --no-cache-dir fastapi uvicorn pillow numpy

# ----------------------------
# Copy API + entrypoint
# ----------------------------
COPY api.py /workspace/sam-3d-objects/api.py
COPY start.sh /workspace/sam-3d-objects/start.sh
RUN chmod +x /workspace/sam-3d-objects/start.sh

EXPOSE 8000
CMD ["/workspace/sam-3d-objects/start.sh"]


