# CUDA base for GPU runtime (you need CUDA 12.1 because you reference cu121 wheels)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 1. System dependencies - Added cmake, libglm-dev, and libgl1 for gsplat/p3d
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential \
    cmake libglm-dev libgl1-mesa-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# ----------------------------
# System deps (wget, git, etc.)
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------
# (1) wget Miniforge installer
# ---------------------------------------------
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    -O /workspace/Miniforge3-Linux-x86_64.sh

# ---------------------------------------------------------
# (2) Install Miniforge non-interactively to /workspace/mamba
# ---------------------------------------------------------
RUN bash /workspace/Miniforge3-Linux-x86_64.sh -b -p /workspace/mamba

# Put mamba/conda on PATH
ENV PATH=/workspace/mamba/bin:$PATH

# ---------------------------------------------------------
# (3)(5) "source ~/.bashrc" is not needed in Docker layers.
# Instead we use bash -lc so PATH/conda works consistently.
# ---------------------------------------------------------
SHELL ["/bin/bash", "-lc"]

# ---------------------------------------------
# (6) git clone
# ---------------------------------------------
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects

WORKDIR /workspace/sam-3d-objects

# ---------------------------------------------
# (8) Create conda env from environments/default.yml
# (9) Activation is handled via "mamba run -n <env> ..."
# ---------------------------------------------
RUN mamba env create -f environments/default.yml

# ---------------------------------------------
# (10) pip extra indexes (set as ENV for later pip installs)
# ---------------------------------------------
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# ---------------------------------------------
# (11)(12)(14) Install extras into the env
# ---------------------------------------------
RUN mamba run -n sam3d-objects pip install -e ".[dev]" && \
    mamba run -n sam3d-objects pip install -e ".[p3d]"

# ---------------------------------------------
# (13) Kaolin find-links (set as ENV)
# ---------------------------------------------
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

RUN mamba run -n sam3d-objects pip install -e ".[inference]"

# ---------------------------------------------
# (15) patching/hydra
# ---------------------------------------------
RUN mamba run -n sam3d-objects ./patching/hydra

# ---------------------------------------------
# (16) huggingface cli (non-interactive auth later)
# ---------------------------------------------
RUN mamba run -n sam3d-objects pip install "huggingface-hub[cli]<1.0"

# FastAPI runtime deps (your API needs these)
RUN mamba run -n sam3d-objects pip install fastapi uvicorn pydantic pillow numpy

# ---------------------------------------------
# Copy API
# ---------------------------------------------
COPY api.py /workspace/sam-3d-objects/api.py
COPY start.sh /workspace/sam-3d-objects/start.sh
RUN chmod +x /workspace/sam-3d-objects/start.sh

# Expose port for RunPod LB serverless
EXPOSE 8000

# ---------------------------------------------
# Start FastAPI
# ---------------------------------------------
CMD ["/workspace/sam-3d-objects/start.sh"]

