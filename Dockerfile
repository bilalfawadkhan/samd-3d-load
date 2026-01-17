# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

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
ENV CONDA_AUTO_ACTIVATE_BASE=false

# ----------------------------
# Clone repo
# ----------------------------
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects
WORKDIR /workspace/sam-3d-objects

# ----------------------------
# Create conda env
# ----------------------------
RUN mamba env create -f environments/default.yml

# Torch index (cu121)
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://pypi.ngc.nvidia.com"

# (Optional) remove binutils activation hook if your platform sets nounset aggressively
RUN rm -f /workspace/mamba/envs/sam3d-objects/etc/conda/activate.d/activate-binutils_linux-64.sh 2>/dev/null || true

# ----------------------------
# Python tooling
# ----------------------------
RUN mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel

# Needed for ./patching/hydra (provides `import hydra`)
RUN mamba run -n sam3d-objects pip install --no-cache-dir "hydra-core>=1.3,<1.4"

# Install torch + torchvision explicitly (CU121)
RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install the repo itself WITHOUT its pinned deps
RUN mamba run -n sam3d-objects pip install --no-cache-dir -e . --no-deps

# Apply repo patch (now works because hydra-core is installed)
RUN mamba run -n sam3d-objects ./patching/hydra

# ---- utils3d (REQUIRED by notebook/inference.py) ----
# IMPORTANT: Do NOT use "pip install utils3d" from PyPI (wrong package).
RUN mamba run -n sam3d-objects pip uninstall -y utils3d || true && \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# Runtime deps (serverless + image/mask handling)
RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    runpod numpy pillow opencv-python-headless imageio tqdm \
    "huggingface-hub[cli]<1.0"

# Sanity check
RUN mamba run -n sam3d-objects python -c "import utils3d; import torch; print('utils3d ok | torch', torch.__version__)"

# ----------------------------
# Copy handler into repo root
# ----------------------------
COPY handler.py /workspace/sam-3d-objects/handler.py

# ----------------------------
# RunPod Serverless entry
# ----------------------------
CMD ["bash", "-lc", "set +u; set +o nounset 2>/dev/null || true; cd /workspace/sam-3d-objects && mamba run -n sam3d-objects python -u handler.py"]
