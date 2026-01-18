FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ----------------------------
# OS packages
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates bash \
    build-essential cmake \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Install Miniforge (mamba)
# ----------------------------
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    -O /tmp/Miniforge3.sh && \
    bash /tmp/Miniforge3.sh -b -p /opt/conda && \
    rm -f /tmp/Miniforge3.sh

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-lc"]
ENV CONDA_AUTO_ACTIVATE_BASE=false

# Quick sanity check (helps catch PATH issues early)
RUN mamba --version && python --version

# ----------------------------
# Clone SAM-3D repo
# ----------------------------
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects
WORKDIR /workspace/sam-3d-objects

# ----------------------------
# Create conda env from repo
# ----------------------------
RUN mamba env create -f environments/default.yml

# ----------------------------
# Python tooling in env
# ----------------------------
RUN mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel

# Hydra patch script needs `import hydra` -> comes from hydra-core
RUN mamba run -n sam3d-objects pip install --no-cache-dir "hydra-core>=1.3,<1.4"

# Install torch/torchvision CUDA 12.1 wheels explicitly
RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install repo without its pinned deps (avoids torchaudio/flash-attn headaches)
RUN mamba run -n sam3d-objects pip install --no-cache-dir -e . --no-deps

# Apply repo patch (now hydra exists)
RUN mamba run -n sam3d-objects ./patching/hydra

# utils3d required by notebook/inference.py (install the correct project)
RUN mamba run -n sam3d-objects pip uninstall -y utils3d || true && \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# Runtime deps (RunPod + image handling)
RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    runpod fastapi uvicorn \
    numpy pillow opencv-python-headless imageio tqdm \
    "huggingface-hub[cli]<1.0"

# (Optional) only if you actually need seaborn
# RUN mamba run -n sam3d-objects pip install --no-cache-dir seaborn

# Sanity check imports
RUN mamba run -n sam3d-objects python -c "import hydra, utils3d, torch; print('ok', torch.__version__)"

# ----------------------------
# Copy your serverless code
# ----------------------------
COPY handler.py /workspace/sam-3d-objects/handler.py
COPY api.py /workspace/sam-3d-objects/api.py
COPY start.sh /workspace/sam-3d-objects/start.sh
RUN chmod +x /workspace/sam-3d-objects/start.sh

# ----------------------------
# RunPod serverless entry
# ----------------------------
CMD ["bash", "-lc", "cd /workspace/sam-3d-objects && exec mamba run -n sam3d-objects python -u handler.py"]
