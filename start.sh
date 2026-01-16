#!/usr/bin/env bash
set -euo pipefail

cd /workspace/sam-3d-objects

TAG="${SAM3D_TAG:-hf}"
DEST_DIR="checkpoints/${TAG}"

# Hugging Face gated model download needs a token.
# Set one of these env vars in RunPod:
#   HUGGINGFACE_HUB_TOKEN (recommended)
# or HF_TOKEN
HF_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"

if [[ ! -d "${DEST_DIR}" ]]; then
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: checkpoints not found and no HF token provided."
    echo "Set HUGGINGFACE_HUB_TOKEN (or HF_TOKEN) in RunPod environment variables."
    exit 1
  fi

  echo "Downloading checkpoints to ${DEST_DIR} ..."

  # (17)(18)(19)(21) hf download and move into expected structure
  # We avoid interactive `hf auth login` and instead pass token.
  mamba run -n sam3d-objects hf download \
    --repo-type model \
    --local-dir "checkpoints/${TAG}-download" \
    --max-workers 1 \
    --token "${HF_TOKEN}" \
    facebook/sam-3d-objects

  mv "checkpoints/${TAG}-download/checkpoints" "${DEST_DIR}"
  rm -rf "checkpoints/${TAG}-download"
else
  echo "Checkpoints already present at ${DEST_DIR}"
fi

# (22) CUDA sanity check
mamba run -n sam3d-objects python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

# Run FastAPI (bind 0.0.0.0 for RunPod networking) :contentReference[oaicite:1]{index=1}
exec mamba run -n sam3d-objects uvicorn api:app --host 0.0.0.0 --port 8000
