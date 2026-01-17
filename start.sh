#!/usr/bin/env bash
set -eo pipefail

set +u
set +o nounset 2>/dev/null || true
export ADDR2LINE="${ADDR2LINE:-addr2line}"

cd /workspace/sam-3d-objects

# Optional: download checkpoints if missing (same env vars)
TAG="${SAM3D_TAG:-hf}"
DEST_DIR="checkpoints/${TAG}"
HF_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"

if [[ ! -d "${DEST_DIR}" ]]; then
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: checkpoints not found and no HF token provided."
    echo "Set HUGGINGFACE_HUB_TOKEN (or HF_TOKEN)."
    exit 1
  fi
  echo "Downloading checkpoints to ${DEST_DIR} ..."
  mamba run -n sam3d-objects python - <<'PY'
import os
from huggingface_hub import snapshot_download

tag = os.environ.get("SAM3D_TAG","hf")
repo = os.environ.get("SAM3D_REPO","facebook/sam-3d-objects")
token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
tmp_dir = f"checkpoints/{tag}-download"
snapshot_download(repo_id=repo, repo_type="model", local_dir=tmp_dir, token=token, max_workers=1)

src = os.path.join(tmp_dir, "checkpoints")
dst = os.path.join("checkpoints", tag)
if os.path.exists(dst):
    import shutil; shutil.rmtree(dst, ignore_errors=True)
os.rename(src, dst)
import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
print("Downloaded to", dst)
PY
fi

mamba run -n sam3d-objects python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
exec mamba run -n sam3d-objects uvicorn api:app --host 0.0.0.0 --port 8000
