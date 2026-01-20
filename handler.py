# handler.py
import base64
import io
import os
import sys
import tempfile
from typing import Any, Dict

import numpy as np
from PIL import Image
import runpod

# Make SAM-3D notebook code importable
# Make SAM-3D notebook code importable
sys.path.append("notebook")

# Ensure CONDA_PREFIX exists even without conda activation
os.environ.setdefault("CONDA_PREFIX", "/workspace/mamba/envs/sam3d-objects")
os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])

from inference import Inference  # noqa: E402


# ----------------------------
# Load model ONCE at cold start
# ----------------------------
TAG = os.getenv("SAM3D_TAG", "hf")
CONFIG_PATH = os.getenv("SAM3D_CONFIG", f"checkpoints/{TAG}/pipeline.yaml")

print(f"[sam3d] Loading model config: {CONFIG_PATH}")
inference = Inference(CONFIG_PATH, compile=False)
print("[sam3d] Model loaded.")


def _decode_b64_to_pil(b64_str: str, mode: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(data))
    return img.convert(mode)


def _binary_mask_from_pil(mask_img: Image.Image) -> Image.Image:
    """
    Ensures mask is crisp binary (0 or 255) in 'L' mode.
    Accepts any incoming L/LA/RGB/RGBA; thresholds >0.
    If mask is multi-channel, uses last channel (often alpha).
    """
    arr = np.array(mask_img)
    if arr.ndim == 3:
        arr = arr[..., -1]
    arr = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def _encode_file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects RunPod payload:
    {
      "input": {
        "imageBase64": "...",
        "maskBase64": "...",
        "options": {"seed": 42, "output": ["ply"]}
      }
    }
    Returns:
    {
      "plyBase64": "...",
      "filename": "splat.ply"
    }
    """
    try:
        inp = job.get("input") or {}
        image_b64 = inp["imageBase64"]
        mask_b64 = inp["maskBase64"]
        options = inp.get("options", {})
        seed = int(options.get("seed", 42))

        # Decode inputs
        image = _decode_b64_to_pil(image_b64, "RGB")
        mask_raw = _decode_b64_to_pil(mask_b64, "L")  # we'll binarize anyway
        mask = _binary_mask_from_pil(mask_raw)

        # Run model
        out = inference(image, mask, seed=seed)

        # Export gaussian splat to PLY and return as base64
        with tempfile.TemporaryDirectory() as td:
            ply_path = os.path.join(td, "splat.ply")
            out["gs"].save_ply(ply_path)

            return {
                "plyBase64": _encode_file_b64(ply_path),
                "filename": "splat.ply",
            }

    except Exception as e:
        # RunPod marks job failed when "error" key is present
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
