import base64
import io
import os
import sys
import tempfile
from typing import Any, Dict, Optional, List

import numpy as np
from PIL import Image
import runpod

# Inference import (repo expects this)
sys.path.append("notebook")
from inference import Inference  # type: ignore

TAG = os.environ.get("SAM3D_TAG", "hf")
CONFIG_PATH = os.environ.get("SAM3D_CONFIG", f"checkpoints/{TAG}/pipeline.yaml")
REPO_ID = os.environ.get("SAM3D_REPO", "facebook/sam-3d-objects")

_infer: Optional[Inference] = None


def _b64_to_pil_rgb(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _b64_to_mask_u8(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    m = Image.open(io.BytesIO(raw))
    if m.mode == "RGBA":
        alpha = np.array(m.split()[-1], dtype=np.uint8)
        mask = (alpha > 0).astype(np.uint8)
    else:
        ml = m.convert("L")
        arr = np.array(ml, dtype=np.uint8)
        mask = (arr > 0).astype(np.uint8)
    return mask


def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _ensure_checkpoints(tag: str) -> str:
    """
    Ensures checkpoints/<tag>/pipeline.yaml exists. Downloads via HF token if missing.
    """
    ckpt_dir = os.path.join("checkpoints", tag)
    pipeline = os.path.join(ckpt_dir, "pipeline.yaml")
    if os.path.exists(pipeline):
        return pipeline

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Checkpoints missing and no HF token provided. Set HUGGINGFACE_HUB_TOKEN (or HF_TOKEN)."
        )

    # Use huggingface_hub snapshot_download (no interactive login)
    from huggingface_hub import snapshot_download

    tmp_dir = os.path.join("checkpoints", f"{tag}-download")
    os.makedirs("checkpoints", exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=tmp_dir,
        token=hf_token,
        max_workers=1,
    )

    # Model repo structure: <tmp_dir>/checkpoints -> move to checkpoints/<tag>
    src = os.path.join(tmp_dir, "checkpoints")
    if not os.path.isdir(src):
        raise RuntimeError(f"Downloaded repo but did not find '{src}' directory.")

    if os.path.exists(ckpt_dir):
        # If partially exists, remove to keep clean
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    os.rename(src, ckpt_dir)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not os.path.exists(pipeline):
        raise RuntimeError(f"pipeline.yaml not found after download at {pipeline}")

    return pipeline


def _get_infer() -> Inference:
    global _infer
    if _infer is None:
        _ensure_checkpoints(TAG)
        _infer = Inference(CONFIG_PATH, compile=False)
    return _infer


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod job payload expected:

    {
      "input": {
        "imageBase64": "...",
        "maskBase64": "...",
        "options": { "output": ["ply","glb"], "seed": 42 }
      }
    }
    """
    inp = job.get("input", {}) or {}

    image_b64 = inp.get("imageBase64")
    mask_b64 = inp.get("maskBase64")
    if not image_b64 or not mask_b64:
        return {"error": "Missing imageBase64 or maskBase64 in input."}

    options = inp.get("options", {}) or {}
    seed = int(options.get("seed", 42))
    outputs: List[str] = options.get("output", ["ply", "glb"])

    infer = _get_infer()

    image_pil = _b64_to_pil_rgb(image_b64)
    mask = _b64_to_mask_u8(mask_b64)

    image = np.array(image_pil, dtype=np.uint8)

    # Resize mask to match image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize((image.shape[1], image.shape[0]), resample=Image.NEAREST)
    mask = (np.array(mask_img) > 0).astype(np.uint8)

    out = infer(image, mask, seed=seed)

    glb_b64 = None
    ply_b64 = None

    with tempfile.TemporaryDirectory() as td:
        if "ply" in outputs:
            ply_path = os.path.join(td, "out.ply")
            gs = out.get("gs") if isinstance(out, dict) else None
            if gs is None or not hasattr(gs, "save_ply"):
                raise RuntimeError("PLY requested but out['gs'].save_ply not available")
            gs.save_ply(ply_path)
            ply_b64 = _file_to_b64(ply_path)

        if "glb" in outputs:
            glb_path = os.path.join(td, "out.glb")
            mesh = out.get("mesh") if isinstance(out, dict) else None
            if mesh is None:
                raise RuntimeError("GLB requested but out['mesh'] is missing")

            if hasattr(mesh, "to_glb"):
                mesh.to_glb(glb_path)
            elif hasattr(mesh, "save_glb"):
                mesh.save_glb(glb_path)
            elif hasattr(mesh, "export"):
                mesh.export(glb_path)
            else:
                raise RuntimeError("Mesh export method not found on out['mesh']")

            glb_b64 = _file_to_b64(glb_path)

    return {
        "glbBase64": glb_b64,
        "plyBase64": ply_b64,
        "meta": {"seed": seed, "tag": TAG},
    }


runpod.serverless.start({"handler": handler})
