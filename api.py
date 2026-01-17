import base64
import io
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append("notebook")
from inference import Inference  # type: ignore

TAG = os.environ.get("SAM3D_TAG", "hf")
CONFIG_PATH = os.environ.get("SAM3D_CONFIG", f"checkpoints/{TAG}/pipeline.yaml")

app = FastAPI(title="SAM3D Objects API")
_infer: Optional[Inference] = None


class Options(BaseModel):
    output: List[str] = Field(default_factory=lambda: ["ply", "glb"])
    seed: int = 42


class Sam3DRequest(BaseModel):
    imageBase64: str
    maskBase64: str
    options: Options = Field(default_factory=Options)


class Sam3DResponse(BaseModel):
    glbBase64: Optional[str] = None
    plyBase64: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


def _b64_to_pil_rgb(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _b64_to_mask_u8(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    m = Image.open(io.BytesIO(raw))
    if m.mode == "RGBA":
        alpha = np.array(m.split()[-1], dtype=np.uint8)
        return (alpha > 0).astype(np.uint8)
    arr = np.array(m.convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@app.get("/ping")
def ping():
    return {"status": "healthy"}


@app.on_event("startup")
def startup():
    global _infer
    _infer = Inference(CONFIG_PATH, compile=False)


@app.post("/run", response_model=Sam3DResponse)
def run(req: Sam3DRequest):
    if _infer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image = np.array(_b64_to_pil_rgb(req.imageBase64), dtype=np.uint8)
    mask = _b64_to_mask_u8(req.maskBase64)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize((image.shape[1], image.shape[0]), resample=Image.NEAREST)
    mask = (np.array(mask_img) > 0).astype(np.uint8)

    out = _infer(image, mask, seed=req.options.seed)

    glb_b64 = None
    ply_b64 = None

    with tempfile.TemporaryDirectory() as td:
        if "ply" in req.options.output:
            ply_path = os.path.join(td, "out.ply")
            out["gs"].save_ply(ply_path)
            ply_b64 = _file_to_b64(ply_path)

        if "glb" in req.options.output:
            glb_path = os.path.join(td, "out.glb")
            mesh = out.get("mesh")
            if hasattr(mesh, "to_glb"):
                mesh.to_glb(glb_path)
            elif hasattr(mesh, "save_glb"):
                mesh.save_glb(glb_path)
            elif hasattr(mesh, "export"):
                mesh.export(glb_path)
            else:
                raise HTTPException(status_code=500, detail="Mesh export method not found")
            glb_b64 = _file_to_b64(glb_path)

    return Sam3DResponse(glbBase64=glb_b64, plyBase64=ply_b64, meta={"seed": req.options.seed})
