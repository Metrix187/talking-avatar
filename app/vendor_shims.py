"""Stand-ins for two FlashHead imports this app does not need.

* `xfuser` is only used for multi-GPU sequence parallelism. On one GPU its
  functions are imported but never called, so a stub avoids a large dependency
  tree.
* `flash_head.utils.facecrop` pulls in a mediapipe version that conflicts with
  numpy 2. The replacement does the same crop with OpenCV's YuNet detector.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"


def _stub_xfuser():
    try:
        import xfuser  # noqa: F401
        return
    except ImportError:
        pass

    def _unavailable(*_a, **_k):
        raise RuntimeError("xfuser is stubbed out: multi-GPU inference is not supported here")

    pkg = types.ModuleType("xfuser")
    core = types.ModuleType("xfuser.core")
    dist = types.ModuleType("xfuser.core.distributed")
    dist.get_sequence_parallel_rank = lambda: 0
    dist.get_sequence_parallel_world_size = lambda: 1
    dist.get_sp_group = _unavailable
    dist.init_distributed_environment = _unavailable
    dist.initialize_model_parallel = _unavailable
    dist.get_world_group = _unavailable
    attn = types.ModuleType("xfuser.core.long_ctx_attention")
    attn.xFuserLongContextAttention = _unavailable
    pkg.core, core.distributed, core.long_ctx_attention = core, dist, attn
    sys.modules.update({
        "xfuser": pkg,
        "xfuser.core": core,
        "xfuser.core.distributed": dist,
        "xfuser.core.long_ctx_attention": attn,
    })


def detect_face(image: Image.Image):
    """Largest face as (x1, y1, x2, y2) in pixels, or None."""
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    # YuNet is unreliable on very large inputs; detect on a bounded copy.
    scale = min(1.0, 960 / max(h, w))
    small = cv2.resize(rgb, (round(w * scale), round(h * scale))) if scale < 1 else rgb
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (small.shape[1], small.shape[0]), 0.6, 0.3, 50)
    _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4] / scale
    return float(x), float(y), float(x + fw), float(y + fh)


def crop_portrait(image: Image.Image, face_ratio: float = 2.5, size: int = 512) -> Image.Image:
    """Square head-and-shoulders crop around the face, as FlashHead expects.
    Same geometry as the vendor crop (side = face width * ratio, face centre
    55% of the way down), with a wider ratio because YuNet boxes are tighter
    than mediapipe ones. Pads with edge colour instead of squashing when the
    crop runs off the picture."""
    image = image.convert("RGB")
    box = detect_face(image)
    if box is None:
        raise ValueError("No face detected")
    x1, y1, x2, y2 = box
    side = (x2 - x1) * face_ratio
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = round(cx - side * 0.5), round(cy - side * 0.55)
    side = round(side)
    right, bottom = left + side, top + side

    arr = np.asarray(image)
    h, w = arr.shape[:2]
    pad = max(0, -left, -top, right - w, bottom - h)
    if pad:
        arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        left, top, right, bottom = left + pad, top + pad, right + pad, bottom + pad
    return Image.fromarray(arr[top:bottom, left:right]).resize((size, size), Image.LANCZOS)


def _stub_facecrop():
    mod = types.ModuleType("flash_head.utils.facecrop")

    def process_image(input_path, face_ratio=2.0, target_size=(512, 512)):
        return crop_portrait(Image.open(input_path), face_ratio, target_size[0])

    mod.process_image = process_image
    sys.modules["flash_head.utils.facecrop"] = mod


def install():
    _stub_xfuser()
    _stub_facecrop()
