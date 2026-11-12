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
    # YuNet misses faces that are large in its input: a face that fills the
    # frame, and most drawn or animal characters, only show up on a smaller
    # copy. So look at three sizes, largest first, and stop at the first hit.
    for long_side in (960, 480, 320):
        scale = min(1.0, long_side / max(h, w))
        small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        det = cv2.FaceDetectorYN.create(str(YUNET), "", (small.shape[1], small.shape[0]), 0.55, 0.3, 50)
        _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if faces is not None and len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4] / scale
            return float(x), float(y), float(x + fw), float(y + fh)
    return None


def center_square(image: Image.Image, size: int = 512) -> Image.Image:
    """The middle square of the picture: what is used when no face is found."""
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def crop_portrait(image: Image.Image, face_ratio: float = 2.5, size: int = 512) -> Image.Image:
    """Square head-and-shoulders crop around the face, as FlashHead expects.
    Same geometry as the vendor crop (side = face width * ratio, face centre
    55% of the way down), with a wider ratio because YuNet boxes are tighter
    than mediapipe ones. A crop that would run off the picture is slid back
    inside it, and shrunk to the picture's short side if it is larger than
    that, so nothing is squashed or padded."""
    image = image.convert("RGB")
    box = detect_face(image)
    if box is None:
        raise ValueError("No face detected")
    x1, y1, x2, y2 = box
    w, h = image.size
    side = min(round((x2 - x1) * face_ratio), w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(round(cx - side * 0.5), 0), w - side)
    top = min(max(round(cy - side * 0.55), 0), h - side)
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def _stub_facecrop():
    mod = types.ModuleType("flash_head.utils.facecrop")

    def process_image(input_path, face_ratio=2.0, target_size=(512, 512)):
        return crop_portrait(Image.open(input_path), face_ratio, target_size[0])

    mod.process_image = process_image
    sys.modules["flash_head.utils.facecrop"] = mod


def install():
    _stub_xfuser()
    _stub_facecrop()
