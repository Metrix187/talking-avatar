"""Does torch.compile (via triton-windows) make a chunk cheaper?

    python scripts/exp_compile.py --steps 4 --dit default --vae default

One configuration per process, so VRAM and compile caches start clean.
Reports compile time, steady-state ms per chunk (back to back, GPU clock up),
the split between denoising and the VAE, and peak VRAM.
"""
import argparse
import contextlib
import io
import os
import re
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--dit", default="none", help="none | default | reduce-overhead | max-autotune-no-cudagraphs | max-autotune")
ap.add_argument("--vae", default="none")
ap.add_argument("--rope", default="vendor", help="vendor | real (app/fast_rope.py)")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--out", default="")
args = ap.parse_args()
os.environ["AVATAR_STEPS"] = str(args.steps)
os.environ["AVATAR_COMPILE"] = "0"

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import engine as E

eng = E.AvatarEngine()
pipe = eng.pipe


def comp(fn, mode):
    return fn if mode == "none" else torch.compile(fn, mode=None if mode == "default" else mode)


if args.rope == "real":
    from app.fast_rope import install as install_rope

    install_rope(pipe.model, (5, 16, 16))

pipe.model = comp(pipe.model, args.dit)
pipe.vae.model.encode = comp(pipe.vae.model.encode, args.vae)
pipe.vae.model.decode = comp(pipe.vae.model.decode, args.vae)

t0 = time.perf_counter()
eng.set_portrait(Image.open(ROOT / "vendor/SoulX-FlashHead/examples/girl.png"), face_crop=False)
first = []
for _ in range(3):
    t = time.perf_counter()
    eng.step(None)
    first.append(time.perf_counter() - t)
print(f"setup + first 3 chunks: {time.perf_counter() - t0:.1f} s  ({', '.join(f'{x:.1f}' for x in first)})", flush=True)

# the engine swallows the vendor's per-stage prints; capture them for the split
E._quiet = lambda: contextlib.nullcontext()
rng = np.random.default_rng(0)
ts, frames, buf = [], [], io.StringIO()
with contextlib.redirect_stdout(buf):
    for i in range(args.n):
        audio = (rng.standard_normal(eng.chunk_samples) * 0.05).astype(np.float32) if i % 2 else None
        t = time.perf_counter()
        frames.append(eng.step(audio))
        ts.append(time.perf_counter() - t)
log = buf.getvalue()


def stage(name):
    v = [float(x) for x in re.findall(name + r": ([0-9.e-]+)s", log)]
    return 1000 * sum(v) / args.n if v else 0.0


k = args.n // 4
print(f"steps={args.steps} dit={args.dit} vae={args.vae} rope={args.rope}: "
      f"mean {np.mean(ts[k:]) * 1000:.0f} ms  worst {max(ts[k:]) * 1000:.0f} ms  "
      f"(denoise {stage('denoise per step'):.0f}, decode {stage('decode video frames'):.0f}, "
      f"encode {stage('encode motion frames'):.0f})  "
      f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB  "
      f"= {eng.chunk_seconds / np.mean(ts[k:]):.2f}x real time", flush=True)

if args.out:
    import cv2

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), E.FPS, (E.SIZE, E.SIZE))
    for f in np.concatenate(frames):
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print("wrote", args.out)
