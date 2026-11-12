"""Compare denoise schedules on the same speech: sharpness numbers plus a
side-by-side strip of the same frame from each, to look at.

    python scripts/exp_steps.py
"""
import os
import sys
from pathlib import Path

os.environ["AVATAR_COMPILE"] = "0"
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import engine as E
from app.voice import Speaker

SCHEDULES = {
    "2: 1000,500": [1000, 500],
    "3: 1000,750,500": [1000, 750, 500],
    "3: 1000,500,250": [1000, 500, 250],
    "4: 1000,750,500,250": [1000, 750, 500, 250],
}

speech = Speaker().say("Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?", "af_heart")
speech = resample_poly(speech, 2, 3).astype(np.float32)

eng = E.AvatarEngine()
import flash_head.src.pipeline.flash_head_pipeline as fhp

n = eng.chunk_samples
chunks = [speech[i:i + n] for i in range(0, len(speech) - n + 1, n)][:6]
portrait = Image.open(ROOT / "vendor/SoulX-FlashHead/examples/girl.png")
strips, rows = [], []
for name, ts in SCHEDULES.items():
    eng.set_portrait(portrait, face_crop=False)
    p = eng.pipe
    t = [torch.tensor([float(x)], device=p.device) for x in ts + [0.0]]
    p.timesteps = [fhp.timestep_transform(x, shift=E.SAMPLE_SHIFT, num_timesteps=p.num_timesteps) for x in t]
    frames = np.concatenate([eng.step(c) for c in chunks])
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames[24:]]          # skip the first chunk
    whole = np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in gray])
    mouth = np.mean([cv2.Laplacian(g[300:400, 190:330], cv2.CV_64F).var() for g in gray])
    rows.append((name, whole, mouth, eng.last_step_s))
    picks = [frames[i] for i in (40, 60, 85, 110)]
    strips.append(np.concatenate(picks, axis=1))
    print(f"{name:22s} sharpness whole {whole:6.1f}  mouth {mouth:6.1f}", flush=True)

out = ROOT / "out" / "steps_compare.png"
Image.fromarray(np.concatenate(strips, axis=0)).save(out)
mouths = [np.concatenate([s[280:420, i * 512 + 170:i * 512 + 350] for i in range(4)], axis=1) for s in strips]
Image.fromarray(cv2.resize(np.concatenate(mouths, axis=0), None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)).save(ROOT / "out" / "steps_mouths.png")
print("wrote", out)
