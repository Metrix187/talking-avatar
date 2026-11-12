"""Render a portrait + audio file through the streaming engine, time every
chunk, report VRAM, and write an mp4 so the result can be looked at.

    python scripts/bench.py --image portrait.png --audio speech.wav --out out.mp4
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.engine import FPS, SAMPLE_RATE, AvatarEngine  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--image", default=str(ROOT / "vendor/SoulX-FlashHead/examples/girl.png"))
p.add_argument("--audio", default=str(ROOT / "vendor/SoulX-FlashHead/examples/podcast_sichuan_16k.wav"))
p.add_argument("--out", default=str(ROOT / "out" / "bench.mp4"))
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--compile", action="store_true")
p.add_argument("--no-crop", action="store_true")
args = p.parse_args()

import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

t0 = time.perf_counter()
eng = AvatarEngine(compile_model=args.compile)
print(f"load: {time.perf_counter() - t0:.1f}s  compiled={eng.compiled}")
t0 = time.perf_counter()
eng.set_portrait(Image.open(args.image), face_crop=not args.no_crop)
print(f"portrait: {time.perf_counter() - t0:.2f}s")

audio, sr = sf.read(args.audio, dtype="float32", always_2d=True)
audio = audio.mean(axis=1)
if sr != SAMPLE_RATE:
    audio = resample_poly(audio, SAMPLE_RATE, sr)
audio = audio[: int(args.seconds * SAMPLE_RATE)]
n = eng.chunk_samples
audio = np.pad(audio, (0, (-len(audio)) % n)).astype(np.float32)

torch.cuda.reset_peak_memory_stats()
frames, times = [], []
for i in range(0, len(audio), n):
    t = time.perf_counter()
    frames.append(eng.step(audio[i : i + n]))
    times.append(time.perf_counter() - t)
    print(f"chunk {i // n}: {times[-1] * 1000:.0f} ms")

warm = times[2:] or times
mean = float(np.mean(warm))
print(f"\nwarm chunk: mean {mean * 1000:.0f} ms, worst {max(warm) * 1000:.0f} ms "
      f"for {eng.chunk_seconds:.2f}s of video -> {eng.chunk_frames / mean:.1f} fps "
      f"({eng.chunk_seconds / mean:.2f}x real time)")
print(f"peak VRAM (torch): {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated, "
      f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB reserved")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
video = np.concatenate(frames)
wav = out.with_suffix(".wav")
sf.write(wav, audio, SAMPLE_RATE)
proc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x512",
     "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-shortest", str(out)],
    stdin=subprocess.PIPE,
)
proc.stdin.write(video.tobytes())
proc.stdin.close()
proc.wait()
wav.unlink()
print(f"wrote {out} ({len(video)} frames)")
