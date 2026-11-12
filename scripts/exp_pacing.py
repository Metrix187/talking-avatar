"""Does the GPU slow down when chunks are paced with pauses between them,
and does filling the pauses with light GPU work prevent it?"""
import sys, time, subprocess
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.engine import AvatarEngine
eng = AvatarEngine()
eng.set_portrait(Image.open(ROOT / "vendor/SoulX-FlashHead/examples/girl.png"), face_crop=False)
def smi():
    return subprocess.run(["nvidia-smi","--query-gpu=power.draw,clocks.gr,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
A = torch.randn(1536, 1536, device="cuda", dtype=torch.float16)
def wait_until(t, mode):
    if mode == "sleep":
        d = t - time.perf_counter()
        if d > 0: time.sleep(d)
    else:
        while time.perf_counter() < t:
            for _ in range(mode): (A @ A)
            torch.cuda.synchronize()
            if mode < 8: time.sleep(0.002)
def run(label, n=40, period=0.96, mode="sleep"):
    ts = []; nxt = time.perf_counter()
    for i in range(n):
        if period:
            nxt += period; wait_until(nxt, mode)
        t = time.perf_counter(); eng.step(None); ts.append(time.perf_counter() - t)
    k = n // 2
    print(f"{label:40s} last-half mean {np.mean(ts[k:])*1000:5.0f} ms  worst {max(ts[k:])*1000:5.0f} ms   [{smi()}]", flush=True)
run("back to back", n=12, period=None)
run("paced, sleeping in the gaps")
run("paced, light GPU filler (2 matmul/2ms)", mode=2)
run("paced, heavy GPU filler (spin)", mode=8)
