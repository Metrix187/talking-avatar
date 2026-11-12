"""Measure lip-sync in a recorded clip: correlate how open the mouth is in each
frame with how loud the audio is at that moment, across a range of offsets.
A peak at 0 frames means lips and voice line up."""
import subprocess, sys
import numpy as np, cv2
path = sys.argv[1]
cap = cv2.VideoCapture(path); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], capture_output=True).stdout
audio = np.frombuffer(raw, np.float32)
n = min(len(frames), len(audio) // 640)
env = np.array([np.sqrt(np.mean(audio[i * 640:(i + 1) * 640] ** 2)) for i in range(n)])
# mouth activity: frame-to-frame change in the lower-middle of the face
roi = np.stack([f[300:420, 176:336] for f in frames[:n]])
motion = np.r_[0, np.abs(np.diff(roi, axis=0)).mean(axis=(1, 2))]
talk = env > 0.02
print(f"{n} frames, {talk.sum()} with speech")
print(f"mouth motion while speaking {motion[talk].mean():.2f} vs silent {motion[~talk].mean():.2f}")
def corr(lag):
    a, b = (env[:n - lag], motion[lag:]) if lag >= 0 else (env[-lag:], motion[:n + lag])
    return np.corrcoef(a, b)[0, 1]
lags = range(-12, 13)
cs = [corr(l) for l in lags]
best = lags[int(np.argmax(cs))]
print("offset(frames): correlation  " + "  ".join(f"{l:+d}:{c:.2f}" for l, c in zip(lags, cs) if l % 2 == 0))
print(f"best offset: video {best:+d} frames relative to audio ({best * 40:+d} ms), r={max(cs):.2f}")
