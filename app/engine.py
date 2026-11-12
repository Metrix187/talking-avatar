"""Streaming avatar renderer built on SoulX-FlashHead Lite.

One call to `AvatarEngine.step(audio)` consumes 0.96 s of 16 kHz audio and
returns the 24 video frames (512x512, 25 fps) that go with it. The model keeps
its own motion context between calls, so calls must stay in playback order.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "SoulX-FlashHead"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

SAMPLE_RATE = 16000
FPS = 25
FRAME_NUM = 33                 # frames the model denoises per call
MOTION_LATENTS = 2             # latent frames carried over as motion context
SAMPLE_STEPS = int(os.environ.get("AVATAR_STEPS", "2"))   # 2 = real time on an 8 GB laptop GPU; 4 = sharper, needs a faster card
SAMPLE_SHIFT = 5
COLOR_CORRECTION = 1.0
AUDIO_CONTEXT_S = 8            # seconds of audio history fed to wav2vec
SIZE = 512
# Laptop GPU governors lower the clock when the load comes in bursts with idle
# gaps, which makes each chunk slower than real time (measured here: 700 ms per
# chunk back to back, 920-1120 ms when sleeping between chunks). Filling the
# gaps with throwaway work keeps the clock up. Costs power; set to 0 on a
# desktop card or if the driver is set to "Prefer maximum performance".
KEEP_GPU_HOT = os.environ.get("AVATAR_KEEP_GPU_HOT", "1") == "1"


class AvatarEngine:
    def __init__(
        self,
        ckpt_dir: str | os.PathLike = ROOT / "models" / "SoulX-FlashHead-1_3B",
        wav2vec_dir: str | os.PathLike = ROOT / "models" / "wav2vec2-base-960h",
        compile_model: bool | None = None,
        device: str = "cuda",
    ):
        from . import vendor_shims

        vendor_shims.install()
        import flash_head.src.pipeline.flash_head_pipeline as fhp

        if compile_model is None:
            compile_model = os.environ.get("AVATAR_COMPILE", "0") == "1"
        # the vendor's own compile switch is left off: it would compile the
        # model with its complex-number RoPE still in it (see _compile)
        fhp.COMPILE_MODEL = False
        fhp.COMPILE_VAE = False
        self.compiled = False

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.pipe = fhp.FlashHeadPipeline(
            checkpoint_dir=str(ckpt_dir),
            model_type="lite",
            wav2vec_dir=str(wav2vec_dir),
            device=device,
        )
        self.device = device
        stride_t = self.pipe.config.vae_stride[0]
        self.motion_frames = (MOTION_LATENTS - 1) * stride_t + 1      # 9
        self.chunk_frames = FRAME_NUM - self.motion_frames            # 24
        self.chunk_samples = self.chunk_frames * SAMPLE_RATE // FPS   # 15360
        self.chunk_seconds = self.chunk_frames / FPS                  # 0.96

        self._audio_end = AUDIO_CONTEXT_S * FPS
        self._audio_start = self._audio_end - FRAME_NUM
        self._audio = deque(maxlen=SAMPLE_RATE * AUDIO_CONTEXT_S)
        self._lock = threading.Lock()
        self._tmp = tempfile.TemporaryDirectory(prefix="avatar_")
        self._portrait_n = 0
        self.ready = False
        self.last_step_s = 0.0
        self._filler = torch.randn(1536, 1536, device=device, dtype=torch.float16)
        if compile_model:
            self._compile()

    def _compile(self):
        """torch.compile the transformer and the VAE (needs `triton-windows`
        on Windows). Measured here at 4 steps: 963 -> 808 ms per chunk. The
        price is about eight minutes of compiling on the first start (about
        20 s on later starts, from the cache)."""
        try:
            import triton  # noqa: F401
        except ImportError:
            print("AVATAR_COMPILE=1 ignored: triton is not installed "
                  "(pip install \"triton-windows>=3.3,<3.4\" for torch 2.7)", flush=True)
            return
        from .fast_rope import install as install_rope

        # keep compiled kernels out of %TEMP%, where they get cleaned up: a
        # cold compile is ~8 min, a cached one ~20 s. The path must be short:
        # triton nests its files deep enough to pass Windows' 260-character
        # limit from a folder inside this project.
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.home() / ".cache" / "talking-avatar"))
        pipe, stride = self.pipe, self.pipe.config.vae_stride
        grid = ((FRAME_NUM - 1) // stride[0] + 1, SIZE // stride[1], SIZE // stride[2])
        install_rope(pipe.model, grid)
        pipe.model = torch.compile(pipe.model)
        pipe.vae.model.encode = torch.compile(pipe.vae.model.encode)
        pipe.vae.model.decode = torch.compile(pipe.vae.model.decode)
        self.compiled = True

    # -- portrait ---------------------------------------------------------
    def set_portrait(self, image: Image.Image, face_crop: bool = True, seed: int = 9999):
        """Switch to a new reference image and reset motion and audio context.
        Raises ValueError if `face_crop` is on and no face can be found."""
        if face_crop:
            # crop here, not in the vendor pipeline: it swallows detection
            # failures and silently falls back to the uncropped picture
            from .vendor_shims import crop_portrait

            image = crop_portrait(image, size=SIZE)
        self._portrait_n += 1
        path = Path(self._tmp.name) / f"portrait{self._portrait_n}.png"
        image.convert("RGB").save(path)
        with self._lock, _quiet():
            self.pipe.prepare_params(
                # the vendor code splits on "/" to name the portrait
                cond_image_path_or_dir=path.as_posix(),
                target_size=(SIZE, SIZE),
                frame_num=FRAME_NUM,
                motion_frames_num=self.motion_frames,
                sampling_steps=SAMPLE_STEPS,
                seed=seed,
                shift=SAMPLE_SHIFT,
                color_correction_strength=COLOR_CORRECTION,
                use_face_crop=False,
            )
            self._audio.clear()
            self._audio.extend([0.0] * self._audio.maxlen)
            self.ready = True

    def portrait_preview(self) -> Image.Image:
        """The 512x512 crop the model is actually conditioned on."""
        t = self.pipe.cond_image_tensor_dict[self.pipe.person_name]   # 1 C 1 H W in [-1, 1]
        arr = ((t[0, :, 0].float().permute(1, 2, 0) + 1) * 127.5).clamp(0, 255)
        return Image.fromarray(arr.byte().cpu().numpy())

    # -- generation -------------------------------------------------------
    @torch.no_grad()
    def step(self, audio: np.ndarray | None = None) -> np.ndarray:
        """Advance by one chunk. `audio` is float32 mono 16 kHz, exactly
        `chunk_samples` long (None = silence). Returns uint8 [24, 512, 512, 3]."""
        if not self.ready:
            raise RuntimeError("set_portrait() first")
        if audio is None:
            audio = np.zeros(self.chunk_samples, dtype=np.float32)
        if len(audio) != self.chunk_samples:
            raise ValueError(f"expected {self.chunk_samples} samples, got {len(audio)}")

        with self._lock, _quiet():
            t0 = time.perf_counter()
            self._audio.extend(audio.tolist())
            ctx = np.fromiter(self._audio, dtype=np.float32, count=len(self._audio))
            emb = self.pipe.preprocess_audio(ctx, sr=SAMPLE_RATE, fps=FPS)
            # each video frame sees a 5-wide window of audio features
            idx = torch.arange(self._audio_start, self._audio_end).unsqueeze(1) + (torch.arange(5) - 2).unsqueeze(0)
            idx = idx.clamp(0, self._audio_end - 1)
            emb = emb[idx][None].contiguous().to(self.device)

            video = self.pipe.generate(emb)                       # C T H W in [-1, 1]
            video = video[:, self.motion_frames:]
            frames = ((video + 1) * 127.5).clamp(0, 255).permute(1, 2, 3, 0).byte()
            out = frames.cpu().numpy()
            torch.cuda.synchronize()
            self.last_step_s = time.perf_counter() - t0
        return out

    def wait(self, seconds: float):
        """Pause between chunks without letting the GPU clock drop."""
        if not KEEP_GPU_HOT or seconds <= 0:
            time.sleep(max(0.0, seconds))
            return
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            for _ in range(8):
                self._filler @ self._filler
            torch.cuda.synchronize()

    def warmup(self, n: int = 2):
        for _ in range(n):
            self.step(None)


@contextlib.contextmanager
def _quiet():
    """The vendor pipeline prints timing for every denoise step."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield
