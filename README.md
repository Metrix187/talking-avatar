# Talking Avatar

Turn one portrait into a face you can talk to. You speak (or type), an LLM
answers, a local voice reads the answer, and a diffusion model animates the
portrait in step with the voice, streamed to a browser tab as it is generated.

Everything except the LLM runs on your own machine. Built and measured on an
RTX 5070 Laptop GPU (8 GB), Windows 11.

![The sample portrait speaking a typed line](docs/demo.gif)

Rendered live on the laptop at 25 fps. [The same clip with sound](docs/demo.mp4)
(the voice is Kokoro; the portrait is the sample image from SoulX-FlashHead).

## Run it

```
.venv\Scripts\python run.py
```

Open http://127.0.0.1:8770 and press **Start**. Loading the models takes about
a minute; the page shows progress.

To give it a brain, copy `.env.example` to `.env` and set one key:

| Key | Provider | Default model |
|---|---|---|
| `MODEL_API_KEY` | Meta Model API | `muse-spark-1.3` |
| `ANTHROPIC_API_KEY` | Anthropic | `claude-opus-5-5` |

Without a key the avatar still renders, hears and speaks; it just says it has
no brain connected.

Optional flags:

```
.venv\Scripts\python run.py --compile --steps 4
```

- `--compile` runs the models through `torch.compile` (needs
  `pip install "triton-windows>=3.3,<3.4"`). Chunks render 12-16% faster. The
  first start compiles for about eight minutes; later starts reuse
  `~/.cache/talking-avatar` and take about 20 s longer than without the flag.
  (The cache is not in the project folder because triton's nested file names
  then exceed Windows' 260-character path limit.)
- `--steps N` sets the denoise steps per chunk (default 2). Four only keeps up
  with real time on this laptop when `--compile` is on.

## Using it

- **Hold Space** to talk (push-to-talk). Starting to talk cuts the avatar off.
- **Mic button** toggles hands-free: it listens for speech and stops after
  0.8 s of silence. It ignores the mic while the avatar is speaking so it
  doesn't hear itself.
- **Portrait card**: drop in any front-facing photo. Auto-crop finds the face
  and frames head and shoulders; the model only does 512x512 head shots.
- **Voice**: 54 Kokoro voices in nine languages.

## How it works

```
mic ──> faster-whisper ──> LLM (streamed) ──> sentences ──> Kokoro TTS ──┐
                                                                         v
browser <── JPEG frames + PCM audio per chunk <── SoulX-FlashHead Lite <── 24 kHz speech
```

- `app/engine.py` wraps SoulX-FlashHead Lite (1.3B-parameter audio-driven
  diffusion transformer, LTX video VAE, wav2vec2 audio features). One call
  takes 0.96 s of audio and returns the 24 frames that go with it; the model
  carries 9 frames of motion context from chunk to chunk.
- `app/server.py` runs a single render thread that owns the GPU. It always
  produces the next chunk just before the browser needs it: from queued speech
  if there is any, from silence otherwise, so the avatar blinks and shifts
  between replies. Audio and frames travel in the same message and the browser
  schedules both on its audio clock, which is what keeps lips and voice locked.
- `app/voice.py` is speech in and out, both on the CPU so the GPU stays free.
- `app/llm.py` streams the reply; text is cut into sentences and voiced as it
  arrives, so the avatar starts talking before the LLM has finished.
- `app/fast_rope.py` replaces FlashHead's float64 complex-number rotary
  embedding with the same rotation in real float32, used only with
  `--compile` because `torch.compile` cannot generate code for complex ops.
- `app/vendor_shims.py` stubs two FlashHead imports (multi-GPU `xfuser`, and a
  mediapipe face cropper replaced with OpenCV YuNet).

## Measured on this laptop

| | |
|---|---|
| Render time per 0.96 s chunk | ~720 ms (1.3x real time, 2 denoise steps) |
| Output | 512x512, 25 fps |
| VRAM | ~5.7 GB peak |
| Lip-sync offset | 0 to -1 frame (voice vs mouth motion, `scripts/sync_check.py`) |
| Typed line to first audible word | ~1.5-2 s without an LLM in the loop |
| Spoken question to first audible word | ~2.6 s plus the LLM's time to its first sentence |

Render time per 0.96 s chunk by configuration (engine alone, back to back):

| Denoise steps | Plain | `--compile` |
|---|---|---|
| 2 | 667 ms (1.44x) | 585 ms (1.64x) |
| 4 | 963 ms (1.00x) | 808 ms (1.19x) |

Inside the server, compiled four-step holds 812 ms per chunk (about 1.17x)
and a 20 s speech test played without a stall. That margin is thin: chunks
rose to about 930 ms for a few seconds while the voice was being synthesized.

Things that were not obvious:

- **Two denoise steps, not four, by default.** Uncompiled, four steps is
  exactly 1.0x real time here, with no margin. And on the sample portrait the
  extra steps buy little: frames from 2, 3 and 4 steps look the same side by
  side and measure the same sharpness (`scripts/exp_steps.py`). Motion was
  not compared.
- **`torch.compile` gives 12-16%, not 2x.** The transformer is bound by raw
  matrix arithmetic (about 3.5 TFLOP per denoise step), which compiling does
  not change. `reduce-overhead` and `max-autotune` modes were no better than
  the default mode.
- **The GPU must never go idle while streaming.** This laptop's GPU governor
  drops from 2640 MHz to about 1600 MHz the moment the load has gaps in it,
  even 45 ms per chunk, and chunks then take 1.1 s. So JPEG encoding runs on a
  second thread and the render thread spins throwaway matmuls while it waits
  (`AvatarEngine.wait`). The price is about 108 W and 80 °C whenever the avatar
  is on screen. `AVATAR_KEEP_GPU_HOT=0` turns the spinning off.
- **The first ten seconds are slower** while the clock ramps up.

The server stops rendering when the tab is hidden and after five minutes
without anyone talking.

## Scripts

```
python scripts/bench.py                     # raw engine speed + an mp4, no server
python scripts/e2e.py --say "Hello."        # headless client against a running server
python scripts/e2e.py --wav question.wav    # same, through speech recognition
python scripts/sync_check.py out/e2e.mp4    # lip-sync offset of a recorded clip
python scripts/exp_pacing.py                # the GPU-governor experiment
python scripts/exp_compile.py --steps 4 --dit default --vae default --rope real   # torch.compile timing
python scripts/exp_steps.py                 # 2 vs 3 vs 4 steps on the same speech, as images
```

## Setup from scratch

```
python -m venv .venv
.venv\Scripts\pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead vendor/SoulX-FlashHead
```

Model files, under `models/`:

| Path | From |
|---|---|
| `SoulX-FlashHead-1_3B/Model_Lite/`, `VAE_LTX/` | huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B |
| `wav2vec2-base-960h/` | huggingface.co/facebook/wav2vec2-base-960h |
| `kokoro/kokoro-v1.0.onnx`, `voices-v1.0.bin` | github.com/thewh1teagle/kokoro-onnx releases (`model-files-v1.0`) |
| `face_detection_yunet_2023mar.onnx` | github.com/opencv/opencv_zoo |

The Whisper model downloads itself on first start.

## Limits and loose ends

- Head and shoulders only, 512x512. No body, hands or camera motion.
- Response latency is seconds, not the sub-second of hosted systems: each
  chunk is 0.96 s long and takes 0.72 s to render, so speech can't start
  sooner than that after the voice is ready.
- A barge-in drops video that was generated but not yet shown, so the face
  can jump slightly at the cut.
- Not tried: a TensorRT build of the transformer, or a smaller VAE decoder
  (decode plus re-encode is 310 ms of every chunk whatever the step count).
- Licensing: the FlashHead code is Apache-2.0, but its weights are tagged
  `license: other` on Hugging Face with no license text in the repo. Fine for
  tinkering; check with Soul AI Lab before shipping anything built on them.
- Only animate faces you have the right to use.

## Credits and licence

The code in this repository is MIT-licensed (see `LICENSE`). It is glue around
other people's models, none of which are included here:

- [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) by Soul AI
  Lab: the avatar model and the sample portrait.
- [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) for the voice,
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for hearing,
  [wav2vec2](https://huggingface.co/facebook/wav2vec2-base-960h) for audio
  features and [YuNet](https://github.com/opencv/opencv_zoo) for face cropping.

The idea came from Meta's Muse Realtime Avatar demo; this is an independent
local build and shares nothing with it.
