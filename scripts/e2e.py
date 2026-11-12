"""Headless end-to-end check against a running server.

Connects like the browser does, plays the role of its audio clock (acks each
chunk with the remaining buffer), asks the avatar to speak, and records what
comes back to an mp4. Reports chunk timing, whether the stream ever ran dry,
and how long speech took to start.

    python scripts/e2e.py --say "Hello there."          # TTS + lip-sync only
    python scripts/e2e.py --ask "What's your name?"     # through the LLM
    python scripts/e2e.py --wav question.wav            # through speech recognition too
"""
import argparse
import asyncio
import json
import struct
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("--url", default="ws://127.0.0.1:8770/ws")
p.add_argument("--say")
p.add_argument("--ask")
p.add_argument("--wav")
p.add_argument("--interrupt-after", type=float, help="seconds of speech to hear before barging in")
p.add_argument("--idle", type=float, default=3.0, help="seconds of idle video before and after")
p.add_argument("--out", default=str(ROOT / "out" / "e2e.mp4"))
args = p.parse_args()


async def main():
    frames, audio, events, gens, dry = [], [], [], [], []
    async with websockets.connect(args.url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        chunk_s, rate = hello["chunk_s"], 24000
        print("brain:", hello["brain"])
        t0 = time.monotonic()
        play_until = t0                     # when our pretend speaker runs out
        sent_at = first_speech = speech_end = None
        interrupted = False
        stage = "pre"

        while True:
            now = time.monotonic()
            if stage == "pre" and now - t0 > args.idle:
                stage = "wait"
                sent_at = now
                if args.wav:
                    pcm, sr = sf.read(args.wav, dtype="int16", always_2d=True)
                    assert sr == 16000, "question wav must be 16 kHz"
                    await ws.send(json.dumps({"type": "mic_start"}))
                    await ws.send(pcm[:, 0].tobytes())
                    await ws.send(json.dumps({"type": "mic_end"}))
                elif args.ask:
                    await ws.send(json.dumps({"type": "text", "text": args.ask}))
                else:
                    await ws.send(json.dumps({"type": "say", "text": args.say or "Hello! This is a quick test of my voice and my lips."}))
            if stage == "post" and now - speech_end > args.idle:
                break
            if now - t0 > 90:
                print("TIMEOUT"); break

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            now = time.monotonic()
            if isinstance(msg, str):
                m = json.loads(msg)
                events.append((round(now - t0, 2), m))
                if m["type"] == "flush":
                    play_until = now
                continue
            hl = struct.unpack_from("<I", msg)[0]
            head = json.loads(msg[4:4 + hl])
            off = 4 + hl
            pcm = np.frombuffer(msg[off:off + head["audio"]], dtype="<i2") if head["audio"] else np.zeros(int(chunk_s * rate), dtype="<i2")
            off += head["audio"]
            for n in head["frames"]:
                frames.append(cv2.imdecode(np.frombuffer(msg[off:off + n], np.uint8), cv2.IMREAD_COLOR))
                off += n
            audio.append(pcm)
            gens.append(head["gen_ms"])
            print(f"  chunk {head['seq']:3d} {'speech' if head['speaking'] else 'idle  '} render {head['gen_ms']} ms + encode {head.get('encode_ms')} ms", flush=True)
            if play_until < now - 0.02 and len(gens) > 3:
                # the browser would have frozen on the last frame for this long
                dry.append((head["seq"], "speech" if head["speaking"] else "idle", round(now - play_until, 2)))
            start = max(play_until, now + 0.06)
            play_until = start + chunk_s
            await ws.send(json.dumps({"type": "ack", "seq": head["seq"], "ahead": play_until - now}))

            if head["speaking"] and first_speech is None:
                first_speech = start
            if head["speaking"] and args.interrupt_after and not interrupted and start - first_speech >= args.interrupt_after:
                interrupted = True
                await ws.send(json.dumps({"type": "interrupt"}))
                print(f"barged in at {now - t0:.2f}s")
            if stage == "wait" and first_speech is not None and not head["speaking"]:
                stage, speech_end = "post", now

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    wav = out.with_suffix(".wav")
    sf.write(wav, np.concatenate(audio), 24000)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25",
         "-i", "-", "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        stdin=subprocess.PIPE)
    proc.stdin.write(np.stack(frames).tobytes()); proc.stdin.close(); proc.wait()
    wav.unlink()

    warm = gens[3:] or gens
    print(f"chunks: {len(gens)}  render: mean {np.mean(warm):.0f} ms, worst {max(warm)} ms per {chunk_s:.2f}s chunk "
          f"({chunk_s * 1000 / np.mean(warm):.2f}x real time)")
    print(f"stream ran dry {len(dry)} time(s)"
          + "".join(f"\n    before {kind} chunk {seq}: frozen {gap}s" for seq, kind, gap in dry))
    if first_speech:
        print(f"request -> first audible word: {first_speech - sent_at:.2f}s")
    for t, m in events:
        if m["type"] in ("state", "user", "flush", "error", "asleep") or (m["type"] == "assistant" and m.get("final")):
            print(f"  {t:6.2f}s  {m}")
    print("wrote", out)


asyncio.run(main())
