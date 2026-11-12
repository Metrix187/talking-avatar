"""Talking-avatar server.

    python -m app.server            ->  http://127.0.0.1:8770

One browser tab at a time talks to it over a WebSocket:

  server -> client   binary: one 0.96 s chunk = header JSON + 24 kHz PCM16 audio + 24 JPEG frames
                     text:   JSON events (state, captions, flush)
  client -> server   binary: 16 kHz PCM16 microphone audio while the user is talking
                     text:   JSON commands (mic_start, mic_end, text, interrupt, ack, ...)

A single render thread owns the GPU. It always generates the next chunk just
before the browser needs it: from queued speech if there is any, from silence
otherwise, so the avatar keeps breathing and blinking between replies.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import queue
import struct
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps
from scipy.signal import resample_poly

from .llm import Brain
from .voice import STT_RATE, TTS_RATE, clean_for_speech, pop_sentence

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
DATA = ROOT / "data"
DEFAULT_PORTRAIT = ROOT / "vendor" / "SoulX-FlashHead" / "examples" / "girl.png"
HOST = os.environ.get("AVATAR_HOST", "127.0.0.1")
PORT = int(os.environ.get("AVATAR_PORT", "8770"))
JPEG_QUALITY = 88
IDLE_SLEEP_S = 300            # stop rendering after this long without anyone talking
SENTENCE_GAP_S = 0.12
HOLD_FOR_SPEECH_S = 1.0       # longest the picture may freeze waiting for the first words



@contextlib.asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=Models.load, daemon=True).start()
    threading.Thread(target=render_loop, daemon=True).start()
    threading.Thread(target=send_loop, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


# --------------------------------------------------------------------------
# models (loaded once, in the background, so the page can show progress)
# --------------------------------------------------------------------------
class Models:
    engine = None
    speaker = None
    listener = None
    stage = "starting"
    error: str | None = None
    ready = threading.Event()

    @classmethod
    def load(cls):
        try:
            cls.stage = "loading the avatar model"
            from .engine import AvatarEngine

            cls.engine = AvatarEngine()
            compiling = cls.engine.compiled
            cls.stage = "compiling the model, about 8 minutes the first time" if compiling else "preparing the portrait"
            saved = DATA / "portrait.png"            # stored already cropped
            try:
                cls.engine.set_portrait(Image.open(saved if saved.exists() else DEFAULT_PORTRAIT), face_crop=False)
            except Exception:
                cls.engine.set_portrait(Image.open(DEFAULT_PORTRAIT), face_crop=False)
            if not compiling:
                cls.stage = "warming up the GPU"
            cls.engine.warmup(4)
            cls.stage = "loading the voice"
            from .voice import Listener, Speaker

            cls.speaker = Speaker()
            cls.stage = "loading speech recognition"
            cls.listener = Listener()
            cls.stage = "ready"
            cls.ready.set()
        except Exception as e:  # surface it in the UI instead of dying silently
            import traceback

            traceback.print_exc()
            cls.error = f"{type(e).__name__}: {e}"
            cls.stage = "failed"


# --------------------------------------------------------------------------
# per-connection state
# --------------------------------------------------------------------------
class SpeechBuffer:
    """Thread-safe FIFO of 24 kHz speech waiting to be lip-synced."""

    def __init__(self):
        self._lock = threading.Lock()
        self._parts: list[np.ndarray] = []
        self._n = 0
        self._expecting = 0
        self._synth = 0                       # sentences queued for, or in, the TTS engine

    def push(self, pcm: np.ndarray):
        with self._lock:
            self._parts.append(pcm)
            self._n += len(pcm)

    def expect(self, delta: int, synth: bool = False):
        with self._lock:
            self._expecting = max(0, self._expecting + delta)
            if synth:
                self._synth = max(0, self._synth + delta)

    def status(self) -> tuple[int, bool, bool]:
        """(samples waiting, more speech is coming, a sentence is being voiced right now)"""
        with self._lock:
            return self._n, self._expecting > 0, self._synth > 0

    def take(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            filled = 0
            while filled < n and self._parts:
                part = self._parts[0]
                k = min(len(part), n - filled)
                out[filled:filled + k] = part[:k]
                filled += k
                if k == len(part):
                    self._parts.pop(0)
                else:
                    self._parts[0] = part[k:]
            self._n -= filled
        return out

    def clear(self):
        with self._lock:
            self._parts.clear()
            self._n = 0
            self._expecting = 0
            self._synth = 0


class Session:
    def __init__(self, ws: WebSocket, loop: asyncio.AbstractEventLoop):
        self.ws, self.loop = ws, loop
        self.alive = True
        self.turn = 0                         # bumps on every interruption
        self.speech = SpeechBuffer()
        self.tts_q: queue.Queue = queue.Queue()
        self.brain = Brain()
        self.voice = Models.speaker.default_voice
        self.mic = bytearray()
        self.mic_open = False
        self.reply_task: asyncio.Task | None = None
        self.spoken: list[str] = []           # sentences of the current reply already voiced
        self.seq = 0
        self.buffered_until = 0.0             # monotonic time the browser runs out of video
        self.last_activity = time.monotonic()
        self.paused = False
        self.asleep = False
        self.was_speaking = False
        self.hold_since: float | None = None
        self.state = "idle"
        threading.Thread(target=self._tts_worker, daemon=True).start()

    # -- outbound ---------------------------------------------------------
    def send_json(self, **msg):
        if self.alive:
            asyncio.run_coroutine_threadsafe(self._send(json.dumps(msg)), self.loop)

    def send_bytes(self, data: bytes):
        if self.alive:
            asyncio.run_coroutine_threadsafe(self._send(data), self.loop)

    async def _send(self, data):
        try:
            if isinstance(data, str):
                await self.ws.send_text(data)
            else:
                await self.ws.send_bytes(data)
        except Exception:
            self.alive = False

    def set_state(self, state: str):
        if state != self.state:
            self.state = state
            self.send_json(type="state", state=state)

    # -- speech -----------------------------------------------------------
    def say(self, text: str):
        text = clean_for_speech(text)
        if text:
            self.speech.expect(+1, synth=True)
            self.tts_q.put((self.turn, text))

    def _tts_worker(self):
        gap = np.zeros(int(SENTENCE_GAP_S * TTS_RATE), dtype=np.float32)
        while self.alive:
            try:
                turn, text = self.tts_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if turn != self.turn:
                continue                       # interrupted before it was voiced; clear() reset the count
            try:
                pcm = Models.speaker.say(text, self.voice)
                if turn == self.turn:
                    self.speech.push(np.concatenate([pcm, gap]))
                    self.spoken.append(text)
            except Exception as e:
                self.send_json(type="error", message=f"speech synthesis failed: {e}")
            finally:
                if turn == self.turn:
                    self.speech.expect(-1, synth=True)

    def interrupt(self, tell_client: bool = True):
        """Stop talking right now (the user barged in)."""
        self.turn += 1
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        self.speech.clear()
        self.brain.commit_partial(" ".join(self.spoken))
        self.spoken = []
        self.buffered_until = 0.0
        if tell_client:
            self.send_json(type="flush")

    def touch(self):
        self.last_activity = time.monotonic()
        if self.asleep:
            self.asleep = False
            self.send_json(type="awake")

    # -- a conversational turn ---------------------------------------------
    def start_reply(self, user_text: str):
        """Drop whatever is being said and answer this instead."""
        self.interrupt(tell_client=self.state == "speaking")
        self.reply_task = asyncio.create_task(self.respond(user_text))

    async def respond(self, user_text: str):
        turn = self.turn
        self.send_json(type="user", text=user_text)
        self.set_state("thinking")
        self.speech.expect(+1)                 # hold chunk padding while the reply streams in
        full, buf, first, said = "", "", True, False
        try:
            async with contextlib.aclosing(self.brain.reply(user_text)) as stream:
                async for delta in stream:
                    if turn != self.turn:
                        return
                    full += delta
                    buf += delta
                    self.send_json(type="assistant", text=full, final=False)
                    while True:
                        sentence, buf = pop_sentence(buf, first)
                        if sentence is None:
                            break
                        first = False
                        said = True
                        self.say(sentence)
            if turn == self.turn:
                self.say(buf)
                self.send_json(type="assistant", text=full, final=True)
                if not said and not clean_for_speech(buf):
                    self.set_state("idle")         # nothing to say; don't sit on "thinking"
        finally:
            if turn == self.turn:
                self.speech.expect(-1)


class Stage:
    """The one session currently allowed to drive the GPU."""
    session: Session | None = None
    portrait_busy = threading.Lock()


# --------------------------------------------------------------------------
# render loop
# --------------------------------------------------------------------------
def render_loop():
    Models.ready.wait()
    eng = Models.engine
    n24 = int(eng.chunk_seconds * TTS_RATE)
    gen_s = 0.7                                   # running estimate of time per chunk
    while True:
        s = Stage.session
        if s is None or not s.alive or s.paused:
            time.sleep(0.05)
            continue
        now = time.monotonic()
        have, expecting, synth = s.speech.status()

        if have == 0 and not expecting and now - s.last_activity > IDLE_SLEEP_S:
            if not s.asleep:
                s.asleep = True
                s.send_json(type="asleep")
            time.sleep(0.1)
            continue

        # Start a chunk only when the browser is about to need it: late enough
        # that a reply can still jump in, early enough to finish in time.
        lead = gen_s * 1.15 + 0.2
        if have > 0:
            lead = max(lead, 1.6)                 # while talking, keep a little more in hand
        ahead = s.buffered_until - now
        if ahead > lead:
            eng.wait(min(0.01, ahead - lead))
            continue
        if 0 < have < n24 and expecting and ahead > gen_s * 1.15 + 0.1:
            eng.wait(0.005)                       # more words are on the way; don't pad yet
            continue
        if have == 0 and synth:
            # The first words of a reply are a fraction of a second from being
            # voiced. Starting an idle chunk now would push them back a whole
            # chunk, so hold the last frame briefly instead.
            if s.hold_since is None:
                s.hold_since = now
            if now - s.hold_since < HOLD_FOR_SPEECH_S:
                eng.wait(0.005)
                continue
        s.hold_since = None

        pcm24 = s.speech.take(n24) if have > 0 else None
        turn = s.turn
        t0 = time.perf_counter()
        try:
            with Stage.portrait_busy:
                audio16 = resample_poly(pcm24, 2, 3).astype(np.float32) if pcm24 is not None else None
                frames = eng.step(audio16)
        except Exception as e:
            s.send_json(type="error", message=f"render failed: {e}")
            time.sleep(1.0)
            continue
        step_s = time.perf_counter() - t0
        gen_s = 0.8 * gen_s + 0.2 * (step_s + 0.05)

        if turn != s.turn and pcm24 is not None:
            continue                              # interrupted mid-render: never play stale speech

        s.seq += 1
        s.buffered_until = max(s.buffered_until, time.monotonic()) + eng.chunk_seconds
        # JPEG encoding and the socket write happen on another thread, so this
        # one goes straight back to the GPU (see AvatarEngine.wait for why even
        # a short gap matters).
        _outbox.put((s, s.seq, turn, frames, pcm24, step_s, expecting))


_outbox: queue.Queue = queue.Queue()


def send_loop():
    while True:
        s, seq, turn, frames, pcm24, step_s, expecting = _outbox.get()
        if not s.alive:
            continue
        speaking = pcm24 is not None
        if speaking and turn != s.turn:
            continue                              # interrupted while waiting to be sent
        t0 = time.perf_counter()
        jpegs = [
            cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
            for f in np.ascontiguousarray(frames[..., ::-1])
        ]
        audio = (np.clip(pcm24, -1, 1) * 32767).astype("<i2").tobytes() if speaking else b""
        header = json.dumps({
            "seq": seq, "fps": 25, "rate": TTS_RATE, "audio": len(audio),
            "frames": [len(j) for j in jpegs], "speaking": speaking,
            "gen_ms": round(step_s * 1000), "encode_ms": round((time.perf_counter() - t0) * 1000),
            "turn": turn,
        }).encode()
        s.send_bytes(b"".join([struct.pack("<I", len(header)), header, audio, *jpegs]))

        if speaking:
            s.was_speaking = True
            s.set_state("speaking")
        elif s.was_speaking and not expecting:
            s.was_speaking = False
            s.spoken = []
            s.set_state("listening" if s.mic_open else "idle")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _same_origin(headers) -> bool:
    """Refuse cross-site pages: this server holds a mic transcript and a GPU."""
    origin = headers.get("origin")
    if not origin:
        return True
    return urlparse(origin).netloc == headers.get("host")




@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    brain = Brain().describe() if Models.ready.is_set() else None
    return {"stage": Models.stage, "ready": Models.ready.is_set(), "error": Models.error, "brain": brain}


@app.get("/api/portrait.jpg")
def portrait_jpg():
    if not Models.ready.is_set():
        raise HTTPException(503, "still loading")
    buf = io.BytesIO()
    Models.engine.portrait_preview().save(buf, "JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/portrait")
async def set_portrait(request: Request, crop: int = 1):
    if not _same_origin(request.headers):
        raise HTTPException(403, "cross-origin request refused")
    if not Models.ready.is_set():
        raise HTTPException(503, "still loading")
    body = await request.body()
    if not body or len(body) > 25 * 2**20:
        raise HTTPException(413, "image must be under 25 MB")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(body))).convert("RGB")
    except Exception:
        raise HTTPException(400, "that file is not an image I can read")

    def apply():
        with Stage.portrait_busy:
            Models.engine.set_portrait(image, face_crop=bool(crop))
            preview = Models.engine.portrait_preview()
        DATA.mkdir(exist_ok=True)
        preview.save(DATA / "portrait.png")       # already cropped; reload without re-detecting
        return preview

    s = Stage.session
    if s:
        s.interrupt()
    try:
        await asyncio.get_running_loop().run_in_executor(None, apply)
    except ValueError as e:                       # no face found
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)
    return {"ok": True}


# --------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not _same_origin(ws.headers):
        await ws.close(code=4403)
        return
    await ws.accept()
    if not Models.ready.is_set():
        await ws.send_text(json.dumps({"type": "error", "message": "still loading"}))
        await ws.close()
        return

    loop = asyncio.get_running_loop()
    old = Stage.session
    if old is not None:
        old.alive = False
        with contextlib.suppress(Exception):
            await old.ws.close(code=4000)
    s = Session(ws, loop)
    eng = Models.engine
    await ws.send_text(json.dumps({
        "type": "hello", "chunk_s": eng.chunk_seconds, "fps": 25, "size": 512,
        "voices": Models.speaker.voices, "voice": s.voice, "brain": s.brain.describe(),
        "mic_rate": STT_RATE,
    }))
    Stage.session = s

    async def handle_utterance():
        pcm = np.frombuffer(bytes(s.mic), dtype="<i2").astype(np.float32) / 32768
        s.mic.clear()
        if len(pcm) < STT_RATE * 0.3:
            s.set_state("idle")
            return
        s.set_state("thinking")
        text = await loop.run_in_executor(None, Models.listener.hear, pcm)
        if not text:
            s.set_state("idle")
            return
        s.start_reply(text)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if s.mic_open and len(s.mic) < STT_RATE * 2 * 60:      # cap one utterance at 60 s
                    s.mic.extend(msg["bytes"])
                continue
            try:
                m = json.loads(msg.get("text") or "{}")
            except json.JSONDecodeError:
                continue
            kind = m.get("type")
            if kind == "ack":
                # the browser reports how much video it still has queued; trust it over our estimate
                behind = max(0, s.seq - int(m.get("seq", s.seq)))
                s.buffered_until = time.monotonic() + float(m.get("ahead", 0)) + behind * eng.chunk_seconds
            elif kind == "mic_start":
                s.touch()
                s.mic.clear()
                s.mic_open = True
                if m.get("interrupt", True):
                    s.interrupt()
                s.set_state("listening")
            elif kind == "mic_end":
                s.touch()
                s.mic_open = False
                if m.get("discard"):
                    s.mic.clear()
                    s.set_state("idle")
                else:
                    await handle_utterance()
            elif kind == "text":
                s.touch()
                text = str(m.get("text", "")).strip()[:2000]
                if text:
                    s.start_reply(text)
            elif kind == "say":                    # speak a line verbatim, no LLM
                s.touch()
                s.interrupt()
                rest = str(m.get("text", ""))[:2000]
                while True:                        # voice it sentence by sentence so it starts sooner
                    sentence, rest = pop_sentence(rest)
                    if sentence is None:
                        break
                    s.say(sentence)
                s.say(rest)
            elif kind == "interrupt":
                s.touch()
                s.interrupt()
                s.set_state("idle")
            elif kind == "voice":
                if m.get("voice") in Models.speaker.voices:
                    s.voice = m["voice"]
            elif kind == "reset":
                s.interrupt()
                s.brain.reset()
            elif kind == "pause":
                s.paused = bool(m.get("paused"))
                if not s.paused:
                    s.buffered_until = 0.0
                    s.touch()
            elif kind == "wake":
                s.touch()
    except WebSocketDisconnect:
        pass
    finally:
        s.alive = False
        if s.reply_task:
            s.reply_task.cancel()
        if Stage.session is s:
            Stage.session = None


def main():
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
