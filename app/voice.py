"""Local speech in and out: faster-whisper for hearing, Kokoro for talking.
Both run on the CPU so the GPU stays free for the avatar."""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TTS_RATE = 24000
STT_RATE = 16000


class Speaker:
    """Kokoro text-to-speech. Returns float32 mono at 24 kHz."""

    def __init__(self):
        from kokoro_onnx import Kokoro

        d = ROOT / "models" / "kokoro"
        self.kokoro = Kokoro(str(d / "kokoro-v1.0.onnx"), str(d / "voices-v1.0.bin"))
        self.voices = sorted(self.kokoro.get_voices())
        self._lock = threading.Lock()
        self.say("Ready.", self.default_voice)        # first call is slow; pay for it at startup

    @property
    def default_voice(self) -> str:
        want = os.environ.get("AVATAR_VOICE", "af_heart")
        return want if want in self.voices else self.voices[0]

    def say(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        lang = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it",
                "j": "ja", "p": "pt-br", "z": "cmn"}.get(voice[:1], "en-us")
        with self._lock:
            pcm, rate = self.kokoro.create(text, voice=voice, speed=speed, lang=lang)
        assert rate == TTS_RATE
        return np.asarray(pcm, dtype=np.float32)


class Listener:
    """faster-whisper speech-to-text. Takes float32 mono at 16 kHz."""

    def __init__(self):
        from faster_whisper import WhisperModel

        name = os.environ.get("AVATAR_STT_MODEL", "base.en")   # small.en hears better but adds ~1 s
        self.multilingual = not name.endswith(".en")
        self.model = WhisperModel(name, device="cpu", compute_type="int8",
                                  cpu_threads=min(8, os.cpu_count() or 4))

    def hear(self, audio: np.ndarray) -> str:
        if len(audio) < STT_RATE * 0.25:
            return ""
        segments, _ = self.model.transcribe(
            audio, language=None if self.multilingual else "en", beam_size=1,
            vad_filter=True, condition_on_previous_text=False, without_timestamps=True,
        )
        # no_speech_prob guards against whisper inventing words from room noise
        return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()


_SENTENCE_END = re.compile(r"([.!?…]+[\"')\]]*)(\s+|$)")
_CLAUSE_END = re.compile(r"([,;:—])\s+")


def pop_sentence(buf: str, first: bool = False) -> tuple[str | None, str]:
    """Split the first speakable piece off streamed text. For the very first
    piece of a reply a long clause is enough, to get the voice going sooner."""
    for m in _SENTENCE_END.finditer(buf):
        end = m.end(1)
        # "3.5", "e.g. x": a period hugging a non-space is not a sentence end,
        # and a boundary at the very end of the buffer may just be mid-stream
        if m.group(2) == "" or len(buf[:end].strip()) < 2:
            continue
        return buf[:end].strip(), buf[m.end():]
    if first and len(buf) > 60:
        for m in _CLAUSE_END.finditer(buf):
            if m.end(1) > 25:
                return buf[: m.end(1)].strip(), buf[m.end():]
    return None, buf


_UNSPEAKABLE = re.compile(r"[*_#`~<>|\[\]{}\\]+")


def clean_for_speech(text: str) -> str:
    return re.sub(r"\s+", " ", _UNSPEAKABLE.sub(" ", text)).strip()
