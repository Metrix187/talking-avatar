"""The avatar's brain: streams a short spoken reply from an LLM.

Two providers, picked from the environment (override with AVATAR_LLM=meta|anthropic):

* Meta Model API (Muse Spark)  - MODEL_API_KEY or META_API_KEY
* Anthropic (Claude)           - ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`

With neither configured the avatar still runs and says so out loud.
"""
from __future__ import annotations

import os
from typing import AsyncIterator

DEFAULT_PERSONA = (
    "You are a friendly character having a live, spoken, face-to-face conversation. "
    "Everything you write is read aloud by a text-to-speech voice and lip-synced by an "
    "animated portrait, so write the way people talk: one to three short sentences, plain "
    "words, no markdown, lists, emoji, stage directions or symbols a voice can't say. "
    "Spell out numbers and abbreviations when that reads more naturally aloud. "
    "The user's words arrive through speech recognition and may contain mishearings; "
    "go with the most sensible reading and ask a short question only when you truly can't tell."
)

NO_KEY_REPLY = (
    "I can hear you, but I don't have a brain connected yet. "
    "Set a Meta or Anthropic API key and restart me, and I'll be able to answer."
)


class _Anthropic:
    name = "anthropic"

    def __init__(self):
        import anthropic

        self.sdk = anthropic
        self.model = os.environ.get("AVATAR_LLM_MODEL", "claude-opus-5-5")
        self.effort = os.environ.get("AVATAR_LLM_EFFORT", "low")   # spoken replies: latency first
        self.client = anthropic.AsyncAnthropic()

    @staticmethod
    def configured() -> bool:
        return bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("ANTHROPIC_PROFILE")
            or os.path.isdir(os.path.expanduser("~/.config/anthropic"))
        )

    async def stream(self, persona: str, history: list[dict]) -> AsyncIterator[str]:
        sdk = self.sdk
        try:
            async with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=16000,
                system=persona,
                messages=history,
                output_config={"effort": self.effort},
                # if the model declines on policy grounds, let the API re-run the
                # request on a fallback model instead of returning nothing
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
            if final.stop_reason == "refusal":
                raise BrainError("Sorry, that's not something I can talk about.")
            # keep the full content (thinking blocks included) so later turns stay valid
            history.append({"role": "assistant", "content": final.content})
        except sdk.AuthenticationError:
            raise BrainError("My API key was rejected. Check the key and restart me.", fatal=True)
        except sdk.RateLimitError:
            raise BrainError("I'm being rate limited right now. Give me a few seconds and try again.")
        except sdk.APIStatusError as e:
            raise BrainError("Something went wrong reaching my brain. Check the server log.",
                             detail=f"{e.status_code}: {e.message}")
        except sdk.APIConnectionError:
            raise BrainError("I can't reach the network right now.")


class _Meta:
    """Meta Model API through its OpenAI-compatible Chat Completions endpoint."""
    name = "meta"

    def __init__(self):
        import openai

        self.sdk = openai
        self.model = os.environ.get("AVATAR_LLM_MODEL", "muse-spark-1.3")
        self.effort = os.environ.get("AVATAR_LLM_EFFORT", "low")
        self.client = openai.AsyncOpenAI(
            base_url=os.environ.get("MODEL_API_BASE_URL", "https://api.meta.ai/v1"),
            api_key=os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY"),
        )

    @staticmethod
    def configured() -> bool:
        return bool(os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY"))

    async def stream(self, persona: str, history: list[dict]) -> AsyncIterator[str]:
        sdk = self.sdk
        text = ""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                # Meta's docs: `developer` is the instruction role; it outranks `user`
                messages=[{"role": "developer", "content": persona}, *history],
                reasoning_effort=self.effort,
                max_completion_tokens=4000,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
                    yield chunk.choices[0].delta.content
            history.append({"role": "assistant", "content": text})
        except sdk.AuthenticationError:
            raise BrainError("My API key was rejected. Check the key and restart me.", fatal=True)
        except sdk.RateLimitError:
            raise BrainError("I'm being rate limited right now. Give me a few seconds and try again.")
        except sdk.APIStatusError as e:
            raise BrainError("Something went wrong reaching my brain. Check the server log.",
                             detail=f"{e.status_code}: {e.message}")
        except sdk.APIConnectionError:
            raise BrainError("I can't reach the network right now.")


class BrainError(Exception):
    """Carries a sentence the avatar can say instead of the reply."""

    def __init__(self, spoken: str, detail: str | None = None, fatal: bool = False):
        super().__init__(detail or spoken)
        self.spoken, self.detail, self.fatal = spoken, detail, fatal


def _pick_provider():
    want = os.environ.get("AVATAR_LLM", "auto").lower()
    order = {"meta": [_Meta], "anthropic": [_Anthropic]}.get(want, [_Meta, _Anthropic])
    for cls in order:
        if cls.configured():
            return cls()
    return None


class Brain:
    def __init__(self, persona: str | None = None):
        self.persona = persona or os.environ.get("AVATAR_PERSONA") or DEFAULT_PERSONA
        self.history: list[dict] = []
        self.error: str | None = None
        try:
            self.provider = _pick_provider()
        except Exception as e:
            self.provider, self.error = None, str(e)

    @property
    def available(self) -> bool:
        return self.provider is not None

    def describe(self) -> dict:
        p = self.provider
        return {"available": p is not None, "provider": p.name if p else None,
                "model": p.model if p else None, "error": self.error}

    def reset(self):
        self.history.clear()

    async def reply(self, user_text: str) -> AsyncIterator[str]:
        """Yield the reply as text deltas. If the consumer stops early (barge-in),
        call `commit_partial()` with what was actually spoken."""
        self.history.append({"role": "user", "content": user_text})
        if self.provider is None:
            self.history.pop()
            yield NO_KEY_REPLY
            return
        try:
            async for delta in self.provider.stream(self.persona, self.history):
                yield delta
        except BrainError as e:
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            self.error = e.detail
            if e.fatal:
                self.provider = None
            yield e.spoken

    def commit_partial(self, spoken: str):
        """Record an interrupted reply as only what the user actually heard."""
        if self.history and self.history[-1]["role"] == "user":
            if spoken.strip():
                self.history.append({"role": "assistant", "content": spoken.strip()})
            else:
                self.history.pop()
