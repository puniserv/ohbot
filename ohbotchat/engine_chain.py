import io
import wave
from collections.abc import Awaitable, Callable
from contextlib import contextmanager

import openai

from .config import Config
from .engine import EngineError, Reply
from .events import EventBus


def _wav(pcm16k: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16k)
    return buf.getvalue()


def _is_quota(e: openai.RateLimitError) -> bool:
    body = e.body if isinstance(e.body, dict) else {}
    inner = body.get("error") if isinstance(body.get("error"), dict) else body
    return "insufficient_quota" in (getattr(e, "code", None), inner.get("code"), inner.get("type"))


@contextmanager
def _mapped_errors():
    try:
        yield
    except EngineError:
        raise
    except (openai.APIConnectionError, openai.APITimeoutError) as e:  # timeout is a subclass of connection error
        raise EngineError("network", str(e)) from e
    except openai.RateLimitError as e:
        raise EngineError("quota" if _is_quota(e) else "other", str(e)) from e
    except openai.AuthenticationError as e:
        raise EngineError("auth", str(e)) from e
    except openai.OpenAIError as e:
        raise EngineError("other", str(e)) from e


class ChainEngine:
    """STT -> moderation -> chat -> moderation -> TTS over an injected AsyncOpenAI-like client."""

    def __init__(self, client, config: Config, bus: EventBus) -> None:
        self._client, self._cfg, self._bus = client, config, bus

    async def _stage(self, name: str, call: Callable[[], Awaitable]):
        self._bus.publish("stage", name=name, status="start")
        with _mapped_errors():
            result = await call()
        self._bus.publish("stage", name=name, status="done")
        return result

    def _persona(self) -> str:
        path = self._cfg.root / "personas" / f"{self._cfg.conversation.persona}.md"
        return path.read_text(encoding="utf-8")

    async def transcribe(self, pcm16k: bytes) -> str:
        c = self._cfg.openai
        # gpt-transcribe takes a list of languages, older models a single hint
        lang = {"languages": ["pl"]} if c.stt_model.startswith("gpt-transcribe") else {"language": "pl"}
        r = await self._stage(
            "stt",
            lambda: self._client.audio.transcriptions.create(
                model=c.stt_model, file=("speech.wav", _wav(pcm16k)), timeout=c.timeout_s, **lang
            ),
        )
        text = (r.text or "").strip()
        if not text:
            raise EngineError("empty")
        return text

    async def _moderate(self, name: str, text: str) -> None:
        c = self._cfg.openai
        r = await self._stage(
            name, lambda: self._client.moderations.create(model=c.moderation_model, input=text, timeout=c.timeout_s)
        )
        if r.results[0].flagged:
            raise EngineError("moderated")

    async def synthesize(self, text: str) -> bytes:
        c = self._cfg.openai
        speech = await self._stage(
            "tts",
            lambda: self._client.audio.speech.create(
                model=c.tts_model, voice=c.tts_voice, input=text, response_format="pcm", timeout=c.timeout_s
            ),
        )
        return speech.content

    async def respond(self, heard: str, history: list[dict]) -> Reply:
        c = self._cfg.openai
        await self._moderate("moderation_in", heard)
        messages = [{"role": "system", "content": self._persona()}, *history, {"role": "user", "content": heard}]
        r = await self._stage(
            "chat",
            lambda: self._client.chat.completions.create(
                model=c.chat_model, messages=messages, max_completion_tokens=c.max_tokens, timeout=c.timeout_s
            ),
        )
        text = (r.choices[0].message.content or "").strip()
        if not text:
            raise EngineError("other", "empty chat reply")
        await self._moderate("moderation_out", text)
        return Reply(heard=heard, text=text, pcm=await self.synthesize(text))
