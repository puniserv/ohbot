from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

GESTURES = ("none", "yes", "no", "surprised", "happy", "thinking")
ErrorKind = Literal["network", "quota", "auth", "moderated", "empty", "nokey", "other"]


class EngineError(Exception):
    def __init__(self, kind: ErrorKind, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind = kind


@dataclass
class Reply:
    text: str  # assistant reply text
    pcm: bytes  # 24 kHz mono s16le
    gesture: str | None = None  # one of GESTURES


class Engine(Protocol):
    async def transcribe(self, pcm16k: bytes) -> str: ...

    async def respond(self, heard: str, history: list[dict], on_safe: Callable[[], None] | None = None) -> Reply:
        """Publishes "stage" events while working; calls `on_safe` once `heard` passed moderation."""
        ...

    async def say(self, text: str, voice: str | None = None, publish: bool = True) -> bytes: ...
