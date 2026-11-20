from dataclasses import dataclass
from typing import Literal, Protocol

ErrorKind = Literal["network", "quota", "auth", "moderated", "empty", "other"]


class EngineError(Exception):
    def __init__(self, kind: ErrorKind, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind = kind


@dataclass
class Reply:
    heard: str  # transcript of the user utterance
    text: str  # assistant reply text
    pcm: bytes  # 24 kHz mono s16le
    gesture: str | None = None  # stage 4


class Engine(Protocol):
    async def transcribe(self, pcm16k: bytes) -> str: ...

    async def respond(self, heard: str, history: list[dict]) -> Reply:
        """Publishes "stage" events while working."""
        ...
