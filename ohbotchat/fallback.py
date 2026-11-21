import hashlib
from pathlib import Path

from .engine import EngineError

PHRASES = {
    "no_network": "Nie mam połączenia, spróbuj za chwilę.",
    "not_heard": "Nie usłyszałem, powtórz proszę.",
    "moderated": "Porozmawiajmy o czymś innym.",
    "error": "Coś mi się pomyliło, spróbuj jeszcze raz.",
    "listening": "Słucham?",
}


class Fallback:
    """Pre-generated speech (24 kHz PCM) for one TTS voice: error phrases that must work offline, fillers, greeting."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def get(self, key: str) -> bytes | None:
        path = self.dir / f"{key}.pcm"
        return path.read_bytes() if path.is_file() else None

    def _text_key(self, text: str) -> str:
        return "say-" + hashlib.sha256(text.encode()).hexdigest()[:16]

    def cached(self, text: str) -> bytes | None:
        return self.get(self._text_key(text))

    def store(self, text: str, pcm: bytes) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{self._text_key(text)}.pcm").write_bytes(pcm)

    def complete(self, fillers: list[str]) -> bool:
        return all(self.get(k) is not None for k in PHRASES) and all(self.cached(t) is not None for t in fillers)

    async def ensure(self, engine, fillers: list[str]) -> None:
        """Generates missing phrases; offline, or a flagged filler, just leaves them missing.

        Fillers are operator text, so they go through `engine.say` and its moderation; the fixed phrases do not.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        todo = [(f"{k}.pcm", t, engine.synthesize) for k, t in PHRASES.items()]
        todo += [(f"{self._text_key(t)}.pcm", t, engine.say) for t in fillers]
        for name, text, speak in todo:
            path = self.dir / name
            if path.is_file():
                continue
            try:
                pcm = await speak(text, publish=False)
            except EngineError:
                continue
            path.write_bytes(pcm)
