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
    """Pre-generated spoken error phrases (24 kHz PCM) so the robot can answer with the network down."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def get(self, key: str) -> bytes | None:
        path = self._dir / f"{key}.pcm"
        return path.read_bytes() if path.is_file() else None

    def complete(self) -> bool:
        return all(self.get(k) is not None for k in PHRASES)

    async def ensure(self, engine) -> None:
        """Generates missing phrases via `engine.synthesize`; offline just leaves them missing."""
        self._dir.mkdir(parents=True, exist_ok=True)
        for key, text in PHRASES.items():
            if self.get(key) is not None:
                continue
            try:
                pcm = await engine.synthesize(text, publish=False)
            except EngineError:
                continue
            (self._dir / f"{key}.pcm").write_bytes(pcm)
