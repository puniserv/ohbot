import asyncio
import threading
from collections.abc import Callable

FRAME_MS = 40


def _default_stream(rate: int):
    import sounddevice as sd

    return sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16")


class Speaker:
    """Plays 16-bit mono PCM; `stream_factory(rate)` returns an object with start/write/stop/abort/close."""

    def __init__(self, stream_factory: Callable[[int], object] = _default_stream) -> None:
        self._factory = stream_factory
        self._stop = threading.Event()  # one per play(): a cancelled to_thread keeps its worker running

    async def play(self, pcm: bytes, rate: int = 24000, on_frame: Callable[[int], None] | None = None) -> bool:
        """True if played to the end, False if stop() interrupted it. `on_frame` runs in the worker thread."""
        self._stop.set()  # one voice at a time: a worker left over from a cancelled play() stops now
        self._stop = stop = threading.Event()
        return await asyncio.to_thread(self._run, pcm, rate, on_frame, stop)

    def stop(self) -> None:
        # only a flag: abort() from another thread while write() blocks makes CoreAudio's write never return
        self._stop.set()

    def _run(self, pcm: bytes, rate: int, on_frame, stop: threading.Event) -> bool:
        step = rate * FRAME_MS // 1000 * 2
        stream = self._factory(rate)
        try:
            stream.start()
            for index, i in enumerate(range(0, len(pcm), step)):
                if stop.is_set():
                    break
                if on_frame:
                    on_frame(index)
                stream.write(pcm[i : i + step])
            if stop.is_set():
                stream.abort()  # drops what is still buffered; a write is only one frame, so this comes quickly
                return False
            stream.stop()  # drains what is still buffered
            return True
        finally:
            try:
                stream.close()
            except Exception:
                pass
