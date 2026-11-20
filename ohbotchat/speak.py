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
        self._stop = threading.Event()
        self._stream = None

    async def play(self, pcm: bytes, rate: int = 24000, on_frame: Callable[[int], None] | None = None) -> bool:
        """True if played to the end, False if stop() interrupted it. `on_frame` runs in the worker thread."""
        self._stop.clear()
        return await asyncio.to_thread(self._run, pcm, rate, on_frame)

    def stop(self) -> None:
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()  # a blocked write() returns at once instead of draining its buffer
            except Exception:
                pass

    def _run(self, pcm: bytes, rate: int, on_frame) -> bool:
        step = rate * FRAME_MS // 1000 * 2
        stream = self._stream = self._factory(rate)
        try:
            stream.start()
            for index, i in enumerate(range(0, len(pcm), step)):
                if self._stop.is_set():
                    break
                if on_frame:
                    on_frame(index)
                stream.write(pcm[i : i + step])
            if not self._stop.is_set():
                stream.stop()  # drains what is still buffered
            return not self._stop.is_set()
        except Exception:
            if self._stop.is_set():  # abort() made the pending write fail
                return False
            raise
        finally:
            self._stream = None
            try:
                stream.close()
            except Exception:
                pass
