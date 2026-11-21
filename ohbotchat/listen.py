import re
import unicodedata
from collections import deque
from collections.abc import Callable

from .audio import level
from .config import Config
from .events import EventBus

RATE = 16000
FRAME_MS = 30
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2
LEAD_IN_FRAMES = 5  # keeps the soft start of a word the VAD only notices a frame late


def _vad_is_speech(aggressiveness: int) -> Callable[[bytes], bool]:
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    return lambda frame: vad.is_speech(frame, RATE)


class Segmenter:
    """Cuts a stream of 30 ms frames into utterances: speech, then `silence_ms` of quiet."""

    def __init__(
        self,
        aggressiveness: int = 2,
        silence_ms: int = 800,
        min_speech_ms: int = 300,
        is_speech: Callable[[bytes], bool] | None = None,
    ) -> None:
        self._is_speech = is_speech or _vad_is_speech(aggressiveness)
        self._silence_frames = max(1, silence_ms // FRAME_MS)
        self._min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._lead: deque[bytes] = deque(maxlen=LEAD_IN_FRAMES)
        self._reset()

    def _reset(self) -> None:
        self._frames: list[bytes] = []
        self._speech = self._quiet = 0

    def reset(self) -> None:
        """Forgets everything heard so far, lead-in included."""
        self._reset()
        self._lead.clear()

    def feed(self, frame: bytes) -> bytes | None:
        """Returns the finished utterance (16 kHz s16le) or None."""
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"frame must be {FRAME_BYTES} bytes")
        speech = self._is_speech(frame)
        if not self._frames:
            if not speech:
                self._lead.append(frame)
                return None
            self._frames = list(self._lead)
            self._lead.clear()
        self._frames.append(frame)
        if speech:
            self._speech += 1
            self._quiet = 0
            return None
        self._quiet += 1
        if self._quiet < self._silence_frames:
            return None
        utterance = b"".join(self._frames) if self._speech >= self._min_speech_frames else None
        self._reset()
        return utterance


def _norm(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.lower().replace("ł", "l"))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def wake_match(text: str, words: list[str]) -> str | None:
    """Text after a leading wake word (diacritics, case and punctuation ignored), or None."""
    tokens = [(_norm(m.group()), m.end()) for m in re.finditer(r"\w+", text)]
    heard = [t for t, _ in tokens]
    for wake in sorted((_norm(w).split() for w in words), key=len, reverse=True):
        if wake and heard[: len(wake)] == wake:
            return text[tokens[len(wake) - 1][1] :].lstrip(" ,.!?:;-")
    return None


class LevelMeter:
    """Publishes the peak `level` of every few 30 ms frames (~10 events/s) for the UI meter."""

    EVERY = 3

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._n, self._peak = 0, 0.0

    def __call__(self, frame: bytes) -> float:
        value = level(frame)
        self._peak = max(self._peak, value)
        self._n += 1
        if self._n >= self.EVERY:
            self._bus.publish("level", level=round(self._peak, 3))
            self._n, self._peak = 0, 0.0
        return value


def _mic_stream(on_chunk: Callable[[bytes], None]):
    import sounddevice as sd

    return sd.RawInputStream(
        samplerate=RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_BYTES // 2,
        callback=lambda indata, frames, time, status: on_chunk(bytes(indata)),
    )


class Listener:
    """Push-to-talk capture; `stream_factory(on_chunk)` returns an object with start/stop/close."""

    def __init__(self, config: Config, bus: EventBus, stream_factory: Callable = _mic_stream) -> None:
        self._cfg, self._bus, self._factory = config, bus, stream_factory
        self._chunks: list[bytes] = []
        self._stream = None
        self._meter = LevelMeter(bus)

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start_ptt(self) -> None:
        if self._stream is not None:  # key auto-repeat must not restart the recording
            return
        chunks: list[bytes] = []
        self._chunks = chunks

        def on_chunk(chunk: bytes) -> None:  # PortAudio thread; publish only schedules onto the loop
            chunks.append(chunk)
            self._meter(chunk)

        self._stream = self._factory(on_chunk)
        self._stream.start()

    def stop_ptt(self) -> bytes:
        stream, self._stream = self._stream, None
        if stream is None:
            return b""
        stream.stop()
        stream.close()
        pcm, self._chunks = b"".join(self._chunks), []
        self._bus.publish("level", level=0.0)
        return pcm
