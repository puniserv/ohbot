import re
import unicodedata
from collections import deque
from collections.abc import Callable

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

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start_ptt(self) -> None:
        if self._stream is not None:  # key auto-repeat must not restart the recording
            return
        self._chunks = []
        self._stream = self._factory(self._chunks.append)
        self._stream.start()

    def stop_ptt(self) -> bytes:
        stream, self._stream = self._stream, None
        if stream is None:
            return b""
        stream.stop()
        stream.close()
        pcm, self._chunks = b"".join(self._chunks), []
        return pcm
