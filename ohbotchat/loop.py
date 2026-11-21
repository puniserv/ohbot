import asyncio
from collections.abc import Callable

from .config import Config
from .events import EventBus
from .listen import FRAME_BYTES, FRAME_MS, LevelMeter, Segmenter, _mic_stream
from .state import State

NO_FRAMES_S = 2.0  # an unplugged USB mic may just go silent instead of finishing the stream
BARGE_GAP_FRAMES = 3  # a child's voice dips between syllables; only this much quiet in a row resets the hold


class MicLoop:
    """Hands-free capture (wake/continuous): one input stream, frames processed on the asyncio loop.

    Only LISTENING frames reach the Segmenter, so the mic is muted while the robot works on a turn.
    While SPEAKING with barge-in enabled, `hold_ms` of loud input calls `on_barge` and that speech
    starts the next utterance.
    """

    def __init__(
        self,
        config: Config,
        bus: EventBus,
        state: Callable[[], State],
        on_utterance: Callable[[bytes], None],
        on_barge: Callable[[], None],
        on_lost: Callable[[], None],
        stream_factory: Callable | None = None,
    ) -> None:
        self._cfg, self._state, self._factory = config, state, stream_factory or _mic_stream
        self._on_utterance, self._on_barge, self._on_lost = on_utterance, on_barge, on_lost
        self._bus = bus
        self._meter = LevelMeter(bus)
        self.segmenter = Segmenter(config.vad.aggressiveness, config.vad.silence_ms, config.vad.min_speech_ms)
        self._stream = None
        self._watchdog: asyncio.TimerHandle | None = None
        self._last_frame = 0.0
        self._buf = bytearray()
        self._loud: list[bytes] = []
        self._quiet = 0

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """Raises what the audio backend raises (no device, PortAudio error)."""
        if self._stream is not None:
            return
        loop = asyncio.get_running_loop()
        # the PortAudio callback must never block, so it only schedules the frame onto the loop
        stream = self._factory(
            lambda chunk: loop.call_soon_threadsafe(self._feed, chunk),
            lambda: loop.call_soon_threadsafe(self._finished, stream),
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._buf.clear()
        self._loud.clear()
        self.segmenter.reset()
        self._stream = stream
        self._last_frame = loop.time()
        self._watchdog = loop.call_later(NO_FRAMES_S, self._watch)

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        self._watchdog.cancel()
        try:
            stream.stop()
            stream.close()
        finally:
            self._bus.publish("level", level=0.0)

    def _finished(self, stream) -> None:
        if stream is self._stream:  # stop() clears _stream first, so only a stream that died on its own gets here
            self._lost()

    def _watch(self) -> None:
        loop = asyncio.get_running_loop()
        silent_until = self._last_frame + NO_FRAMES_S
        if loop.time() >= silent_until:
            self._lost()
        else:
            self._watchdog = loop.call_later(silent_until - loop.time(), self._watch)

    def _lost(self) -> None:
        self.stop()
        self._on_lost()

    def _feed(self, chunk: bytes) -> None:
        if self._stream is None:  # callbacks already queued when stop() ran
            return
        self._last_frame = asyncio.get_running_loop().time()
        self._buf += chunk
        while len(self._buf) >= FRAME_BYTES:
            frame = bytes(self._buf[:FRAME_BYTES])
            del self._buf[:FRAME_BYTES]
            self._frame(frame)

    def _frame(self, frame: bytes) -> None:
        value = self._meter(frame)
        state = self._state()
        if state == State.SPEAKING and self._cfg.barge_in.enabled:
            self._watch_barge(frame, value)
            return
        self._loud.clear()
        if state != State.LISTENING:
            self.segmenter.reset()
            return
        if utterance := self.segmenter.feed(frame):
            self._on_utterance(utterance)

    def _watch_barge(self, frame: bytes, value: float) -> None:
        if value <= self._cfg.barge_in.threshold:
            self._quiet += 1
            if self._quiet >= BARGE_GAP_FRAMES:
                self._loud.clear()
            return
        self._quiet = 0
        self._loud.append(frame)
        if len(self._loud) * FRAME_MS < self._cfg.barge_in.hold_ms:
            return
        frames, self._loud = self._loud, []
        self._on_barge()
        self.segmenter.reset()
        for f in frames:
            self.segmenter.feed(f)
