import asyncio
import logging

from .audio import LIP_CLOSED, envelope, lip_positions
from .config import Config
from .engine import EngineError
from .events import EventBus
from .fallback import Fallback
from .listen import Listener
from .robot import OhbotDriver, Robot
from .speak import FRAME_MS, Speaker
from .state import State, transition

log = logging.getLogger(__name__)

MIN_PCM_BYTES = 16000 * 2 * 3 // 10  # under 0.3 s is a click, not speech
_FALLBACK_KEY = {"network": "no_network", "moderated": "moderated", "empty": "not_heard"}


class App:
    """Wires bus, engine, speaker, robot and listener and runs one conversation turn at a time."""

    def __init__(self, config: Config, engine=None, speaker=None, robot=None, fallback=None, listener=None) -> None:
        self.config = config
        self.bus = EventBus()
        self.engine = engine or self._default_engine()
        self.speaker = speaker or Speaker()
        self.robot = robot or Robot(OhbotDriver(), config, self.bus)
        self.fallback = fallback or Fallback(config.root / "cache" / "fallback")
        self.listener = listener or Listener(config, self.bus)
        self.state = State.IDLE
        self.history: list[dict] = []
        self._task: asyncio.Task | None = None
        self._abort_action: str | None = None
        self._ensure_task: asyncio.Task | None = None
        self._background: list[asyncio.Task] = []

    def _default_engine(self):
        import openai

        from .engine_chain import ChainEngine

        # retries would stretch the wait before the "no network" phrase to minutes
        client = openai.AsyncOpenAI(api_key=self.config.api_key, max_retries=0)
        return ChainEngine(client, self.config, self.bus)

    async def start(self) -> None:
        self.bus.bind(asyncio.get_running_loop())
        await asyncio.to_thread(self.robot.start)
        self._background = [asyncio.create_task(self.robot.idle_loop())]
        self._ensure_fallback()

    async def close(self) -> None:
        tasks = [*self._background, *filter(None, [self._task, self._ensure_task])]
        for t in tasks:
            t.cancel()
        self.speaker.stop()
        self.listener.stop_ptt()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(self.robot.close)

    def _ensure_fallback(self) -> None:
        if self._ensure_task is None or self._ensure_task.done():
            self._ensure_task = asyncio.create_task(self._ensure_phrases())

    async def _ensure_phrases(self) -> None:
        try:
            await self.fallback.ensure(self.engine)
        except Exception:
            log.exception("fallback phrase generation failed")

    def _act(self, action: str) -> None:
        new = transition(self.state, action)
        if new != self.state:
            self.state = new
            self.robot.set_state(new)
            self.bus.publish("state", state=new.value)

    def _busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def ptt_start(self) -> None:
        """Pressing the button while the robot talks or thinks interrupts it, then starts listening."""
        if self.state in (State.TRANSCRIBING, State.THINKING, State.SPEAKING, State.ERROR):
            self.stop(relisten=True)
        self._act("listen")
        try:
            self.listener.start_ptt()
        except Exception as e:
            log.exception("microphone unavailable")
            self.bus.publish("error", kind="other", message=f"Nie mogę włączyć mikrofonu: {e}")
            self._act("cancel")

    def ptt_stop(self) -> asyncio.Task | None:
        """Ends the recording and starts the turn in its own task; None if nothing was recording."""
        if not self.listener.recording:
            return None
        return self.start_turn(self.listener.stop_ptt())

    def start_turn(self, pcm: bytes) -> asyncio.Task:
        if not self._busy():
            self._abort_action = None
            self._task = asyncio.create_task(self.handle_utterance(pcm))
        return self._task

    def new_conversation(self) -> None:
        self.history.clear()

    def stop(self, relisten: bool = False) -> None:
        """Interrupts the turn. Ends in LISTENING if the mic is wanted next (barge-in, button), else IDLE."""
        self._abort_action = "stop" if relisten or self.config.conversation.mode != "ptt" else "cancel"
        self.speaker.stop()
        if self._busy():
            self._task.cancel()
        elif self.state == State.LISTENING:
            self.listener.stop_ptt()
            self._act(self._abort_action)

    async def handle_utterance(self, pcm: bytes) -> None:
        try:
            if self.state == State.IDLE:
                self._act("listen")
            self._act("utterance")
            if len(pcm) < MIN_PCM_BYTES:
                raise EngineError("empty")
            heard = await self.engine.transcribe(pcm)
            self.bus.publish("heard", text=heard)
            self._act("heard")
            reply = await self.engine.respond(heard, self.history)
            self._remember(heard, reply.text)
            self.bus.publish("reply", text=reply.text)
            self._act("reply")
            if await self._speak(reply.pcm):
                self._act("done")
            else:
                self._act(self._abort_action or "cancel")
            if not self.fallback.complete():
                self._ensure_fallback()
        except EngineError as e:
            await self._fail(e)
        except asyncio.CancelledError:
            self._act(self._abort_action or "cancel")
            raise
        except Exception as e:
            log.exception("turn failed")
            await self._fail(EngineError("other", str(e)))

    def _remember(self, heard: str, text: str) -> None:
        n = self.config.conversation.history_turns
        self.history = [*self.history, {"role": "user", "content": heard}, {"role": "assistant", "content": text}]
        self.history = self.history[-2 * n :] if n else []

    async def _fail(self, e: EngineError) -> None:
        self.bus.publish("error", kind=e.kind, message=str(e))
        self._act("error")
        try:
            pcm = self.fallback.get(_FALLBACK_KEY.get(e.kind, "error"))
            if pcm:
                await self._speak(pcm)
        except asyncio.CancelledError:  # handlers of handle_utterance's own try do not cover this one
            self._act(self._abort_action or "cancel")
            raise
        except Exception:
            log.exception("could not play the fallback phrase")
        self._act("done")

    async def _speak(self, pcm: bytes) -> bool:
        positions = lip_positions(envelope(pcm, 24000, FRAME_MS))
        loop = asyncio.get_running_loop()
        delay = self.config.robot.lip_delay_ms / 1000

        def lips(top: float, bottom: float) -> None:  # robot.lips only enqueues, so any thread may call it
            if delay:
                loop.call_soon_threadsafe(loop.call_later, delay, self.robot.lips, top, bottom)
            else:
                self.robot.lips(top, bottom)

        def on_frame(i: int) -> None:  # called from the speaker thread
            if i < len(positions):
                lips(*positions[i])

        try:
            return await self.speaker.play(pcm, 24000, on_frame)
        except asyncio.CancelledError:
            self.speaker.stop()
            raise
        finally:
            lips(LIP_CLOSED, LIP_CLOSED)
