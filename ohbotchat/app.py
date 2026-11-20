import asyncio

from .audio import LIP_CLOSED, envelope, lip_positions
from .config import Config
from .engine import EngineError
from .events import EventBus
from .fallback import Fallback
from .listen import Listener
from .robot import OhbotDriver, Robot
from .speak import FRAME_MS, Speaker
from .state import State, transition

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
        self._stopping = False
        self._background: list[asyncio.Task] = []

    def _default_engine(self):
        import openai

        from .engine_chain import ChainEngine

        client = openai.AsyncOpenAI(api_key=self.config.api_key)
        return ChainEngine(client, self.config, self.bus)

    async def start(self) -> None:
        await asyncio.to_thread(self.robot.start)
        self._background = [
            asyncio.create_task(self.robot.idle_loop()),
            asyncio.create_task(self.fallback.ensure(self.engine)),
        ]

    async def close(self) -> None:
        for t in self._background:
            t.cancel()
        self.speaker.stop()
        self.listener.stop_ptt()
        await asyncio.gather(*self._background, return_exceptions=True)
        await asyncio.to_thread(self.robot.close)

    def _act(self, action: str) -> None:
        new = transition(self.state, action)
        if new != self.state:
            self.state = new
            self.robot.set_state(new)
            self.bus.publish("state", state=new.value)

    def ptt_start(self) -> None:
        self._act("listen")
        self.listener.start_ptt()

    async def ptt_stop(self) -> None:
        await self.handle_utterance(self.listener.stop_ptt())

    def new_conversation(self) -> None:
        self.history.clear()

    def stop(self) -> None:
        """Interrupts whatever is going on; a speaking robot then lands in LISTENING (see state table)."""
        self.speaker.stop()
        if self.state in (State.TRANSCRIBING, State.THINKING) and self._task:
            self._stopping = True
            self._task.cancel()
        elif self.state == State.LISTENING:
            self.listener.stop_ptt()
            self._act("stop")

    async def handle_utterance(self, pcm: bytes) -> None:
        self._task = asyncio.current_task()
        self._stopping = False
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
            self._act("done" if await self._speak(reply.pcm) else "stop")
        except EngineError as e:
            await self._fail(e)
        except asyncio.CancelledError:
            if not self._stopping:
                raise
            self._task.uncancel()  # our own stop(), not the caller's cancellation
            self._act("stop")
        finally:
            self._task = None

    def _remember(self, heard: str, text: str) -> None:
        self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": text}]
        del self.history[: -self.config.conversation.history_turns * 2]

    async def _fail(self, e: EngineError) -> None:
        self.bus.publish("error", kind=e.kind, message=str(e))
        self._act("error")
        pcm = self.fallback.get(_FALLBACK_KEY.get(e.kind, "error"))
        if pcm:
            await self._speak(pcm)
        self._act("done")

    async def _speak(self, pcm: bytes) -> bool:
        positions = lip_positions(envelope(pcm, 24000, FRAME_MS))

        def on_frame(i: int) -> None:  # called from the speaker thread; robot.lips only enqueues
            if i < len(positions):
                self.robot.lips(*positions[i])

        try:
            return await self.speaker.play(pcm, 24000, on_frame)
        except asyncio.CancelledError:
            self.speaker.stop()
            raise
        finally:
            self.robot.lips(LIP_CLOSED, LIP_CLOSED)
