import asyncio
import logging

from .audio import LIP_CLOSED, envelope, lip_positions
from .config import Config
from .engine import EngineError
from .events import EventBus
from .fallback import PHRASES, Fallback
from .listen import Listener, wake_match
from .loop import MicLoop
from .robot import OhbotDriver, Robot
from .speak import FRAME_MS, Speaker
from .state import State, transition

log = logging.getLogger(__name__)

PCM_BYTES_PER_S = 16000 * 2
MIN_PCM_BYTES = PCM_BYTES_PER_S * 3 // 10  # under 0.3 s is a click, not speech
WAKE_WINDOW_S = 8  # after a bare "Ohbot!" the next utterance needs no wake word
_FALLBACK_KEY = {"network": "no_network", "moderated": "moderated", "empty": "not_heard"}


class App:
    """Wires bus, engine, speaker, robot and listener and runs one conversation turn at a time."""

    def __init__(
        self, config: Config, engine=None, speaker=None, robot=None, fallback=None, listener=None, mic_stream=None
    ) -> None:
        self.config = config
        self.bus = EventBus()
        self.engine = engine or self._default_engine()
        self.speaker = speaker or Speaker()
        self.robot = robot or Robot(OhbotDriver(), config, self.bus)
        self.fallback = fallback or Fallback(config.root / "cache" / "fallback")
        self.listener = listener or Listener(config, self.bus)
        self.mic = MicLoop(config, self.bus, lambda: self.state, self.start_turn, self._barge_in, self._mic_lost, mic_stream)
        self.state = State.IDLE
        self.history: list[dict] = []
        self._task: asyncio.Task | None = None
        self._abort_action: str | None = None
        self._ensure_task: asyncio.Task | None = None
        self._background: list[asyncio.Task] = []
        self._rec_timer: asyncio.TimerHandle | None = None
        self._wake_until = 0.0
        self._closed = False
        self.mic_error: str | None = None  # a client that connects after the failure still has to see it

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
        if self.config.conversation.mode != "ptt":
            self._start_mic()

    async def close(self) -> None:
        if self._closed:  # called from uvicorn shutdown and again from the caller's finally
            return
        self._closed = True
        self._cancel_rec_timer()
        tasks = [*self._background, *filter(None, [self._task, self._ensure_task])]
        for t in tasks:
            t.cancel()
        self.speaker.stop()
        self.listener.stop_ptt()
        self.mic.stop()
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
        if self.mic.running:  # hands-free: the loop already listens, the button only interrupts
            return
        self._act("listen")
        try:
            self.listener.start_ptt()
        except Exception as e:
            log.exception("microphone unavailable")
            self.bus.publish("error", kind="other", message=f"Nie mogę włączyć mikrofonu: {e}")
            self._act("cancel")
            return
        self.mic_error = None
        self._cancel_rec_timer()
        self._rec_timer = asyncio.get_running_loop().call_later(self.config.conversation.max_record_s, self.ptt_stop)

    def _cancel_rec_timer(self) -> None:
        if self._rec_timer:
            self._rec_timer.cancel()
            self._rec_timer = None

    def set_mode(self, mode: str) -> None:
        if mode == self.config.conversation.mode:
            return
        self.config.conversation.mode = mode
        self.ptt_abort()
        if mode != "ptt":
            self._start_mic()
            return
        self.mic.stop()
        self._wake_until = 0.0
        if self.state == State.LISTENING:
            self._act("cancel")

    def _start_mic(self) -> None:
        try:
            self.mic.start()
        except Exception as e:
            log.exception("microphone loop unavailable")
            self._fall_back_to_ptt(f"Nie mogę włączyć mikrofonu, wracam do trybu z przyciskiem: {e}")
            return
        self.mic_error = None
        if self.state == State.IDLE:
            self._act("listen")

    def _mic_lost(self) -> None:
        log.error("microphone stream lost")
        self._fall_back_to_ptt("Mikrofon odłączony – wracam do trybu z przyciskiem")

    def _fall_back_to_ptt(self, message: str) -> None:
        self.mic_error = message
        self.bus.publish("error", kind="other", message=message)
        self.config.conversation.mode = "ptt"
        self._wake_until = 0.0
        self.bus.publish("settings", mode="ptt")
        if self.state == State.LISTENING:
            self._act("cancel")

    def _barge_in(self) -> None:
        self.stop(relisten=True)
        if self.config.conversation.mode == "wake":  # talking over the robot is meant for it
            self._open_wake_window()

    def _open_wake_window(self) -> None:
        self._wake_until = asyncio.get_running_loop().time() + WAKE_WINDOW_S

    def ptt_abort(self) -> None:
        """Drops a recording without a turn (client vanished while the button was held)."""
        self._cancel_rec_timer()
        if self.listener.recording:
            self.listener.stop_ptt()
            self._act("cancel")

    def ptt_stop(self) -> asyncio.Task | None:
        """Ends the recording and starts the turn in its own task; None if nothing was recording."""
        self._cancel_rec_timer()
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
        self._wake_until = 0.0
        self.speaker.stop()
        if self._busy():
            self._task.cancel()
        elif self.state == State.LISTENING and not self.mic.running:
            self._cancel_rec_timer()
            self.listener.stop_ptt()
            self._act(self._abort_action)

    async def handle_utterance(self, pcm: bytes) -> None:
        mode = self.config.conversation.mode  # a switch from the UI applies from the next utterance
        try:
            if self.state == State.IDLE:
                self._act("listen")
            self._act("utterance")
            if len(pcm) < MIN_PCM_BYTES:
                raise EngineError("empty")
            # STT latency must not eat into the wake window
            spoke_at = asyncio.get_running_loop().time() - len(pcm) / PCM_BYTES_PER_S
            heard = await self.engine.transcribe(pcm)
            if mode == "wake":
                heard = await self._wake(heard, spoke_at)
                if heard is None:
                    return
            await self._respond(heard)
        except EngineError as e:
            if e.kind == "empty" and mode != "ptt":  # room noise must not make the robot say "nie usłyszałem"
                self._ignore("(nic nie rozpoznano)")
            else:
                await self._fail(e)
        except asyncio.CancelledError:
            self._act(self._abort_action or "cancel")
            raise
        except Exception as e:
            log.exception("turn failed")
            await self._fail(EngineError("other", str(e)))
        finally:
            if self.mic.running and self.state == State.IDLE:
                self._act("listen")

    async def _wake(self, text: str, spoke_at: float) -> str | None:
        """What to answer in wake mode, or None when the utterance was not for the robot."""
        if spoke_at < self._wake_until:
            self._wake_until = 0.0
            return text
        rest = wake_match(text, self.config.wake.words)
        if rest is None:
            self._ignore("zignorowano (brak słowa „Ohbot”)")
            return None
        if rest:
            return rest
        self.bus.publish("heard", text=text)
        self.bus.publish("reply", text=PHRASES["listening"])
        self._act("heard")
        self._act("reply")
        self._open_wake_window()  # a barge-in cuts "Słucham?" short, so the window cannot wait for its end
        if pcm := self.fallback.get("listening"):
            await self._speak(pcm)
        self._act("done")
        return None

    def _ignore(self, reason: str) -> None:
        # never the transcript: it has not been through moderation
        self.bus.publish("info", text=reason)
        self._act("ignore" if self.mic.running else "cancel")  # the mic may have been lost during STT

    async def _respond(self, heard: str) -> None:
        """The rest of a turn once the child's words are known as text."""
        shown = False

        def show() -> None:
            nonlocal shown
            shown = True
            self.bus.publish("heard", text=heard)

        self._act("heard")
        try:
            reply = await self.engine.respond(heard, self.history, show)
        except EngineError as e:
            if e.kind == "moderated" and not shown:
                self.bus.publish("heard", text="(ukryto)")
            raise
        self._remember(heard, reply.text)
        self.bus.publish("reply", text=reply.text)
        self._act("reply")
        if await self._speak(reply.pcm, reply.gesture):
            self._act("done")
        else:
            self._act(self._abort_action or "cancel")
        if not self.fallback.complete():
            self._ensure_fallback()

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

    async def _speak(self, pcm: bytes, gesture: str | None = None) -> bool:
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

        moving = asyncio.create_task(self.robot.gesture(gesture)) if gesture else None
        try:
            played = await self.speaker.play(pcm, 24000, on_frame)
            if played and moving:  # a short "Tak!" should still get its whole nod
                await moving
            return played
        except asyncio.CancelledError:
            self.speaker.stop()
            raise
        finally:
            if moving:
                moving.cancel()
                await asyncio.gather(moving, return_exceptions=True)
            lips(LIP_CLOSED, LIP_CLOSED)
