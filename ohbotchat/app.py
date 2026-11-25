import asyncio
import logging
import math
import random
import time

from .audio import envelope, lip_positions
from .config import Config, save_api_key, validate_api_key, validate_settings
from .engine import EngineError
from .events import EventBus
from .fallback import PHRASES, Fallback
from .listen import Listener, wake_match
from .loop import MicLoop
from .robot import HEADNOD, HEADTURN, OhbotDriver, Robot
from .speak import FRAME_MS, Speaker
from .state import State, transition
from .vision import Vision

log = logging.getLogger(__name__)

PCM_BYTES_PER_S = 16000 * 2
MIN_PCM_BYTES = PCM_BYTES_PER_S * 3 // 10  # under 0.3 s is a click, not speech
FILLER_DELAY_S = 0.7  # a reply ready sooner needs no "Hmm…"
VOICE_SAMPLE = "Cześć! Tak brzmi mój głos."
GENERIC_ERROR = "Coś poszło nie tak – szczegóły w logu"
_FALLBACK_KEY = {"network": "no_network", "moderated": "moderated", "empty": "not_heard"}
TICK_S = 0.1  # eyes follow the face at camera speed (~10 fps); the old 0.25 s made the gaze lag
GREET_AFTER_S = 1.0  # a child walking past must not be greeted
STALE_FACE_S = 1.0  # a hung cap.read() must not leave the robot tracking a ghost


class App:
    """Wires bus, engine, speaker, robot and listener and runs one conversation turn at a time."""

    def __init__(
        self, config: Config, engine=None, speaker=None, robot=None, fallback=None, listener=None, mic_stream=None, vision=None
    ) -> None:
        self.config = config
        self.bus = EventBus()
        self.engine = engine or self._default_engine()
        self.speaker = speaker or Speaker()
        self.robot = robot or Robot(OhbotDriver(), config, self.bus)
        self.fallback = fallback or Fallback(self._fallback_dir())
        self.listener = listener or Listener(config, self.bus)
        self.vision = vision or Vision(config, self.bus)
        self.mic = MicLoop(
            config, self.bus, lambda: self.state, self.start_turn, self._barge_in, self._mic_lost, mic_stream,
            muted=lambda: self.robot.hold_idle,
        )
        self.state = State.IDLE
        self.history: list[dict] = []
        self._task: asyncio.Task | None = None
        self._abort_action: str | None = None
        self._ensure_task: asyncio.Task | None = None
        self._gesture: asyncio.Task | None = None
        self._background: list[asyncio.Task] = []
        self._rec_timer: asyncio.TimerHandle | None = None
        self._wake_until = 0.0
        self._face_since = self._face_last = self._greeted_at = -math.inf
        self._greet_due = False
        self._closed = False
        self.asleep = False
        self._sleep_lock = asyncio.Lock()
        self.mic_error: str | None = None  # a client that connects after the failure still has to see it

    def _fallback_dir(self):
        c = self.config.openai
        return self.config.root / "cache" / "fallback" / f"{c.tts_model}-{c.tts_voice}"  # a new voice must not reuse the old recordings

    def _default_engine(self):
        import openai

        from .engine_chain import ChainEngine

        # retries would stretch the wait before the "no network" phrase to minutes
        client = openai.AsyncOpenAI(api_key=self.config.api_key, max_retries=0)
        return ChainEngine(client, self.config, self.bus)

    async def set_api_key(self, key: str) -> None:
        """Checks the key with OpenAI, writes it to .env and switches to it; ValueError says why a key was refused."""
        key = validate_api_key(key)
        await self._check_key(key)
        await asyncio.to_thread(save_api_key, self.config.root, key)
        self.config.api_key = key
        self.engine = self._default_engine()
        self._ensure_fallback()  # the fillers could not be recorded without a key

    async def _check_key(self, key: str) -> None:
        import openai

        client = openai.AsyncOpenAI(api_key=key, max_retries=0)
        try:
            await client.models.list(timeout=self.config.openai.timeout_s)
        except openai.AuthenticationError:
            raise ValueError("OpenAI odrzuciło ten klucz.") from None
        except Exception as e:  # offline: keep the key, the turn will say what is wrong
            log.warning("API key not verified: %s", type(e).__name__)
        finally:
            await client.close()

    async def start(self) -> None:
        self.bus.bind(asyncio.get_running_loop())
        await asyncio.to_thread(self.robot.start)
        if self._closed:  # Ctrl+C during the connect: close() already ran and must stay the last word
            return
        self._background = [asyncio.create_task(self._tick_loop())]
        self._ensure_fallback()
        if self.asleep:  # put to sleep during the connect; wake starts the camera and mic
            return
        self.vision.start()
        if self.robot.connected:
            self.greet()
        if self.config.conversation.mode != "ptt":
            self._start_mic()

    def calibrate(self) -> str:
        """Camera offsets that make tracking reproduce the head pose the operator set with the test sliders."""
        face, c = self.vision.face, self.config.camera
        if face is None or time.monotonic() - face.t > STALE_FACE_S:
            return "Nie widzę twarzy – stań przed kamerą."
        if not c.head_gain:
            return "Ustaw „Ruch głowy” powyżej 0."
        offsets = {}
        for key, v, offset, motor in (("offset_x", face.x, c.offset_x, HEADTURN), ("offset_y", face.y, c.offset_y, HEADNOD)):
            want = (self.robot.manual_pos.get(motor, 5.0) - 5) / c.head_gain
            offsets[key] = round(min(1.0, max(-1.0, want - (v - offset))), 2)  # face.x/y already include the old offset
        self.apply_settings(validate_settings({"camera": offsets}))
        return f"Skalibrowano: poziomo {offsets['offset_x']}, pionowo {offsets['offset_y']}. Kliknij „Zapisz” w Ustawieniach."

    def apply_settings(self, values: dict[str, dict]) -> None:
        """`values` must come from config.validate_settings; everything else reads the config when used."""
        voice, fillers = self.config.openai.tts_voice, self.config.conversation.filler_phrases
        for section, keys in values.items():
            for key, value in keys.items():
                setattr(getattr(self.config, section), key, value)
        self.mic.segmenter.set_silence_ms(self.config.vad.silence_ms)
        if "colours" in values:
            self.robot.repaint()
        if self.config.openai.tts_voice != voice or self.config.conversation.filler_phrases != fillers:
            if self._ensure_task:  # it would finish the old voice's phrases, or write new ones into the old folder
                self._ensure_task.cancel()
            self.fallback.dir = self._fallback_dir()
            self._ensure_task = asyncio.create_task(self._ensure_phrases())

    async def prepare_fillers(self, phrases: list[str]) -> None:
        """Moderates and caches new filler phrases for the current voice; raises EngineError naming the failed one."""
        for text in phrases:
            if self.fallback.cached(text) is not None:  # cached means it already passed moderation
                continue
            try:
                self.fallback.store(text, await self.engine.say(text, publish=False))
            except EngineError as e:
                raise EngineError(e.kind, text) from e

    def ready(self) -> bool:
        """Manual test commands (motors, test sentence) run only while the robot does nothing else.

        Hands-free modes rest in LISTENING, so that counts too unless a child is holding the button.
        """
        return self.state in (State.IDLE, State.LISTENING) and not self._busy() and not self.listener.recording

    def manual_gesture(self, name: str) -> None:
        if self._gesture:
            self._gesture.cancel()  # its finally queues the neutral pose before the new gesture starts
        self.robot.pause_idle()
        self._gesture = asyncio.create_task(self.robot.gesture(name))

    async def set_asleep(self, asleep: bool) -> None:
        """Sleep frees the serial port, camera and microphone and keeps every loop off the robot; wake undoes it."""
        async with self._sleep_lock:  # a double click must not interleave a sleep with a wake
            if asleep == self.asleep:
                return
            try:
                if asleep:
                    self.asleep = True
                    self.mic.stop()
                    self.ptt_abort()
                    self.stop()
                    self._abort_action = "cancel"  # the turn must not end in LISTENING with the mic off
                    await asyncio.gather(*filter(None, [self._task, self._gesture]), return_exceptions=True)
                    if self.state == State.LISTENING:
                        self._act("cancel")
                    await asyncio.to_thread(self.vision.stop)
                    await asyncio.to_thread(self.robot.sleep)
                else:
                    await asyncio.to_thread(self.robot.start)
                    if self._closed:
                        return
                    self.vision.start()
                    self.asleep = False  # only now, or the tick loop would retry the connect start() is doing
                    if self.config.conversation.mode != "ptt":
                        self._start_mic()
            except Exception:
                log.exception("sleep/wake failed")  # runs as a fire-and-forget task, nobody else would log it
            finally:
                self.bus.publish("sleep", asleep=self.asleep)  # else the UI button stays on "Usypiam…"/"Budzę…"

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        if self._closed:  # called from uvicorn shutdown and again from the caller's finally
            return
        self._closed = True
        self._cancel_rec_timer()
        tasks = [*self._background, *filter(None, [self._task, self._ensure_task, self._gesture])]
        for t in tasks:
            t.cancel()
        self.speaker.stop()
        self.listener.stop_ptt()
        self.mic.stop()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(self.vision.stop)
        await asyncio.to_thread(self.robot.close)

    async def _tick_loop(self) -> None:
        while True:
            now, face = time.monotonic(), self.vision.face
            if face is not None and now - face.t > STALE_FACE_S:
                face = None
            if not self.asleep:
                try:
                    self.robot.tick(now, face)
                    self._auto_greet(now, face)
                except Exception:  # else one bad tick ends tracking, blinking and the reconnect until restart
                    log.exception("tick failed")
            await asyncio.sleep(TICK_S)

    def _auto_greet(self, now: float, face) -> None:
        """Greets a child who walked up: a face for GREET_AFTER_S after nobody was there for greet_cooldown_s."""
        c = self.config.camera
        if face is None:
            return
        if now - self._face_last >= c.lost_after_s:  # a new arrival, not a missed detection
            self._face_since = now
            self._greet_due = now - self._face_last >= c.greet_cooldown_s
        self._face_last = now
        if not self._greet_due or now - self._face_since < GREET_AFTER_S:
            return
        self._greet_due = False  # one chance per arrival: a busy robot is already talking with them
        if not (c.auto_greet and self.ready()) or now - self._greeted_at < c.greet_cooldown_s:
            return
        self.new_conversation()  # whoever talked before left at least greet_cooldown_s ago
        task = self.greet()
        if self.config.conversation.mode == "wake":  # the child may answer without saying "Ohbot"
            task.add_done_callback(lambda t: t.cancelled() or self._open_wake_window())

    def _ensure_fallback(self) -> None:
        if self._ensure_task is None or self._ensure_task.done():
            self._ensure_task = asyncio.create_task(self._ensure_phrases())

    async def _ensure_phrases(self) -> None:
        try:
            await self.fallback.ensure(self.engine, self.config.conversation.filler_phrases)
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
            if not self.asleep:
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
        self._wake_until = asyncio.get_running_loop().time() + self.config.wake.window_s

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
        self.ptt_abort()  # a greeting must not play over a held button whose recording would then be dropped
        if self._busy():  # the old exchange would land in the fresh history once its reply arrives
            self.stop()
        self.history.clear()

    def stop(self, relisten: bool = False) -> None:
        """Interrupts the turn. Ends in LISTENING if the mic is wanted next (barge-in, button), else IDLE."""
        self._abort_action = "stop" if relisten or self.config.conversation.mode != "ptt" else "cancel"
        self._wake_until = 0.0
        self.speaker.stop()
        if self._gesture:
            self._gesture.cancel()
        if self._busy():
            self._task.cancel()
        elif self.state == State.LISTENING and not self.mic.running:
            self._cancel_rec_timer()
            self.listener.stop_ptt()
            self._act(self._abort_action)

    def greet(self) -> asyncio.Task:
        self._greeted_at = time.monotonic()  # startup and "Nowa rozmowa" greetings count for the auto greet cooldown
        return self.say(self.config.conversation.greeting, cache=True)

    def voice_sample(self, voice: str) -> asyncio.Task:
        return self.say(VOICE_SAMPLE, voice=voice)

    def say(self, text: str, voice: str | None = None, cache: bool = False) -> asyncio.Task:
        """Speaks text that is not a reply (greeting, test sentence) after any turn being stopped; never in history."""
        prev = self._task if self._busy() else None
        self._task = asyncio.create_task(self._guarded(lambda: self._say(text, voice, cache, prev), self.config.conversation.mode))
        return self._task

    async def _say(self, text: str, voice: str | None, cache: bool, prev: asyncio.Task | None) -> None:
        if prev:
            await asyncio.gather(prev, return_exceptions=True)
        self._abort_action = None
        self._act("say")
        pcm = self.fallback.cached(text) if cache else None
        if pcm is None:
            pcm = await self.engine.say(text, voice)
            if cache:
                self.fallback.store(text, pcm)
        await self._reply(text, pcm)

    async def handle_utterance(self, pcm: bytes) -> None:
        mode = self.config.conversation.mode  # a switch from the UI applies from the next utterance
        await self._guarded(lambda: self._utterance(pcm, mode), mode)

    async def _utterance(self, pcm: bytes, mode: str) -> None:
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

    async def _guarded(self, work, mode: str) -> None:
        """Runs one turn: errors become the spoken fallback, a stop ends in the state `stop()` chose."""
        try:
            await work()
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
        pending = asyncio.create_task(self.engine.respond(heard, self.history, show))
        # a reply that fails while a stopped filler plays is never awaited; retrieve it so asyncio does not log it
        pending.add_done_callback(lambda t: t.cancelled() or t.exception())
        try:
            if self.config.conversation.fillers:
                done, _ = await asyncio.wait({pending}, timeout=FILLER_DELAY_S)
                if not done and (filler := self._filler()):
                    await self._speak(filler)
            reply = await pending
        except EngineError as e:
            if e.kind == "moderated" and not shown:
                self.bus.publish("heard", text="(ukryto)")
            raise
        finally:
            pending.cancel()  # a stop during the filler must drop the reply too
        self._remember(heard, reply.text)
        await self._reply(reply.text, reply.pcm, reply.gesture)
        if not self.fallback.complete(self.config.conversation.filler_phrases):
            self._ensure_fallback()

    def _filler(self) -> bytes | None:
        cached = [pcm for text in self.config.conversation.filler_phrases if (pcm := self.fallback.cached(text))]
        return random.choice(cached) if cached else None

    async def _reply(self, text: str, pcm: bytes, gesture: str | None = None) -> None:
        self.bus.publish("reply", text=text)
        self._act("reply")
        if await self._speak(pcm, gesture):
            self._act("done")
        else:
            self._act(self._abort_action or "cancel")

    def _remember(self, heard: str, text: str) -> None:
        n = self.config.conversation.history_turns
        self.history = [*self.history, {"role": "user", "content": heard}, {"role": "assistant", "content": text}]
        self.history = self.history[-2 * n :] if n else []

    async def _fail(self, e: EngineError) -> None:
        log.error("turn failed (%s): %s", e.kind, e)
        # SDK error text is English, technical and may quote the request; it belongs in the log, not the UI
        self.bus.publish("error", kind=e.kind, message=GENERIC_ERROR if e.kind == "other" else str(e))
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
        r = self.config.robot
        positions = lip_positions(envelope(pcm, 24000, FRAME_MS), r.lip_closed, r.lip_open)
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

        await self._end_gesture()
        if gesture:
            self._gesture = asyncio.create_task(self.robot.gesture(gesture))
        played = False
        try:
            played = await self.speaker.play(pcm, 24000, on_frame)
            return played
        except asyncio.CancelledError:
            self.speaker.stop()
            raise
        finally:
            # a finished reply leaves its gesture running so the turn, and the muted mic, end with the audio
            if not played:
                await self._end_gesture()
            lips(r.lip_closed, r.lip_closed)

    async def _end_gesture(self) -> None:
        if self._gesture:
            self._gesture.cancel()
            await asyncio.gather(self._gesture, return_exceptions=True)  # its finally queues the neutral pose
            self._gesture = None
