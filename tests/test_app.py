import asyncio
import time

import pytest

from ohbotchat.app import App
from ohbotchat.config import load_config
from ohbotchat.engine import EngineError, Reply
from ohbotchat.events import EventBus
from ohbotchat.listen import Segmenter
from ohbotchat.robot import BOTTOMLIP, EYETURN, HEADNOD, HEADTURN, TOPLIP, NullDriver, Robot
from ohbotchat.vision import Face

SPEECH_PCM = b"\x00\x20" * 24000  # 1 s of loud-ish signal at 24 kHz
LONG_ENOUGH = b"\x00\x00" * 16000


class FakeEngine:
    def __init__(self, transcript="Cześć", fail=None, fail_stage="respond"):
        self.transcript, self.fail, self.fail_stage = transcript, fail, fail_stage
        self.history_seen = []

    async def transcribe(self, pcm):
        if self.fail and self.fail_stage == "transcribe":
            raise self.fail
        return self.transcript

    async def respond(self, heard, history, on_safe=None):
        self.history_seen.append(list(history))
        if self.fail and self.fail_stage == "respond":  # input moderation or anything before it
            raise self.fail
        if on_safe:
            on_safe()
        if self.fail and self.fail_stage == "reply":
            raise self.fail
        return Reply(heard=heard, text="Hej!", pcm=SPEECH_PCM)

    async def synthesize(self, text):
        return b"\x00\x00"

    async def say(self, text, voice=None):
        self.said = [*getattr(self, "said", []), (text, voice)]
        if self.fail and self.fail_stage == "say":
            raise self.fail
        return b"SAY-" + text.encode()[:4]


class FakeSpeaker:
    def __init__(self, block=False, fail=None):
        self.played, self.block, self.fail = [], block, fail
        self.started = asyncio.Event()
        self._stop = asyncio.Event()

    async def play(self, pcm, rate=24000, on_frame=None):
        self.played.append(pcm)
        if self.fail:
            raise self.fail
        if on_frame:
            for i in range(3):
                on_frame(i)
        if self.block:
            self.started.set()
            await self._stop.wait()
            return False
        return True

    def stop(self):
        self._stop.set()


class FakeFallback:
    def __init__(self, complete=True):
        self.asked, self.ensured, self._complete = [], 0, complete
        self.dir, self.texts = None, {}

    def complete(self):
        return self._complete

    def get(self, key):
        self.asked.append(key)
        pcm = b"FB-" + key.encode()
        return pcm + b"\0" * (len(pcm) % 2)

    async def ensure(self, engine):
        self.ensured += 1

    def cached(self, text):
        return self.texts.get(text)

    def store(self, text, pcm):
        self.texts[text] = pcm


class FakeListener:
    def __init__(self, fail=None):
        self.recording, self.starts, self.fail = False, 0, fail

    def start_ptt(self):
        if self.fail:
            raise self.fail
        self.starts += 1
        self.recording = True

    def stop_ptt(self):
        self.recording = False
        return LONG_ENOUGH


class FakeVision:
    def __init__(self):
        self.face, self.ok, self.running = None, None, False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


def make(engine=None, speaker=None, turns=None, mode=None, delay=None, fallback=None):
    cfg = load_config(env={})
    if turns is not None:
        cfg.conversation.history_turns = turns
    if mode:
        cfg.conversation.mode = mode
    if delay:
        cfg.robot.lip_delay_ms = delay
    driver = NullDriver()
    app = App(
        cfg,
        engine=engine or FakeEngine(),
        speaker=speaker or FakeSpeaker(),
        fallback=fallback or FakeFallback(),
        listener=FakeListener(),
        vision=FakeVision(),
        mic_stream=MicStreams(),
    )
    app.robot = Robot(driver, cfg, app.bus)
    app.robot.start()
    return app, driver


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def states(events):
    return [e.data["state"] for e in events if e.type == "state"]


async def test_full_cycle_states_and_events():
    app, driver = make()
    q = app.bus.subscribe()
    await app.handle_utterance(LONG_ENOUGH)
    events = drain(q)
    assert states(events) == ["listening", "transcribing", "thinking", "speaking", "idle"]
    assert [e.data["text"] for e in events if e.type in ("heard", "reply")] == ["Cześć", "Hej!"]
    app.robot.flush()
    assert any(c[:2] == ("move", TOPLIP) for c in driver.calls)
    assert [c for c in driver.calls if c[0] == "move"][-2:] == [("move", TOPLIP, 1.5, 10), ("move", BOTTOMLIP, 1.5, 10)]  # mouth closed at the end


@pytest.mark.parametrize(
    "kind,stage,phrase",
    [
        ("network", "respond", "no_network"),
        ("moderated", "respond", "moderated"),
        ("empty", "transcribe", "not_heard"),
        ("quota", "respond", "error"),
    ],
)
async def test_engine_error_plays_fallback_and_publishes(kind, stage, phrase):
    speaker = FakeSpeaker()
    app, _ = make(FakeEngine(fail=EngineError(kind), fail_stage=stage), speaker)
    q = app.bus.subscribe()
    await app.handle_utterance(LONG_ENOUGH)
    events = drain(q)
    assert [p.rstrip(b"\0") for p in speaker.played] == [b"FB-" + phrase.encode()]
    assert [e.data["kind"] for e in events if e.type == "error"] == [kind]
    assert states(events)[-2:] == ["error", "idle"]
    assert app.history == []


async def test_too_short_recording_is_not_sent_to_engine():
    engine = FakeEngine()
    app, _ = make(engine)
    await app.handle_utterance(b"\x00\x00" * 100)
    assert engine.history_seen == []
    assert app.fallback.asked == ["not_heard"]


async def test_stop_during_speaking_goes_idle_in_ptt_mode():
    speaker = FakeSpeaker(block=True)
    app, _ = make(speaker=speaker)
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    app.stop()
    await asyncio.gather(task, return_exceptions=True)
    assert app.state == "idle"


async def test_stop_during_speaking_goes_listening_outside_ptt_mode():
    speaker = FakeSpeaker(block=True)
    app, _ = make(speaker=speaker, mode="continuous")
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    app.stop()
    await asyncio.gather(task, return_exceptions=True)
    assert app.state == "listening"


async def test_stop_during_thinking_cancels_request():
    gate = asyncio.Event()

    class Slow(FakeEngine):
        async def respond(self, heard, history, on_safe=None):
            await gate.wait()

    app, _ = make(Slow())
    task = app.start_turn(LONG_ENOUGH)
    while app.state != "thinking":
        await asyncio.sleep(0)
    app.stop()
    await asyncio.gather(task, return_exceptions=True)
    assert app.state == "idle"


async def test_ptt_start_interrupts_running_turn_then_listens():
    speaker = FakeSpeaker(block=True)
    app, _ = make(speaker=speaker)
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    app.ptt_start()
    await asyncio.gather(task, return_exceptions=True)
    assert app.state == "listening" and app.listener.recording


async def test_ptt_start_twice_is_harmless():
    app, _ = make()
    app.ptt_start()
    app.ptt_start()
    assert app.state == "listening" and app.listener.recording


async def test_ptt_stop_returns_turn_task_and_ignores_when_not_recording():
    app, _ = make()
    assert app.ptt_stop() is None
    app.ptt_start()
    task = app.ptt_stop()
    assert task is not None and app.ptt_stop() is None
    await task
    assert app.state == "idle"


async def test_start_turn_ignored_while_a_turn_runs():
    speaker = FakeSpeaker(block=True)
    app, _ = make(speaker=speaker)
    first = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    assert app.start_turn(LONG_ENOUGH) is first
    app.stop()
    await asyncio.gather(first, return_exceptions=True)


@pytest.mark.parametrize("where", ["speak", "respond"])
async def test_unexpected_exception_ends_in_idle_with_error_event(where):
    if where == "speak":
        app, _ = make(speaker=FakeSpeaker(fail=OSError("no output device")))
    else:

        class Broken(FakeEngine):
            async def respond(self, heard, history, on_safe=None):
                raise FileNotFoundError("persona")

        app, _ = make(Broken())
    q = app.bus.subscribe()
    await app.handle_utterance(LONG_ENOUGH)
    assert app.state == "idle"
    assert [e.data["kind"] for e in drain(q) if e.type == "error"] == ["other"]


async def test_other_error_shows_a_generic_message_and_logs_the_raw_one(caplog):
    app, _ = make(FakeEngine(fail=EngineError("other", "Error code: 400 - {'message': 'sdk detail'}")))
    q = app.bus.subscribe()
    await app.handle_utterance(LONG_ENOUGH)
    [error] = [e for e in drain(q) if e.type == "error"]
    assert error.data["message"] == "Coś poszło nie tak – szczegóły w logu"
    assert "sdk detail" in caplog.text


async def test_lip_delay_defers_mouth_moves_including_closing():
    app, driver = make(delay=200)
    await app.handle_utterance(LONG_ENOUGH)
    app.robot.flush()
    assert not [c for c in driver.calls if c[0] == "move"]
    await asyncio.sleep(0.4)
    app.robot.flush()
    moves = [c for c in driver.calls if c[0] == "move"]
    assert moves and moves[-2:] == [("move", TOPLIP, 1.5, 10), ("move", BOTTOMLIP, 1.5, 10)]


async def test_missing_fallback_phrases_are_retried_after_a_good_turn():
    fb = FakeFallback(complete=False)
    app, _ = make(fallback=fb)
    await app.handle_utterance(LONG_ENOUGH)
    await app._ensure_task
    assert fb.ensured == 1
    fb2 = FakeFallback(complete=True)
    app, _ = make(fallback=fb2)
    await app.handle_utterance(LONG_ENOUGH)
    assert app._ensure_task is None and fb2.ensured == 0


def test_default_engine_does_not_retry_in_sdk():
    cfg = load_config(env={})
    cfg.api_key = "sk-test"
    app = App(cfg, speaker=FakeSpeaker(), fallback=FakeFallback(), listener=FakeListener(), robot=object())
    assert app.engine._client.max_retries == 0


def test_fallback_phrases_are_cached_per_tts_model_and_voice():
    cfg = load_config(env={})
    cfg.openai.tts_model, cfg.openai.tts_voice = "tts-x", "ash"
    app = App(cfg, engine=FakeEngine(), speaker=FakeSpeaker(), listener=FakeListener(), robot=object())
    assert app.fallback.dir == cfg.root / "cache" / "fallback" / "tts-x-ash"


async def test_zero_history_turns_keeps_nothing():
    app, _ = make(turns=0)
    await app.handle_utterance(LONG_ENOUGH)
    assert app.history == []


async def test_history_trimmed_and_cleared():
    engine = FakeEngine()
    app, _ = make(engine, turns=2)
    for _ in range(4):
        await app.handle_utterance(LONG_ENOUGH)
    assert len(app.history) == 4
    assert len(engine.history_seen[-1]) == 4
    assert app.history[0] == {"role": "user", "content": "Cześć"}
    app.new_conversation()
    assert app.history == []


async def test_new_conversation_drops_the_turn_in_flight():
    gate = asyncio.Event()

    class Slow(FakeEngine):
        async def respond(self, heard, history, on_safe=None):
            await gate.wait()
            return await super().respond(heard, history, on_safe)

    app, _ = make(Slow())
    task = app.start_turn(LONG_ENOUGH)
    await asyncio.sleep(0.01)
    app.new_conversation()
    gate.set()
    await asyncio.gather(task, return_exceptions=True)
    assert app.history == [] and app.state == "idle"


async def test_stop_while_fallback_plays_does_not_stick_in_error():
    speaker = FakeSpeaker(block=True)
    app, _ = make(FakeEngine(fail=EngineError("network")), speaker)
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    assert app.state == "error"
    app.stop()
    await asyncio.gather(task, return_exceptions=True)
    assert app.state == "idle"


async def test_ptt_start_without_mic_publishes_error_and_stays_idle():
    app, _ = make()
    app.listener = FakeListener(fail=OSError("no input device"))
    q = app.bus.subscribe()
    app.ptt_start()
    events = drain(q)
    assert [e.data["kind"] for e in events if e.type == "error"] == ["other"]
    assert app.state == "idle"


async def test_ptt_abort_discards_audio_without_a_turn():
    app, _ = make()
    app.ptt_start()
    app.ptt_abort()
    assert not app.listener.recording and app.state == "idle" and app._task is None


async def test_recording_is_stopped_and_processed_after_max_record_s():
    app, _ = make()
    app.config.conversation.max_record_s = 0.05
    app.ptt_start()
    await asyncio.sleep(0.15)
    assert not app.listener.recording
    await app._task
    assert app.state == "idle" and app.history


async def test_manual_stop_cancels_the_recording_timer():
    app, _ = make()
    app.config.conversation.max_record_s = 0.05
    app.ptt_start()
    await app.ptt_stop()
    await asyncio.sleep(0.15)
    assert len(app.history) == 2  # one turn only


async def test_set_mode_stores_mode_and_close_is_idempotent():
    app, driver = make()
    app.set_mode("wake")
    assert app.config.conversation.mode == "wake"
    await app.close()
    await app.close()
    assert driver.closed


async def test_repeated_ptt_start_does_not_leave_an_orphan_timer():
    app, _ = make()
    app.config.conversation.max_record_s = 0.1
    app.ptt_start()
    await asyncio.sleep(0.05)
    app.ptt_start()  # key auto-repeat re-arms the timer
    await app.ptt_stop()
    app.ptt_start()
    await asyncio.sleep(0.07)  # past the first timer, before the new one
    assert app.listener.recording
    app.ptt_abort()


# --- hands-free modes (continuous / wake) ---

LOUD = b"\x00\x40" * 480
QUIET = b"\x00\x00" * 480
SAY = [LOUD] * 12 + [QUIET] * 3  # 360 ms of speech, then the segmenter's silence


class MicStreams:
    def __init__(self, fail=None):
        self.streams, self.fail = [], fail

    def __call__(self, cb, finished=None):
        test = self

        class S:
            def __init__(self):
                self.cb, self.finished, self.open = cb, finished, False

            def start(self):
                if test.fail:
                    raise test.fail
                self.open = True

            def stop(self):
                self.open = False

            def close(self):
                self.open = False

        self.streams.append(S())
        return self.streams[-1]

    async def say(self, frames=SAY):
        for f in frames:
            self.streams[-1].cb(f)
        await asyncio.sleep(0)


class Recorder(FakeEngine):
    def __init__(self, *transcripts, **kw):
        super().__init__(**kw)
        self.transcripts, self.heard, self.transcribed = list(transcripts), [], 0

    async def transcribe(self, pcm):
        self.transcribed += 1
        if self.transcripts:
            self.transcript = self.transcripts.pop(0)
        return await super().transcribe(pcm)

    async def respond(self, heard, history, on_safe=None):
        self.heard.append(heard)
        return await super().respond(heard, history, on_safe)


def hands_free(mode, engine=None, speaker=None, fail=None):
    mic = MicStreams(fail)
    cfg = load_config(env={})
    cfg.vad.silence_ms, cfg.vad.min_speech_ms = 90, 60
    app = App(cfg, engine=engine or Recorder(), speaker=speaker or FakeSpeaker(), fallback=FakeFallback(), listener=FakeListener(), vision=FakeVision(), mic_stream=mic)
    app.robot = Robot(NullDriver(), cfg, app.bus)
    app.robot.start()
    app.mic.segmenter = Segmenter(silence_ms=90, min_speech_ms=60, is_speech=lambda f: f == LOUD)
    app.set_mode(mode)
    return app, mic


async def settle(app):
    if app._task:
        await asyncio.gather(app._task, return_exceptions=True)


async def test_continuous_every_utterance_is_a_turn_and_mic_listens_again():
    engine = Recorder("raz", "dwa")
    app, mic = hands_free("continuous", engine)
    assert app.state == "listening" and app.mic.running
    for _ in range(2):
        await mic.say()
        await settle(app)
    assert engine.heard == ["raz", "dwa"] and app.state == "listening"


async def test_mic_is_muted_while_a_turn_runs():
    speaker = FakeSpeaker(block=True)
    engine = Recorder("raz", "dwa")
    app, mic = hands_free("continuous", engine, speaker)
    await mic.say()
    await speaker.started.wait()
    await mic.say()  # the robot's own voice or a child talking over it
    speaker.stop()
    await settle(app)
    await mic.say(SAY[-3:])
    assert engine.heard == ["raz"] and app.state == "listening" and not app._busy()


async def test_wake_without_wake_word_is_ignored_with_info():
    engine = Recorder("jak się masz")
    app, mic = hands_free("wake", engine)
    q = app.bus.subscribe()
    await mic.say()
    await settle(app)
    events = drain(q)
    assert engine.heard == [] and app.speaker.played == []
    assert [e.data["text"] for e in events if e.type == "info"] == ["zignorowano (brak słowa „Ohbot”)"]
    assert states(events) == ["transcribing", "listening"]


async def test_wake_word_with_question_answers_the_rest_transcribing_once():
    engine = Recorder("Ohbot, ile to jest dwa plus dwa?")
    app, mic = hands_free("wake", engine)
    await mic.say()
    await settle(app)
    assert engine.heard == ["ile to jest dwa plus dwa?"] and engine.transcribed == 1
    assert app.state == "listening"


async def test_bare_wake_word_says_listening_then_accepts_next_utterance(monkeypatch):
    engine = Recorder("Ohbot!", "opowiedz dowcip", "a teraz zagadka")
    app, mic = hands_free("wake", engine)
    await mic.say()
    await settle(app)
    assert [p.rstrip(b"\0") for p in app.speaker.played] == [b"FB-listening"] and engine.heard == []
    assert app.state == "listening"
    await mic.say()
    await settle(app)
    assert engine.heard == ["opowiedz dowcip"]
    await mic.say()
    await settle(app)
    assert engine.heard == ["opowiedz dowcip"]  # the window is used up


async def test_wake_window_expires():
    engine = Recorder("Ohbot", "opowiedz dowcip")
    app, mic = hands_free("wake", engine)
    app.config.wake.window_s = 0.01
    await mic.say()
    await settle(app)
    await asyncio.sleep(0.2)
    await mic.say([LOUD] * 2 + [QUIET] * 3)  # 150 ms, so it started after the window closed
    await settle(app)
    assert engine.heard == []


async def test_hands_free_empty_transcript_is_ignored_silently():
    app, mic = hands_free("continuous", Recorder(fail=EngineError("empty"), fail_stage="transcribe"))
    q = app.bus.subscribe()
    await mic.say()
    await settle(app)
    events = drain(q)
    assert app.speaker.played == [] and not [e for e in events if e.type == "error"]
    assert [e.data["text"] for e in events if e.type == "info"] == ["(nic nie rozpoznano)"] and app.state == "listening"


async def test_stop_while_speaking_in_continuous_keeps_mic_listening():
    speaker = FakeSpeaker(block=True)
    engine = Recorder("raz", "dwa")
    app, mic = hands_free("continuous", engine, speaker)
    await mic.say()
    await speaker.started.wait()
    app.stop()
    await settle(app)
    assert app.state == "listening" and app.mic.running
    app.stop()  # Stop while just listening must not switch the mic off
    assert app.state == "listening"
    speaker.block = False
    await mic.say()
    await settle(app)
    assert engine.heard == ["raz", "dwa"]


async def test_barge_in_stops_speech_and_the_interruption_becomes_the_next_turn():
    speaker = FakeSpeaker(block=True)
    engine = Recorder("raz", "dwa")
    app, mic = hands_free("continuous", engine, speaker)
    app.config.barge_in.enabled, app.config.barge_in.hold_ms = True, 90
    await mic.say()
    await speaker.started.wait()
    speaker.block = False
    await mic.say(SAY[:3])
    await asyncio.sleep(0)
    assert app.state == "listening"
    await mic.say(SAY[3:])
    await settle(app)
    assert engine.heard == ["raz", "dwa"]


async def test_set_mode_switches_mic_loop_at_runtime():
    app, mic = hands_free("continuous")
    app.set_mode("ptt")
    assert not app.mic.running and not mic.streams[0].open and app.state == "idle"
    app.ptt_start()
    app.set_mode("wake")
    assert not app.listener.recording and app.mic.running and app.state == "listening"
    await app.close()
    assert not app.mic.running


async def test_ptt_button_does_not_record_in_hands_free_mode():
    app, _ = hands_free("continuous")
    app.ptt_start()
    assert not app.listener.recording and app.ptt_stop() is None and app.state == "listening"


async def test_mic_error_falls_back_to_ptt():
    app, _ = hands_free("ptt", fail=OSError("PortAudio: no device"))
    q = app.bus.subscribe()
    app.set_mode("continuous")
    events = drain(q)
    assert [e.data["kind"] for e in events if e.type == "error"] == ["other"]
    assert "mikrofon" in [e for e in events if e.type == "error"][0].data["message"]
    assert app.config.conversation.mode == "ptt" and app.state == "idle" and not app.mic.running


async def test_start_runs_mic_loop_when_configured_hands_free():
    mic = MicStreams()
    cfg = load_config(env={})
    cfg.conversation.mode = "continuous"
    app = App(cfg, engine=Recorder(), speaker=FakeSpeaker(), fallback=FakeFallback(), listener=FakeListener(), robot=Robot(NullDriver(), cfg, EventBus()), vision=FakeVision(), mic_stream=mic)
    await app.start()
    assert app.mic.running and app.state == "listening" and app.mic_error is None
    await app.close()


async def test_mic_error_at_start_is_kept_for_clients_that_connect_later():
    cfg = load_config(env={})
    cfg.conversation.mode = "wake"
    mic = MicStreams(OSError("PortAudio: no device"))
    app = App(cfg, engine=Recorder(), speaker=FakeSpeaker(), fallback=FakeFallback(), listener=FakeListener(), robot=Robot(NullDriver(), cfg, EventBus()), vision=FakeVision(), mic_stream=mic)
    await app.start()
    assert "mikrofon" in app.mic_error and cfg.conversation.mode == "ptt"
    app.ptt_start()
    assert app.mic_error is None
    await app.close()


async def test_mic_lost_mid_session_falls_back_to_ptt():
    app, mic = hands_free("wake")
    q = app.bus.subscribe()
    mic.streams[-1].finished()
    await asyncio.sleep(0)
    events = drain(q)
    message = "Mikrofon odłączony – wracam do trybu z przyciskiem"
    assert [e.data["message"] for e in events if e.type == "error"] == [message]
    assert [e.data["mode"] for e in events if e.type == "settings"] == ["ptt"]
    assert app.config.conversation.mode == "ptt" and app.state == "idle" and not app.mic.running
    assert app.mic_error == message


async def test_mode_switch_during_stt_does_not_change_the_running_turn():
    engine = Recorder("raz")
    app, mic = hands_free("continuous", engine)
    orig = engine.transcribe

    async def switching(pcm):
        app.set_mode("wake")
        return await orig(pcm)

    engine.transcribe = switching
    await mic.say()
    await settle(app)
    assert engine.heard == ["raz"]


async def test_wake_window_counts_from_when_the_child_spoke_not_after_stt():
    engine = Recorder("Ohbot", "opowiedz dowcip")
    orig = engine.transcribe

    async def slow(pcm):
        text = await orig(pcm)
        if engine.transcribed == 2:
            await asyncio.sleep(0.1)  # STT latency
        return text

    engine.transcribe = slow
    app, mic = hands_free("wake", engine)
    app.config.wake.window_s = 0.05
    await mic.say()
    await settle(app)
    await mic.say()
    await settle(app)
    assert engine.heard == ["opowiedz dowcip"]


@pytest.mark.parametrize("first", ["Ohbot!", "Ohbot, opowiedz bajkę"])
async def test_wake_barge_in_opens_the_window_for_the_interruption(first):
    speaker = FakeSpeaker(block=True)
    engine = Recorder(first, "opowiedz dowcip")
    app, mic = hands_free("wake", engine, speaker)
    app.config.barge_in.enabled, app.config.barge_in.hold_ms = True, 90
    await mic.say()
    await speaker.started.wait()  # "Słucham?" or the reply
    speaker.block = False
    await mic.say(SAY[:3])  # the child talks over it
    await asyncio.sleep(0)
    await mic.say(SAY[3:])
    await settle(app)
    assert engine.heard[-1] == "opowiedz dowcip"


@pytest.mark.parametrize("stage,shown", [("respond", "(ukryto)"), ("reply", "Cześć")])
async def test_transcript_shown_only_after_input_moderation_passed(stage, shown):
    app, _ = make(FakeEngine(fail=EngineError("moderated"), fail_stage=stage))
    q = app.bus.subscribe()
    await app.handle_utterance(LONG_ENOUGH)
    assert [e.data["text"] for e in drain(q) if e.type == "heard"] == [shown]


async def test_switch_to_ptt_mid_turn_ends_idle_with_mic_off():
    speaker = FakeSpeaker(block=True)
    app, mic = hands_free("continuous", Recorder("raz"), speaker)
    await mic.say()
    await speaker.started.wait()
    app.set_mode("ptt")
    speaker.stop()
    await settle(app)
    assert app.state == "idle" and not app.mic.running


@pytest.mark.parametrize("mode,transcript", [("wake", "co tam"), ("continuous", "")])
async def test_mic_lost_during_stt_then_ignored_ends_idle(mode, transcript):
    # regression: _ignore went to LISTENING although the mic was already off
    engine = Recorder(transcript)
    gate = asyncio.Event()
    orig = engine.transcribe

    async def blocked(pcm):
        await gate.wait()
        text = await orig(pcm)
        if not text:
            raise EngineError("empty")
        return text

    engine.transcribe = blocked
    app, mic = hands_free(mode, engine)
    await mic.say()
    mic.streams[-1].finished()
    await asyncio.sleep(0)
    gate.set()
    await settle(app)
    assert app.state == "idle" and not app.mic.running


class GestureEngine(FakeEngine):
    async def respond(self, heard, history, on_safe=None):
        return Reply(heard=heard, text="Tak!", pcm=SPEECH_PCM, gesture="yes")


async def test_reply_gesture_runs_on_the_robot_while_speaking():
    speaker = FakeSpeaker(block=True)
    app, _ = make(GestureEngine(), speaker=speaker)
    seen = []

    async def gesture(name):
        seen.append((name, app.state))

    app.robot.gesture = gesture
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    assert seen == [("yes", "speaking")]
    speaker.stop()
    await task


async def test_stop_cuts_the_gesture_short():
    speaker = FakeSpeaker(block=True)
    app, _ = make(GestureEngine(), speaker=speaker)
    cancelled = asyncio.Event()

    async def gesture(name):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    app.robot.gesture = gesture
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    app.stop()
    await asyncio.gather(task, return_exceptions=True)
    assert cancelled.is_set()
    assert app.state == "idle"


def slow_gesture(app):
    """Replaces robot.gesture with one that outlives the audio; returns the events of each call."""
    calls = []

    async def gesture(name):
        cancelled = asyncio.Event()
        calls.append(cancelled)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    app.robot.gesture = gesture
    return calls


async def test_turn_ends_with_the_audio_and_stop_cancels_the_leftover_gesture():
    app, _ = make(GestureEngine())
    calls = slow_gesture(app)
    await app.handle_utterance(LONG_ENOUGH)
    await asyncio.sleep(0)  # FakeSpeaker never yields, the real one plays in a thread
    assert app.state == "idle"  # a gesture still moving must not keep the mic muted
    assert not calls[0].is_set()
    app.stop()
    await asyncio.sleep(0)
    assert calls[0].is_set()


async def test_next_turn_cancels_the_leftover_gesture():
    app, _ = make(GestureEngine())
    calls = slow_gesture(app)
    await app.handle_utterance(LONG_ENOUGH)
    await asyncio.sleep(0)
    await app.handle_utterance(LONG_ENOUGH)
    await asyncio.sleep(0)
    assert calls[0].is_set() and not calls[1].is_set()


async def test_close_lets_the_leftover_gesture_reach_neutral():
    app, driver = make(GestureEngine())
    app.config.robot.gesture_step_s = 60
    await app.handle_utterance(LONG_ENOUGH)
    await asyncio.sleep(0)
    await app.close()
    assert ("move", HEADNOD, 5, app.config.robot.gesture_speed) in driver.calls
    assert driver.closed


GREETING = "Cześć! Jestem Ohbot. O czym porozmawiamy?"


async def test_greeting_is_spoken_moderated_once_then_from_cache_and_never_in_history():
    engine = FakeEngine()
    app, _ = make(engine)
    q = app.bus.subscribe()
    await app.greet()
    await app.greet()
    assert engine.said == [(GREETING, None)]  # the second one comes from the cache
    assert app.speaker.played == [b"SAY-Cze\xc5", b"SAY-Cze\xc5"] and app.history == []
    events = drain(q)
    assert states(events) == ["thinking", "speaking", "idle"] * 2
    assert [e.data["text"] for e in events if e.type == "reply"] == [GREETING, GREETING]


async def test_new_conversation_greeting_waits_for_the_stopped_turn():
    gate = asyncio.Event()

    class Slow(FakeEngine):
        async def respond(self, heard, history, on_safe=None):
            await gate.wait()
            return await super().respond(heard, history, on_safe)

    app, _ = make(Slow())
    app.start_turn(LONG_ENOUGH)
    await asyncio.sleep(0.01)
    app.new_conversation()
    await app.greet()
    assert app.speaker.played == [b"SAY-Cze\xc5"] and app.history == [] and app.state == "idle"


async def test_flagged_greeting_or_test_sentence_plays_the_moderated_phrase():
    app, _ = make(FakeEngine(fail=EngineError("moderated"), fail_stage="say"))
    await app.say("brzydkie")
    assert app.fallback.asked == ["moderated"] and app.fallback.texts == {} and app.state == "idle"


async def test_voice_sample_uses_the_given_voice_and_is_not_cached():
    engine = FakeEngine()
    app, _ = make(engine)
    await app.voice_sample("ash")
    assert engine.said[0][1] == "ash" and app.fallback.texts == {}


@pytest.mark.parametrize("connected", [True, False])
async def test_start_greets_only_with_the_robot_connected(connected):
    cfg = load_config(env={})
    engine = FakeEngine()
    app = App(cfg, engine=engine, speaker=FakeSpeaker(), fallback=FakeFallback(), listener=FakeListener(), robot=Robot(NullDriver(connected), cfg, EventBus()), vision=FakeVision(), mic_stream=MicStreams())
    await app.start()
    await settle(app)
    assert getattr(engine, "said", []) == ([(GREETING, None)] if connected else [])
    await app.close()


async def test_hands_free_greeting_mutes_the_mic_and_listens_again():
    app, mic = hands_free("continuous")
    q = app.bus.subscribe()
    await app.greet()
    assert states(drain(q)) == ["thinking", "speaking", "idle", "listening"]


async def test_ready_only_when_idle_and_free():
    app, _ = make()
    assert app.ready()
    app.ptt_start()  # LISTENING, but a child is talking
    assert not app.ready()


async def test_hands_free_resting_in_listening_is_ready():
    app, mic = hands_free("continuous")
    assert app.state == "listening" and app.ready()
    await mic.say()
    assert not app.ready()  # the utterance is being handled
    await settle(app)
    assert app.ready()


async def test_open_test_panel_mutes_the_hands_free_mic():
    engine = Recorder()
    app, mic = hands_free("continuous", engine)
    app.robot.hold_idle = True  # servo whine from slider moves must not open an utterance
    await mic.say()
    await settle(app)
    assert engine.transcribed == 0 and app.state == "listening"
    app.robot.hold_idle = False
    await mic.say()
    await settle(app)
    assert engine.transcribed == 1


class SlowEngine(FakeEngine):
    def __init__(self, delay=0.05, **kw):
        super().__init__(**kw)
        self.delay = delay

    async def respond(self, heard, history, on_safe=None):
        await asyncio.sleep(self.delay)
        return await super().respond(heard, history, on_safe)


def with_filler(app):
    app.fallback.texts["Hmm…"] = b"FILLER"


async def test_slow_reply_gets_a_filler_first_and_the_filler_stays_out_of_history(monkeypatch):
    monkeypatch.setattr("ohbotchat.app.FILLER_DELAY_S", 0.01)
    app, _ = make(SlowEngine())
    with_filler(app)
    await app.handle_utterance(LONG_ENOUGH)
    assert app.speaker.played == [b"FILLER", SPEECH_PCM]
    assert [m["content"] for m in app.history] == ["Cześć", "Hej!"]


async def test_fast_reply_or_fillers_off_has_no_filler(monkeypatch):
    app, _ = make()
    with_filler(app)
    await app.handle_utterance(LONG_ENOUGH)
    assert app.speaker.played == [SPEECH_PCM]
    monkeypatch.setattr("ohbotchat.app.FILLER_DELAY_S", 0.01)
    app, _ = make(SlowEngine())
    with_filler(app)
    app.config.conversation.fillers = False
    await app.handle_utterance(LONG_ENOUGH)
    assert app.speaker.played == [SPEECH_PCM]


async def test_stop_during_the_filler_cuts_the_filler_and_the_reply(monkeypatch):
    monkeypatch.setattr("ohbotchat.app.FILLER_DELAY_S", 0.01)
    engine = SlowEngine(delay=0.05)
    speaker = FakeSpeaker(block=True)
    app, _ = make(engine, speaker)
    with_filler(app)
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    app.stop()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.1)  # the reply would have arrived by now
    assert speaker.played == [b"FILLER"] and app.history == [] and app.state == "idle"


async def test_apply_settings_switches_voice_cache_and_vad_silence():
    app, _ = make()
    app.apply_settings({"openai": {"tts_voice": "ash"}, "vad": {"silence_ms": 900}, "robot": {"lip_open": 7.0}})
    assert app.config.openai.tts_voice == "ash" and app.config.robot.lip_open == 7.0
    assert app.fallback.dir.name.endswith("-ash")
    await app._ensure_task
    assert app.fallback.ensured == 1
    assert app.mic.segmenter._silence_frames == 900 // 30


async def test_manual_gesture_runs_and_pauses_idle():
    app, driver = make()
    app.config.robot.gesture_step_s = 0
    app.manual_gesture("yes")
    await app._gesture
    app.robot.flush()
    assert ("move", HEADNOD, 7, app.config.robot.gesture_speed) in driver.calls
    assert app.robot._idle_after > 0


async def test_new_conversation_while_the_button_is_held_drops_the_recording_before_greeting():
    app, _ = make()
    app.ptt_start()
    app.new_conversation()
    assert not app.listener.recording and app.state == "idle"
    await app.greet()
    assert app.ptt_stop() is None and app.state == "idle"  # releasing the button later starts nothing


async def test_reply_error_during_a_stopped_filler_is_retrieved(monkeypatch):
    monkeypatch.setattr("ohbotchat.app.FILLER_DELAY_S", 0.01)
    made = []
    create = asyncio.create_task
    monkeypatch.setattr(asyncio, "create_task", lambda coro, **kw: made.append(create(coro, **kw)) or made[-1])

    class Failing(FakeEngine):
        async def respond(self, heard, history, on_safe=None):
            await asyncio.sleep(0.02)
            raise EngineError("network")

    speaker = FakeSpeaker(block=True)
    app, _ = make(Failing(), speaker)
    with_filler(app)
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    await asyncio.sleep(0.05)  # the reply fails while the filler still plays
    app.stop()
    await asyncio.gather(task, return_exceptions=True)
    pending = next(t for t in made if "respond" in t.get_coro().__qualname__)
    # _log_traceback is what makes asyncio log "Task exception was never retrieved" once the task is collected
    assert pending.done() and not pending._log_traceback


async def test_voice_change_restarts_phrase_generation_for_the_new_voice():
    class Slow(FakeFallback):
        def __init__(self):
            super().__init__(complete=False)
            self.cancelled = 0

        async def ensure(self, engine):
            self.ensured += 1
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise

    fb = Slow()
    app, _ = make(fallback=fb)
    app._ensure_fallback()
    await asyncio.sleep(0)
    app.apply_settings({"openai": {"tts_voice": "ash"}})
    await asyncio.sleep(0)
    assert fb.cancelled == 1 and fb.ensured == 2
    app._ensure_task.cancel()


FACE = Face(0.0, 0.0, 0.0)
GREETING_PCM = b"SAY-Cze\xc5"


def arrive(app, t):
    """A face seen at t and still there 1.1 s later, as the tick loop would report it."""
    app._auto_greet(t, FACE)
    app._auto_greet(t + 1.1, FACE)


async def test_child_who_walks_up_is_greeted_once_after_a_second():
    engine = FakeEngine()
    app, _ = make(engine)
    t = time.monotonic()
    app._auto_greet(t, FACE)
    app._auto_greet(t + 0.5, FACE)
    assert app._task is None  # walking past
    app._auto_greet(t + 1.1, FACE)
    await settle(app)
    assert engine.said == [(GREETING, None)]
    arrive(app, t + 30)  # left and came back within greet_cooldown_s
    await settle(app)
    assert app.speaker.played == [GREETING_PCM]


async def test_child_arriving_after_the_cooldown_gets_a_fresh_conversation():
    app, _ = make()
    t = time.monotonic()
    arrive(app, t)
    await settle(app)
    await app.handle_utterance(LONG_ENOUGH)
    assert app.history
    arrive(app, t + 62.1)  # nobody for 61 s
    await settle(app)
    assert app.speaker.played.count(GREETING_PCM) == 2 and app.history == []


async def test_no_auto_greet_right_after_a_greeting_or_when_switched_off():
    app, _ = make()
    await app.greet()  # startup or "Nowa rozmowa"
    arrive(app, time.monotonic())
    await settle(app)
    assert app.speaker.played == [GREETING_PCM]
    app, _ = make()
    app.config.camera.auto_greet = False
    arrive(app, time.monotonic())
    await settle(app)
    assert app.speaker.played == []


async def test_no_auto_greet_for_a_child_already_talking_to_the_robot():
    speaker = FakeSpeaker(block=True)
    app, _ = make(speaker=speaker)
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    t = time.monotonic()
    arrive(app, t)
    speaker.stop()
    await asyncio.gather(task, return_exceptions=True)
    app._auto_greet(t + 1.5, FACE)
    await settle(app)
    assert GREETING_PCM not in speaker.played


async def test_auto_greet_in_wake_mode_opens_the_window_for_the_child():
    engine = Recorder("opowiedz dowcip")
    app, mic = hands_free("wake", engine)
    arrive(app, time.monotonic())
    await settle(app)
    await mic.say()
    await settle(app)
    assert engine.heard == ["opowiedz dowcip"]  # no "Ohbot" needed


async def test_auto_greet_in_ptt_mode_only_greets():
    app, _ = make()
    arrive(app, time.monotonic())
    await settle(app)
    assert app.speaker.played == [GREETING_PCM] and app._wake_until == 0.0 and app.state == "idle"


async def test_start_runs_the_camera_and_the_tick_feeds_its_face_to_the_robot():
    cfg = load_config(env={})
    driver, vision = NullDriver(), FakeVision()
    vision.face = Face(0.0, 0.0, time.monotonic())
    app = App(cfg, engine=FakeEngine(), speaker=FakeSpeaker(), fallback=FakeFallback(), listener=FakeListener(), robot=Robot(driver, cfg, EventBus()), vision=vision, mic_stream=MicStreams())
    await app.start()
    await asyncio.sleep(0.05)
    app.robot.flush()
    assert vision.running and ("move", EYETURN, 5.0, 8) in driver.calls
    await app.close()
    assert not vision.running


async def test_a_stale_face_from_a_hung_camera_is_not_tracked():
    cfg = load_config(env={})
    driver, vision = NullDriver(), FakeVision()
    vision.face = Face(0.5, 0.0, time.monotonic() - 2)  # cap.read() hung after this frame
    app = App(cfg, engine=FakeEngine(), speaker=FakeSpeaker(), fallback=FakeFallback(), listener=FakeListener(), robot=Robot(driver, cfg, EventBus()), vision=vision, mic_stream=MicStreams())
    await app.start()
    await asyncio.sleep(0.05)
    app.robot.flush()
    assert ("move", EYETURN, 7.0, 8) not in driver.calls
    await app.close()


async def test_calibrate_sets_offsets_so_the_tracked_pose_matches_the_slider_pose():
    app, _ = make()
    c = app.config.camera
    c.head_gain, c.offset_x, c.offset_y = 2.0, 0.1, 0.0
    app.vision.face = Face(0.3, -0.2, time.monotonic())  # raw x 0.2, raw y -0.2
    app.robot.move(HEADTURN, 7.0)  # operator turned the head to look at the child
    app.robot.move(HEADNOD, 6.0)
    msg = app.calibrate()
    assert (c.offset_x, c.offset_y) == (0.8, 0.7)  # (7-5)/2 - 0.2, (6-5)/2 + 0.2
    assert "Zapisz" in msg


async def test_calibrate_without_a_face_changes_nothing():
    app, _ = make()
    app.config.camera.offset_x = 0.1
    app.vision.face = None
    assert "twarz" in app.calibrate()
    assert app.config.camera.offset_x == 0.1
