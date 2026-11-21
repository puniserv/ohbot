import asyncio

import pytest

from ohbotchat.app import App
from ohbotchat.config import load_config
from ohbotchat.engine import EngineError, Reply
from ohbotchat.robot import BOTTOMLIP, TOPLIP, NullDriver, Robot

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

    async def respond(self, heard, history):
        self.history_seen.append(list(history))
        if self.fail and self.fail_stage == "respond":
            raise self.fail
        return Reply(heard=heard, text="Hej!", pcm=SPEECH_PCM)

    async def synthesize(self, text):
        return b"\x00\x00"


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

    def complete(self):
        return self._complete

    def get(self, key):
        self.asked.append(key)
        pcm = b"FB-" + key.encode()
        return pcm + b"\0" * (len(pcm) % 2)

    async def ensure(self, engine):
        self.ensured += 1


class FakeListener:
    def __init__(self):
        self.recording, self.starts = False, 0

    def start_ptt(self):
        self.starts += 1
        self.recording = True

    def stop_ptt(self):
        self.recording = False
        return LONG_ENOUGH


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
    assert [c for c in driver.calls if c[0] == "move"][-2:] == [("move", TOPLIP, 5.0, 10), ("move", BOTTOMLIP, 5.0, 10)]  # mouth closed at the end


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
        async def respond(self, heard, history):
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
            async def respond(self, heard, history):
                raise FileNotFoundError("persona")

        app, _ = make(Broken())
    q = app.bus.subscribe()
    await app.handle_utterance(LONG_ENOUGH)
    assert app.state == "idle"
    assert [e.data["kind"] for e in drain(q) if e.type == "error"] == ["other"]


async def test_lip_delay_defers_mouth_moves_including_closing():
    app, driver = make(delay=200)
    await app.handle_utterance(LONG_ENOUGH)
    app.robot.flush()
    assert not [c for c in driver.calls if c[0] == "move"]
    await asyncio.sleep(0.4)
    app.robot.flush()
    moves = [c for c in driver.calls if c[0] == "move"]
    assert moves and moves[-2:] == [("move", TOPLIP, 5.0, 10), ("move", BOTTOMLIP, 5.0, 10)]


async def test_missing_fallback_phrases_are_retried_after_a_good_turn():
    fb = FakeFallback(complete=False)
    app, _ = make(fallback=fb)
    await app.handle_utterance(LONG_ENOUGH)
    await app.wait_background()
    assert fb.ensured == 1
    fb2 = FakeFallback(complete=True)
    app, _ = make(fallback=fb2)
    await app.handle_utterance(LONG_ENOUGH)
    await app.wait_background()
    assert fb2.ensured == 0


def test_default_engine_does_not_retry_in_sdk():
    cfg = load_config(env={})
    cfg.api_key = "sk-test"
    app = App(cfg, speaker=FakeSpeaker(), fallback=FakeFallback(), listener=FakeListener(), robot=object())
    assert app.engine._client.max_retries == 0


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
