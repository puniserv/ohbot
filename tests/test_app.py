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
    def __init__(self, block=False):
        self.played, self.block = [], block
        self.started = asyncio.Event()
        self._stop = asyncio.Event()

    async def play(self, pcm, rate=24000, on_frame=None):
        self.played.append(pcm)
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
    def __init__(self):
        self.asked = []

    def get(self, key):
        self.asked.append(key)
        pcm = b"FB-" + key.encode()
        return pcm + b"\0" * (len(pcm) % 2)

    async def ensure(self, engine):
        pass


def make(engine=None, speaker=None, turns=None):
    cfg = load_config(env={})
    if turns:
        cfg.conversation.history_turns = turns
    driver = NullDriver()
    app = App(cfg, engine=engine or FakeEngine(), speaker=speaker or FakeSpeaker(), fallback=FakeFallback())
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


async def test_stop_during_speaking_returns_to_listening():
    speaker = FakeSpeaker(block=True)
    app, _ = make(speaker=speaker)
    task = asyncio.create_task(app.handle_utterance(LONG_ENOUGH))
    await speaker.started.wait()
    app.stop()
    await task
    assert app.state == "listening"


async def test_stop_during_thinking_cancels_request():
    gate = asyncio.Event()

    class Slow(FakeEngine):
        async def respond(self, heard, history):
            await gate.wait()

    app, _ = make(Slow())
    task = asyncio.create_task(app.handle_utterance(LONG_ENOUGH))
    while app.state != "thinking":
        await asyncio.sleep(0)
    app.stop()
    await task
    assert app.state == "listening" and not task.cancelled()


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
