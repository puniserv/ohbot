import asyncio

import pytest

from ohbotchat.config import load_config
from ohbotchat.events import EventBus
from ohbotchat.listen import FRAME_BYTES, Segmenter
from ohbotchat.loop import MicLoop
from ohbotchat.state import State

LOUD = b"\x00\x40" * (FRAME_BYTES // 2)  # level 0.5
SOFT = b"\x00\x02" * (FRAME_BYTES // 2)  # level ~0.016
QUIET = b"\x00\x00" * (FRAME_BYTES // 2)


class FakeStream:
    def __init__(self, cb, fail=None):
        self.cb, self.fail, self.log = cb, fail, []

    def start(self):
        if self.fail:
            raise self.fail
        self.log.append("start")

    def stop(self):
        self.log.append("stop")

    def close(self):
        self.log.append("close")


class Rig:
    def __init__(self, fail=None):
        self.cfg = load_config(env={})
        self.cfg.vad.silence_ms, self.cfg.vad.min_speech_ms = 90, 60
        self.cfg.barge_in.threshold, self.cfg.barge_in.hold_ms = 0.15, 90
        self.bus = EventBus()
        self.state = State.LISTENING
        self.utterances, self.barges, self.streams = [], 0, []

        def factory(cb):
            self.streams.append(FakeStream(cb, fail))
            return self.streams[-1]

        def barge():
            self.barges += 1
            self.state = State.LISTENING

        self.mic = MicLoop(self.cfg, self.bus, lambda: self.state, self.utterances.append, barge, factory)
        self.mic.segmenter = Segmenter(silence_ms=90, min_speech_ms=60, is_speech=lambda f: f == LOUD)

    async def feed(self, *frames):
        for f in frames:
            self.streams[-1].cb(f)  # what the PortAudio thread does
        await asyncio.sleep(0)


async def test_listening_frames_become_utterances():
    rig = Rig()
    rig.mic.start()
    await rig.feed(*[LOUD] * 4, *[QUIET] * 3)
    assert len(rig.utterances) == 1 and rig.utterances[0].count(LOUD) == 4


async def test_chunks_of_any_size_are_cut_into_frames():
    rig = Rig()
    rig.mic.start()
    await rig.feed(LOUD * 4 + QUIET, QUIET * 2)
    assert len(rig.utterances) == 1


@pytest.mark.parametrize("state", [State.SPEAKING, State.TRANSCRIBING, State.THINKING, State.IDLE])
async def test_frames_dropped_outside_listening_and_segmenter_reset(state):
    rig = Rig()
    rig.mic.start()
    await rig.feed(LOUD, LOUD, LOUD)
    rig.state = state
    await rig.feed(LOUD, LOUD, QUIET, QUIET, QUIET)
    rig.state = State.LISTENING
    await rig.feed(QUIET, QUIET, QUIET)
    assert rig.utterances == []


async def test_level_published_about_ten_times_a_second():
    rig = Rig()
    q = rig.bus.subscribe()
    rig.mic.start()
    await rig.feed(*[QUIET] * 5, LOUD, *[QUIET] * 3)
    levels = [e.data["level"] for e in [q.get_nowait() for _ in range(q.qsize())] if e.type == "level"]
    assert levels == [0.0, pytest.approx(0.5, abs=0.01), 0.0]


async def test_barge_in_after_hold_and_trigger_starts_next_utterance():
    rig = Rig()
    rig.cfg.barge_in.enabled = True
    rig.mic.start()
    rig.state = State.SPEAKING
    await rig.feed(LOUD, LOUD)
    assert rig.barges == 0
    await rig.feed(LOUD)
    assert rig.barges == 1 and rig.state == State.LISTENING
    await rig.feed(QUIET, QUIET, QUIET)
    assert len(rig.utterances) == 1 and rig.utterances[0].count(LOUD) == 3


async def test_no_barge_in_below_threshold_or_with_gaps_or_disabled():
    rig = Rig()
    rig.cfg.barge_in.enabled = True
    rig.mic.start()
    rig.state = State.SPEAKING
    await rig.feed(*[SOFT] * 10, LOUD, LOUD, QUIET, QUIET, QUIET, LOUD, LOUD, QUIET, QUIET, QUIET)
    rig.cfg.barge_in.enabled = False
    await rig.feed(*[LOUD] * 10)
    assert rig.barges == 0


async def test_barge_in_hold_tolerates_short_gaps():
    rig = Rig()
    rig.cfg.barge_in.enabled = True
    rig.mic.start()
    rig.state = State.SPEAKING
    await rig.feed(LOUD, QUIET, QUIET, LOUD, QUIET, LOUD)  # speech between syllables dips for a frame or two
    assert rig.barges == 1


async def test_stop_closes_stream_and_ignores_late_frames():
    rig = Rig()
    rig.mic.start()
    rig.mic.start()
    assert len(rig.streams) == 1 and rig.mic.running
    rig.mic.stop()
    assert rig.streams[0].log == ["start", "stop", "close"] and not rig.mic.running
    await rig.feed(*[LOUD] * 4, *[QUIET] * 3)
    assert rig.utterances == []


async def test_start_failure_raises_and_leaves_loop_stopped():
    rig = Rig(fail=OSError("no input device"))
    with pytest.raises(OSError):
        rig.mic.start()
    assert not rig.mic.running and rig.streams[0].log == ["close"]
