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
    def __init__(self, cb, fail=None, finished=None):
        self.cb, self.fail, self.log, self.finished = cb, fail, [], finished

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
        self.utterances, self.barges, self.streams, self.lost = [], 0, [], 0

        def factory(cb, finished=None):
            self.streams.append(FakeStream(cb, fail, finished))
            return self.streams[-1]

        def barge():
            self.barges += 1
            self.state = State.LISTENING

        def lost():
            self.lost += 1

        self.mic = MicLoop(self.cfg, self.bus, lambda: self.state, self.utterances.append, barge, lost, factory)
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


async def test_stream_finishing_on_its_own_reports_the_mic_lost():
    rig = Rig()
    rig.mic.start()
    stream = rig.streams[0]
    await asyncio.to_thread(stream.finished)  # PortAudio calls it from its own thread
    await asyncio.sleep(0)
    assert rig.lost == 1 and not rig.mic.running and stream.log[-1] == "close"


async def test_own_stop_is_not_reported_as_mic_lost():
    rig = Rig()
    rig.mic.start()
    rig.mic.stop()
    rig.streams[0].finished()
    await asyncio.sleep(0)
    assert rig.lost == 0


async def test_no_frames_for_a_while_reports_the_mic_lost(monkeypatch):
    monkeypatch.setattr("ohbotchat.loop.NO_FRAMES_S", 0.05)
    rig = Rig()
    rig.mic.start()
    for _ in range(4):
        await asyncio.sleep(0.03)
        await rig.feed(QUIET)
    assert rig.lost == 0
    await asyncio.sleep(0.1)
    assert rig.lost == 1 and not rig.mic.running


async def test_mic_lost_is_reported_even_when_the_dead_stream_fails_to_stop():
    rig = Rig()
    levels = []
    rig.bus._deliver = levels.append
    rig.mic.start()
    stream = rig.streams[0]

    def dead():
        raise OSError("device gone")

    stream.stop = dead
    with pytest.raises(OSError):
        rig.mic._finished(stream)  # what the finished callback runs on the loop
    assert rig.lost == 1 and not rig.mic.running and stream.log == ["start", "close"]
    assert levels[-1].data == {"level": 0.0}


def test_segmenter_max_length_comes_from_max_record_s():
    rig = Rig()
    rig.cfg.conversation.max_record_s = 0.3
    mic = MicLoop(rig.cfg, rig.bus, lambda: rig.state, rig.utterances.append, None, None, None)
    assert mic.segmenter._max_frames == 10


async def test_echo_tail_after_speaking_is_not_an_utterance(monkeypatch):
    monkeypatch.setattr("ohbotchat.loop.ECHO_TAIL_S", 0.05)
    rig = Rig()
    rig.mic.start()
    rig.state = State.SPEAKING
    await rig.feed(QUIET)
    rig.state = State.LISTENING
    await rig.feed(*[LOUD] * 4, *[QUIET] * 3)  # servo whine or the speaker's last syllable
    assert rig.utterances == []
    await asyncio.sleep(0.06)
    await rig.feed(*[LOUD] * 4, *[QUIET] * 3)
    assert len(rig.utterances) == 1


async def test_barge_in_speech_is_kept_while_the_turn_is_still_being_cancelled():
    rig = Rig()
    rig.cfg.barge_in.enabled = True
    rig.mic.start()
    rig.state = State.SPEAKING
    await rig.feed(LOUD, LOUD, LOUD)
    rig.state = State.SPEAKING  # the cancelled turn has not reached LISTENING yet
    await rig.feed(QUIET)
    rig.state = State.LISTENING
    await rig.feed(LOUD, QUIET, QUIET, QUIET)
    assert len(rig.utterances) == 1
