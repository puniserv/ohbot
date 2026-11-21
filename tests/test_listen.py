import pytest

from ohbotchat.config import load_config
from ohbotchat.events import EventBus
from ohbotchat.listen import FRAME_BYTES, Listener, Segmenter, wake_match

SPEECH, QUIET = b"\x01" * FRAME_BYTES, b"\x00" * FRAME_BYTES


def seg(**kw):
    return Segmenter(silence_ms=90, min_speech_ms=60, is_speech=lambda f: f == SPEECH, **kw)


def feed(s, pattern):
    return [u for u in (s.feed(SPEECH if c == "s" else QUIET) for c in pattern) if u is not None]


def test_segmenter_yields_after_silence():
    out = feed(seg(), "qqsssssqqq")
    assert len(out) == 1
    assert out[0].count(SPEECH) == 5


def test_segmenter_waits_for_full_silence():
    assert feed(seg(), "ssssqq") == []


def test_segmenter_drops_clicks():
    assert feed(seg(), "qsqqqqq") == []


def test_segmenter_two_utterances():
    assert len(feed(seg(), "ssssqqqqssssqqq")) == 2


def test_segmenter_keeps_a_little_lead_in():
    out = feed(seg(), "qqqqqqssssqqq")[0]
    assert out.startswith(QUIET) and out.count(QUIET) > 3


def test_segmenter_rejects_wrong_frame_size():
    with pytest.raises(ValueError):
        seg().feed(b"\x00")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Ohbot, jak się masz?", "jak się masz?"),
        ("ochbot ile to jest", "ile to jest"),
        ("Hej Ohbot! Opowiedz dowcip", "Opowiedz dowcip"),
        ("OŁBOT", ""),
        ("robot jak się masz", None),
        ("a potem ohbot", None),
        ("", None),
    ],
)
def test_wake_match(text, expected):
    assert wake_match(text, load_config(env={}).wake.words) == expected


class FakeStream:
    def __init__(self, cb):
        self.cb, self.log = cb, []

    def start(self):
        self.log.append("start")

    def stop(self):
        self.log.append("stop")

    def close(self):
        self.log.append("close")


def test_ptt_collects_frames_between_start_and_stop():
    streams = []

    def factory(cb):
        streams.append(FakeStream(cb))
        return streams[-1]

    lis = Listener(load_config(env={}), EventBus(), factory)
    lis.start_ptt()
    streams[0].cb(b"ab")
    streams[0].cb(b"cd")
    assert lis.stop_ptt() == b"abcd"
    assert streams[0].log == ["start", "stop", "close"]
    assert lis.stop_ptt() == b""


def test_ptt_frames_after_stop_are_dropped():
    streams = []
    lis = Listener(load_config(env={}), EventBus(), lambda cb: streams.append(FakeStream(cb)) or streams[-1])
    lis.start_ptt()
    lis.stop_ptt()
    streams[0].cb(b"late")
    lis.start_ptt()
    assert lis.stop_ptt() == b""


def test_start_ptt_twice_keeps_first_recording():
    streams = []
    lis = Listener(load_config(env={}), EventBus(), lambda cb: streams.append(FakeStream(cb)) or streams[-1])
    lis.start_ptt()
    streams[0].cb(b"ab")
    lis.start_ptt()  # key auto-repeat
    assert len(streams) == 1 and lis.recording
    assert lis.stop_ptt() == b"ab" and not lis.recording


def test_ptt_publishes_level_for_the_meter():
    streams, bus = [], EventBus()
    q = []
    bus._deliver = q.append  # no loop needed to observe what is published
    lis = Listener(load_config(env={}), bus, lambda cb: streams.append(FakeStream(cb)) or streams[-1])
    lis.start_ptt()
    for _ in range(3):
        streams[0].cb(b"\x00\x40" * 480)
    lis.stop_ptt()
    assert [e.data["level"] for e in q if e.type == "level"] == [pytest.approx(0.5, abs=0.01), 0.0]
