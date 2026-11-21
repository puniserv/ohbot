import threading
import asyncio
import time

import numpy as np

from ohbotchat.audio import envelope, level, lip_positions
from ohbotchat.speak import Speaker

RATE = 24000


def sine(seconds, amp=1.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 440 * t) * amp * 32767).astype("<i2").tobytes()


def silence(seconds, rate=RATE):
    return bytes(int(seconds * rate) * 2)


def test_level():
    assert level(silence(0.1)) == 0
    assert 0.65 < level(sine(0.1)) < 0.75
    assert level(b"") == 0


def test_envelope_length_and_range():
    env = envelope(sine(1.0, 0.5), RATE, frame_ms=40)
    assert len(env) == 25
    assert all(0 <= v <= 1 for v in env)
    assert envelope(silence(1.0), RATE) == [0.0] * 25


def test_lip_positions():
    assert lip_positions([0.0]) == [(5, 5)]
    (top, bottom), = lip_positions([1.0])
    # regression: the ohbot lib opens both lips upward from 5 (5 + val/2); the top lip pushed below 5 jammed
    assert top > 5 and bottom > 5
    assert all(0 <= p <= 10 for p in (top, bottom))


class FakeStream:
    def __init__(self, delay=0.0):
        self.delay, self.written, self.aborted, self.stopped = delay, 0, False, False

    def start(self): pass

    def write(self, data):
        time.sleep(self.delay)
        self.written += len(data)

    def stop(self): self.stopped = True

    def abort(self): self.aborted = True

    def close(self): pass


async def test_play_all_frames():
    stream = FakeStream()
    frames = []
    done = await Speaker(lambda rate: stream).play(sine(1.0), on_frame=frames.append)
    assert done is True
    assert frames == list(range(25))
    assert stream.written == len(sine(1.0))


async def test_stop_ends_play_early():
    stream = FakeStream(delay=0.04)  # real-time pace: 1 s of audio takes ~1 s
    speaker = Speaker(lambda rate: stream)
    task = asyncio.create_task(speaker.play(sine(5.0)))
    await asyncio.sleep(0.2)
    t0 = time.monotonic()
    speaker.stop()
    assert await task is False
    assert time.monotonic() - t0 < 0.1
    assert stream.aborted
    assert stream.written < len(sine(5.0))


async def test_stop_aborts_from_the_playback_thread():
    # regression: abort() from another thread while write() blocked made CoreAudio's write never return
    class ThreadStream(FakeStream):
        def write(self, data):
            self.writer = threading.get_ident()
            super().write(data)

        def abort(self):
            self.aborted_from = threading.get_ident()
            super().abort()

    stream = ThreadStream(delay=0.04)
    speaker = Speaker(lambda rate: stream)
    task = asyncio.create_task(speaker.play(sine(5.0)))
    await asyncio.sleep(0.2)
    speaker.stop()
    assert await task is False
    assert stream.aborted_from == stream.writer
