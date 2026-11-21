import asyncio
import sys
import threading
import time
from types import SimpleNamespace as NS

import pytest

from ohbotchat import robot as robot_mod
from ohbotchat.config import Config
from ohbotchat.events import EventBus
from ohbotchat.robot import (
    BOTTOMLIP,
    EYETILT,
    EYETURN,
    GESTURE_FLASH,
    HEADNOD,
    HEADTURN,
    LIDBLINK,
    TOPLIP,
    NullDriver,
    OhbotDriver,
    Robot,
)
from ohbotchat.state import State
from ohbotchat.vision import Face


def make(driver=None, **robot_cfg):
    cfg = Config()  # code defaults: tuning config.toml on the robot must not break these tests
    cfg.camera.eye_gain = 4.0  # the expected eye positions in the tracking tests are worked out with this gain
    for k, v in robot_cfg.items():
        setattr(cfg.robot, k, v)
    driver = driver or NullDriver()
    bus = EventBus()
    robot = Robot(driver, cfg, bus)
    robot.start()
    return robot, driver, cfg


def motors(driver):
    return [c[1] for c in driver.calls if c[0] == "move"]


def test_state_colours_from_config():
    robot, driver, cfg = make()
    robot.set_state(State.LISTENING)
    robot.set_state(State.THINKING)
    robot.set_state(State.ERROR)
    robot.flush()
    assert [c[1:] for c in driver.calls if c[0] == "eye"][1:] == [  # first call is the idle colour on connect
        tuple(cfg.colours.listening),
        tuple(cfg.colours.thinking),
        tuple(cfg.colours.error),
    ]


def test_blink_only_when_idle():
    robot, driver, _ = make()
    robot.set_state(State.LISTENING)
    robot.tick(1e9)
    robot.flush()
    assert LIDBLINK not in motors(driver)
    robot.set_state(State.IDLE)
    robot.tick(2e9)
    robot.flush()
    assert LIDBLINK in motors(driver)


def test_glance_recentres_later():
    robot, driver, _ = make()
    robot.set_state(State.LISTENING)  # keep tick() from glancing again
    robot.glance(now=0)
    robot.tick(10)
    robot.flush()
    moves = [c for c in driver.calls if c[0] == "move" and c[1] in (HEADTURN, EYETURN)]
    assert moves[-2:] == [("move", HEADTURN, 5, 3), ("move", EYETURN, 5, 5)]


def test_lips_mapped_and_inverted():
    robot, driver, _ = make()
    robot.lips(3.0, 7.0)
    robot.flush()
    assert ("move", TOPLIP, 3.0, 10) in driver.calls and ("move", BOTTOMLIP, 7.0, 10) in driver.calls
    robot2, driver2, _ = make(lip_invert_top=True)
    robot2.lips(3.0, 7.0)
    robot2.flush()
    assert ("move", TOPLIP, 7.0, 10) in driver2.calls and ("move", BOTTOMLIP, 7.0, 10) in driver2.calls


async def test_driver_error_disconnects_and_publishes():
    bus = EventBus()
    q = bus.subscribe()
    driver = NullDriver()
    robot = Robot(driver, Config(), bus)
    robot.start()
    assert (await q.get()).data == {"connected": True}
    driver.fail = OSError("unplugged")
    robot.lips(5, 5)
    robot.flush()
    ev = await q.get()
    assert ev.type == "robot" and ev.data["connected"] is False
    driver.fail = None
    robot.lips(5, 5)
    robot.flush()
    assert not [c for c in driver.calls if c[0] == "move"]  # dropped while disconnected


def test_no_robot_is_harmless_and_reconnects():
    driver = NullDriver(connected=False)
    robot, _, _ = make(driver)
    robot.set_state(State.SPEAKING)
    robot.lips(4, 6)
    robot.flush()
    assert driver.calls == []
    driver.available = True
    robot.tick(1e9)  # retry is due
    robot.flush()
    robot.lips(4, 6)
    robot.flush()
    assert motors(driver)


def test_close_closes_driver():
    robot, driver, _ = make()
    robot.close()
    assert driver.closed


def test_close_switches_eyes_off_and_closes_lips_first():
    robot, driver, _ = make()
    robot.close()
    assert driver.calls[-3:] == [("eye", 0, 0, 0), ("move", TOPLIP, 1.5, 10), ("move", BOTTOMLIP, 1.5, 10)]
    assert driver.closed


def test_tick_skips_connect_while_one_is_in_flight():
    gate = threading.Event()

    class Slow(NullDriver):
        n = 0

        def connect(self):
            Slow.n += 1
            if Slow.n > 1:
                gate.wait(2)
            return super().connect()

    driver = Slow(connected=False)
    robot, _, _ = make(driver)
    robot.tick(100)
    robot.tick(200)
    gate.set()
    robot.flush()
    assert Slow.n == 2


def test_ohbot_driver_reconnect_after_unplug_reports_disconnected(monkeypatch):
    class FakeOhbot:
        connected, port, plugged = True, "/dev/x", True
        ser = NS(close=lambda: None)

        def init(self):
            # like the real library: only ever sets connected/port, never clears them
            if self.plugged:
                self.connected, self.port = True, "/dev/x"

    fake = FakeOhbot()
    monkeypatch.setitem(sys.modules, "ohbot", NS(ohbot=fake))
    d = OhbotDriver()
    assert d.connect() is True
    fake.plugged = False
    assert d.connect() is False
    fake.plugged = True
    assert d.connect() is True


def test_calls_after_close_are_dropped_silently():
    # regression guard: lip timers firing after close() hit the shut-down pool and raised RuntimeError
    robot, driver, _ = make()
    robot.close()
    robot.lips(5, 5)
    robot.set_state(State.LISTENING)


GESTURE_EXPECTED = {
    "yes": [(HEADNOD, 7), (HEADNOD, 3), (HEADNOD, 7), (HEADNOD, 3)],
    "no": [(HEADTURN, 3), (HEADTURN, 7), (HEADTURN, 3), (HEADTURN, 7)],
    "surprised": [(LIDBLINK, 10), (HEADNOD, 7)],
    "happy": [(HEADNOD, 6), (HEADTURN, 6), "flash", (HEADTURN, 4)],
    "thinking": [(EYETILT, 8), (EYETURN, 7)],
}


GESTURE_SPEED = 8


def _gesture_calls(steps):
    return [("eye", *GESTURE_FLASH) if s == "flash" else ("move", *s, GESTURE_SPEED) for s in steps]


NEUTRAL_CALLS = [("move", m, p, GESTURE_SPEED) for m, p in ((HEADNOD, 5), (HEADTURN, 5), (EYETURN, 5), (EYETILT, 5), (LIDBLINK, 10))]


@pytest.mark.parametrize("name", list(GESTURE_EXPECTED))
async def test_gesture_driver_sequence_ends_in_neutral(name, monkeypatch):
    robot, driver, cfg = make(gesture_step_s=0)
    robot.set_state(State.SPEAKING)
    robot.flush()
    driver.calls.clear()
    await robot.gesture(name)
    robot.flush()
    assert driver.calls == _gesture_calls(GESTURE_EXPECTED[name]) + NEUTRAL_CALLS + [("eye", *cfg.colours.speaking)]


@pytest.mark.parametrize("name", [None, "none", "dance"])
async def test_unknown_gesture_does_nothing(name):
    robot, driver, _ = make()
    robot.flush()
    driver.calls.clear()
    await robot.gesture(name)
    robot.flush()
    assert driver.calls == []


async def test_cancelled_gesture_stops_and_returns_to_neutral():
    robot, driver, cfg = make(gesture_step_s=60)
    robot.flush()
    driver.calls.clear()
    task = asyncio.create_task(robot.gesture("yes"))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    robot.flush()
    assert driver.calls == [("move", HEADNOD, 7, GESTURE_SPEED)] + NEUTRAL_CALLS + [("eye", *cfg.colours.idle)]


async def test_gesture_scale_shrinks_moves_around_rest_and_clamps():
    robot, driver, _ = make(gesture_step_s=0, gesture_scale=0.5)
    robot.flush()
    driver.calls.clear()
    await robot.gesture("yes")
    robot.flush()
    assert driver.calls[:2] == [("move", HEADNOD, 6, GESTURE_SPEED), ("move", HEADNOD, 4, GESTURE_SPEED)]
    robot, driver, _ = make(gesture_step_s=0, gesture_scale=4)
    robot.flush()
    driver.calls.clear()
    await robot.gesture("yes")
    robot.flush()
    assert driver.calls[:2] == [("move", HEADNOD, 10, GESTURE_SPEED), ("move", HEADNOD, 0, GESTURE_SPEED)]


def test_ohbot_driver_reconnect_clears_a_write_stuck_by_unplug(monkeypatch):
    # regression guard: ohbot._serwrite sets `writing` without try/finally, so a write that failed on unplug
    # left it True and the next write after replug spun forever on the worker thread
    fake = NS(connected=True, port="/dev/x", writing=False, ser=NS(close=lambda: None), init=lambda: None)
    monkeypatch.setitem(sys.modules, "ohbot", NS(ohbot=fake))
    d = OhbotDriver()
    d.connect()
    fake.writing = True
    d.connect()
    assert fake.writing is False


def test_manual_move_is_clamped_and_pauses_idle_moves():
    robot, driver, _ = make()
    robot.move(HEADTURN, 12)
    robot.move(EYETILT, -3)
    robot.flush()
    assert driver.calls[-2:] == [("move", HEADTURN, 10, 10), ("move", EYETILT, 0, 10)]
    driver.calls.clear()
    robot.tick(time.monotonic())
    robot.flush()
    assert driver.calls == []  # no blink or glance right after the operator moved a motor
    robot.tick(time.monotonic() + robot_mod.MANUAL_PAUSE_S + 1)
    robot.flush()
    assert LIDBLINK in motors(driver)


def test_hold_idle_pauses_blink_and_glance():
    robot, driver, _ = make()
    robot.hold_idle = True
    robot.tick(1e9)
    robot.flush()
    assert not [c for c in driver.calls if c[0] == "move"]


def test_manual_move_cancels_a_pending_glance_recentre():
    robot, driver, _ = make()
    robot.glance(now=0)
    robot.move(HEADTURN, 2)
    robot.hold_idle = True
    robot.tick(10)
    robot.flush()
    assert [c for c in driver.calls if c[:2] == ("move", HEADTURN)][-1] == ("move", HEADTURN, 2, 10)


def test_close_mouth_and_neutral():
    robot, driver, cfg = make(lip_invert_bottom=True)
    robot.close_mouth()
    robot.flush()
    assert driver.calls[-2:] == [("move", TOPLIP, 1.5, 10), ("move", BOTTOMLIP, 8.5, 10)]
    driver.calls.clear()
    robot.neutral()
    robot.flush()
    assert driver.calls == NEUTRAL_CALLS + [("move", TOPLIP, 1.5, 10), ("move", BOTTOMLIP, 8.5, 10)]


def face(x=0.0, y=0.0):
    return Face(x, y, 0.0)


def moves(driver):
    return [c for c in driver.calls if c[0] == "move" and c[1] != LIDBLINK]


def test_eyes_and_head_turn_to_the_face_eyes_faster():
    robot, driver, _ = make()
    robot.set_state(State.SPEAKING)  # tracking runs in every state; SPEAKING keeps blinks out of the way
    robot.tick(100, face(0.5, -0.25))
    robot.flush()
    assert moves(driver) == [
        ("move", EYETURN, 7.0, 8),
        ("move", EYETILT, 4.0, 8),
        ("move", HEADTURN, 6.5, 3),
        ("move", HEADNOD, 4.25, 3),
    ]


def test_small_changes_are_not_sent_and_the_head_waits_for_the_deadband():
    robot, driver, _ = make()
    robot.set_state(State.SPEAKING)
    robot.tick(100, face(0.5))
    robot.flush()
    driver.calls.clear()
    robot.tick(100.1, face(0.53))  # eyes +0.12, head +0.09: both under 0.2
    robot.tick(100.2, face(0.625))  # eyes +0.5 move; head +0.375 stays inside the 0.5 deadband
    robot.flush()
    assert moves(driver) == [("move", EYETURN, 7.5, 8)]
    robot.tick(100.3, face(0.75))
    robot.flush()
    assert moves(driver)[-2:] == [("move", EYETURN, 8.0, 8), ("move", HEADTURN, 7.25, 3)]


async def test_a_running_gesture_owns_the_head_then_tracking_resends():
    robot, driver, _ = make(gesture_step_s=60)
    robot.set_state(State.SPEAKING)
    robot.tick(100, face(0.5))
    task = asyncio.create_task(robot.gesture("yes"))
    await asyncio.sleep(0)
    robot.flush()
    driver.calls.clear()
    robot.tick(100.1, face(0.75))
    robot.flush()
    assert moves(driver) == []
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    robot.tick(100.2, face(0.5))
    robot.flush()
    # unchanged targets are sent again, because the gesture's neutral pose moved the motors
    assert moves(driver)[-4:] == [
        ("move", EYETURN, 7.0, 8),
        ("move", EYETILT, 5.0, 8),
        ("move", HEADTURN, 6.5, 3),
        ("move", HEADNOD, 5.0, 3),
    ]


def test_test_panel_and_manual_moves_pause_tracking():
    robot, driver, _ = make()
    robot.set_state(State.SPEAKING)
    robot.hold_idle = True
    robot.tick(100, face(0.5))
    robot.flush()
    assert moves(driver) == []
    robot.hold_idle = False
    robot.move(HEADTURN, 2)
    robot.flush()
    driver.calls.clear()
    now = time.monotonic()
    robot.tick(now, face(0.5))
    robot.flush()
    assert moves(driver) == []
    robot.tick(now + robot_mod.MANUAL_PAUSE_S + 1, face(0.5))
    robot.flush()
    assert ("move", HEADTURN, 6.5, 3) in driver.calls


def test_idle_tracking_keeps_blinking_but_does_not_glance_while_a_face_was_seen_recently():
    robot, driver, cfg = make()
    robot.tick(100, face(0.5))
    robot.tick(100 + cfg.camera.lost_after_s - 0.5, None)  # a missed detection, not yet lost
    robot.flush()
    assert ("move", EYETURN, 7.0, 8) in driver.calls and LIDBLINK in motors(driver)
    assert not [c for c in moves(driver) if c[1] == EYETURN and c[3] == 5]  # a glance moves the eyes at speed 5


def test_lost_face_recentres_once_then_glances_resume():
    robot, driver, cfg = make()
    robot.tick(100, face(0.5))
    robot.flush()
    driver.calls.clear()
    lost = 100 + cfg.camera.lost_after_s
    robot.tick(lost, None)
    robot.tick(lost + 0.1, None)
    robot.flush()
    m = moves(driver)
    assert m[:4] == [("move", HEADTURN, 5, 3), ("move", HEADNOD, 5, 3), ("move", EYETURN, 5, 8), ("move", EYETILT, 5, 8)]
    assert len(m) == 6 and m[5][1] == EYETURN and m[5][3] == 5  # then one glance
