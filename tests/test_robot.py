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
    pursue,
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


def test_blink_when_idle_or_listening_only():
    robot, driver, _ = make()
    robot.set_state(State.THINKING)
    robot.tick(1e9)
    robot.flush()
    assert LIDBLINK not in motors(driver)
    robot.set_state(State.IDLE)
    robot.tick(2e9)
    robot.flush()
    assert LIDBLINK in motors(driver)


def test_listening_blinks_but_does_not_glance():
    robot, driver, _ = make()
    robot.set_state(State.LISTENING)  # where wake and continuous modes rest
    robot.tick(1e9)
    robot.flush()
    assert LIDBLINK in motors(driver)
    assert robot._recentre_at is None and not robot._goal


def test_glance_glides_away_and_back():
    robot, driver, _ = make()
    robot.set_state(State.LISTENING)  # keep tick() from glancing again
    robot._rng = NS(uniform=lambda lo, hi: hi)
    robot.glance(now=0)
    for i in range(20):
        robot.tick(0.1 * i)
    robot.tick(10)
    for i in range(20):
        robot.tick(10 + 0.1 * i)
    robot.flush()
    head = [c[2] for c in driver.calls if c[:2] == ("move", HEADTURN)]
    assert len(head) > 4 and head[-1] == 5
    assert all(abs(b - a) <= 0.4 + 1e-9 for a, b in zip([5.0, *head], head))
    assert [c for c in driver.calls if c[:2] == ("move", EYETURN)][-1][2] == 5


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


# with gesture_step_s=0 each step is one frame; motors already in place are not sent, and the end eases back to rest
GESTURE_EXPECTED = {
    "yes": [(HEADNOD, 7), (HEADNOD, 3), (HEADNOD, 7), (HEADNOD, 3), (HEADNOD, 5)],
    "no": [(HEADTURN, 3), (HEADTURN, 7), (HEADTURN, 3), (HEADTURN, 7), (HEADTURN, 5)],
    "surprised": [(HEADNOD, 7), (HEADNOD, 5)],
    "happy": ["flash", (HEADNOD, 6), (HEADTURN, 6), (HEADTURN, 4), (HEADNOD, 5), (HEADTURN, 5)],
    "thinking": [(EYETILT, 8), (EYETURN, 7), (EYETURN, 5), (EYETILT, 5)],
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
    assert driver.calls == _gesture_calls(GESTURE_EXPECTED[name]) + [("eye", *cfg.colours.speaking)]


async def test_gesture_steps_are_eased_in_frames():
    robot, driver, _ = make(gesture_step_s=0.2)
    await robot.gesture("yes")
    robot.flush()
    nod = [c[2] for c in driver.calls if c[:2] == ("move", HEADNOD)]
    assert len(nod) == 5 * 4  # 5 steps (4 + back to rest), 0.2 s / 0.05 s frames
    assert nod[3] == 7 and nod[-1] == 5
    first = [b - a for a, b in zip([5.0, *nod[:4]], nod[:4])]
    assert first[0] < first[1] and first[3] < first[2]  # slow in, slow out


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
    first, back = [c for c in driver.calls if c[0] == "move"]
    assert first[:2] == ("move", HEADNOD) and 5 < first[2] < 5.01 and first[3] == GESTURE_SPEED
    assert back == ("move", HEADNOD, 5, cfg.robot.head_speed)  # cut short: glides back at the slow head speed
    assert driver.calls[-1] == ("eye", *cfg.colours.idle)


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
    for i in range(30):
        robot.tick(10 + 0.1 * i)
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


def test_pursue_closes_part_of_the_gap_within_min_and_max_step():
    assert pursue(5, 8, 0.5, 1.0) == 6  # half of 3 capped at 1
    assert pursue(5, 6, 0.3, 1.0) == 5.3
    assert pursue(5, 5.5, 0.1, 1.0) == 5.2  # never slower than MIN_STEP
    assert pursue(5, 4.9, 0.5, 1.0) == 4.9  # the last bit snaps onto the goal
    assert pursue(5, 2, 0.5, 0.4) == 4.6


def track(robot, x, y=0.0, t0=100.0, ticks=30):
    for i in range(ticks):
        robot.tick(t0 + 0.1 * i, face(x, y))
    robot.flush()


def positions(driver, motor):
    return [c[2] for c in driver.calls if c[:2] == ("move", motor)]


def test_head_pursues_the_face_capped_and_eyes_lead():
    robot, driver, cfg = make()
    robot.set_state(State.SPEAKING)  # tracking runs in every state; SPEAKING keeps blinks out of the way
    robot.tick(100, face(0.5, -0.25))
    robot.flush()
    first = {c[1]: c[2:] for c in moves(driver)}
    assert first[HEADTURN] == (5.4, cfg.robot.head_speed)  # 1.5 away, capped at head_max_step 0.4
    assert first[EYETURN][1] == robot_mod.EYE_SPEED and first[EYETURN][0] - 5 > 0.4  # eyes start faster
    track(robot, 0.5, -0.25, t0=100.1)
    head = positions(driver, HEADTURN)
    assert head[-1] == 6.5 and positions(driver, EYETURN)[-1] == 7.0
    assert positions(driver, HEADNOD)[-1] == 4.25 and positions(driver, EYETILT)[-1] == 4.0
    assert all(0 < b - a <= cfg.robot.head_max_step + 1e-9 for a, b in zip([5.0, *head], head))


def test_settled_head_ignores_jitter_inside_the_deadband_and_stops_sending():
    robot, driver, _ = make()
    robot.set_state(State.SPEAKING)
    track(robot, 0.5)
    driver.calls.clear()
    track(robot, 0.5, t0=110)
    assert moves(driver) == []  # settled: nothing more on the serial link
    track(robot, 0.6, t0=120)  # head +0.3 stays inside the 0.5 deadband, eyes +0.4 follow
    assert positions(driver, HEADTURN) == [] and positions(driver, EYETURN)[-1] == 7.4
    track(robot, 0.75, t0=130)  # head target 7.25: off by 0.75, so it glides all the way there
    assert positions(driver, HEADTURN)[-1] == 7.25


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
    track(robot, 0.5, t0=100.3)
    # the gesture's neutral pose moved the motors, so tracking pursues the face again
    assert positions(driver, HEADTURN)[-1] == 6.5 and positions(driver, EYETURN)[-1] == 7.0


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
    track(robot, 0.5, t0=now + robot_mod.MANUAL_PAUSE_S + 1, ticks=40)
    assert positions(driver, HEADTURN)[0] == 2.4 and positions(driver, HEADTURN)[-1] == 6.5


def test_idle_tracking_keeps_blinking_but_does_not_glance_while_a_face_was_seen_recently():
    robot, driver, cfg = make()
    robot.tick(100, face(0.5))
    robot.tick(100 + cfg.camera.lost_after_s - 0.5, None)  # a missed detection, not yet lost
    robot.flush()
    assert positions(driver, EYETURN) and LIDBLINK in motors(driver)
    assert robot._goal[EYETURN] == 7.0  # a glance would have moved the goal


def test_lost_face_recentres_once_then_glances_resume():
    robot, driver, cfg = make()
    robot.set_state(State.LISTENING)  # no glance yet
    track(robot, 0.5, 0.5)
    lost = 100 + 3 + cfg.camera.lost_after_s
    for i in range(30):
        robot.tick(lost + 0.1 * i, None)
    robot.flush()
    assert [positions(driver, m)[-1] for m in (HEADTURN, HEADNOD, EYETURN, EYETILT)] == [5, 5, 5, 5]
    robot.set_state(State.IDLE)
    robot.tick(lost + 10, None)
    assert robot._recentre_at is not None  # then a glance


def test_sleep_parks_the_robot_frees_the_port_and_start_reconnects(monkeypatch):
    monkeypatch.setattr(robot_mod, "SETTLE_S", 0)
    robot, driver, cfg = make()
    robot.flush()
    driver.calls.clear()
    robot.sleep()
    assert driver.calls[-3:] == [("move", TOPLIP, 1.5, 10), ("move", BOTTOMLIP, 1.5, 10), ("eye", 0, 0, 0)]
    assert NEUTRAL_CALLS[0] in driver.calls and driver.closed and not robot.connected
    driver.calls.clear()
    robot.lips(5, 5)
    robot.flush()
    assert driver.calls == []
    robot.start()
    assert robot.connected and driver.calls == [("eye", *cfg.colours.idle)]


def test_ohbot_driver_close_releases_the_serial_port(monkeypatch):
    port = NS(closed=False)
    port.close = lambda: setattr(port, "closed", True)
    fake = NS(connected=True, port="/dev/x", writing=False, ser=port, init=lambda: None, close=lambda: None)
    monkeypatch.setitem(sys.modules, "ohbot", NS(ohbot=fake))
    d = OhbotDriver()
    d.connect()
    d.close()
    assert port.closed and fake.connected is False  # sleep must leave the port free for another program
