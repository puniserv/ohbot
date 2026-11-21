import sys
import threading
from types import SimpleNamespace as NS

from ohbotchat.config import load_config
from ohbotchat.events import EventBus
from ohbotchat.robot import BOTTOMLIP, EYETURN, HEADTURN, LIDBLINK, TOPLIP, NullDriver, OhbotDriver, Robot
from ohbotchat.state import State


def make(driver=None, **robot_cfg):
    cfg = load_config(env={})
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
    robot = Robot(driver, load_config(env={}), bus)
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
    assert driver.calls[-3:] == [("eye", 0, 0, 0), ("move", TOPLIP, 5, 10), ("move", BOTTOMLIP, 5, 10)]
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
