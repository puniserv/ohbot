import asyncio
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .config import Config
from .events import EventBus
from .state import State
from .vision import Face, motor_target

HEADNOD, HEADTURN, EYETURN, LIDBLINK, TOPLIP, BOTTOMLIP, EYETILT = range(7)

_COLOUR_KEY = {
    State.IDLE: "idle",
    State.LISTENING: "listening",
    State.TRANSCRIBING: "thinking",
    State.THINKING: "thinking",
    State.SPEAKING: "speaking",
    State.ERROR: "error",
}
RETRY_S = 5.0
MANUAL_PAUSE_S = 5.0  # a blink or glance right after a manual move would undo what the operator is looking at
EYE_SPEED, HEAD_SPEED = 8, 3  # eyes lead and the head follows, as people do
MIN_STEP = 0.2  # smaller tracking moves are detector noise and would flood the serial link

GESTURE_FLASH = (10, 10, 0)
_NEUTRAL = ((HEADNOD, 5), (HEADTURN, 5), (EYETURN, 5), (EYETILT, 5), (LIDBLINK, 10))
_REST = dict(_NEUTRAL)
# Each step is held for [robot] gesture_step_s. Directions are unverified on hardware; tune positions here.
_GESTURES = {
    "yes": [[(HEADNOD, 7)], [(HEADNOD, 3)], [(HEADNOD, 7)], [(HEADNOD, 3)]],
    "no": [[(HEADTURN, 3)], [(HEADTURN, 7)], [(HEADTURN, 3)], [(HEADTURN, 7)]],
    "surprised": [[(LIDBLINK, 10), (HEADNOD, 7)], [], []],
    # Ohbot has no roll motor, so a "tilt" is a nod plus a turn
    "happy": [[(HEADNOD, 6), (HEADTURN, 6), "flash"], [(HEADTURN, 4)]],
    "thinking": [[(EYETILT, 8), (EYETURN, 7)], [], [], []],
}


class RobotDriver(Protocol):
    def connect(self) -> bool: ...
    def move(self, motor: int, pos: float, speed: int) -> None: ...  # pos 0..10
    def eye_colour(self, r: int, g: int, b: int) -> None: ...
    def close(self) -> None: ...


class NullDriver:
    """Records calls instead of talking to hardware; `available=False` simulates a missing robot."""

    def __init__(self, connected: bool = True) -> None:
        self.available = connected
        self.calls: list[tuple] = []
        self.fail: Exception | None = None
        self.closed = False

    def connect(self) -> bool:
        return self.available

    def _record(self, call: tuple) -> None:
        if self.fail:
            raise self.fail
        self.calls.append(call)

    def move(self, motor: int, pos: float, speed: int) -> None:
        self._record(("move", motor, pos, speed))

    def eye_colour(self, r: int, g: int, b: int) -> None:
        self._record(("eye", r, g, b))

    def close(self) -> None:
        self.closed = True


class OhbotDriver:
    """The `ohbot` library connects as a side effect of its first import and creates ohbotData/ in the CWD."""

    def __init__(self) -> None:
        self._o = None

    def connect(self) -> bool:
        try:
            if self._o is None:
                from ohbot import ohbot

                self._o = ohbot
            else:
                self._forget_port()
                self._o.init()
            return bool(self._o.connected)
        except Exception:
            return False

    def _forget_port(self) -> None:
        # init() only ever sets connected/port, so after an unplug a retry would "succeed" on the dead port
        try:
            self._o.ser.close()
        except Exception:
            pass
        self._o.connected, self._o.port = False, ""
        # _serwrite sets `writing` without try/finally, so a write that failed on unplug would block every later one
        self._o.writing = False

    @property
    def port(self) -> str:
        return getattr(self._o, "port", "")

    def move(self, motor: int, pos: float, speed: int) -> None:
        self._o.move(motor, pos, speed)

    def eye_colour(self, r: int, g: int, b: int) -> None:
        self._o.eyeColour(r, g, b)

    def close(self) -> None:
        if self._o is not None:
            self._o.close()


class Robot:
    """Every driver call runs on one worker thread (serial is not thread-safe); calls are dropped while disconnected."""

    def __init__(self, driver: RobotDriver, config: Config, bus: EventBus, rng: random.Random | None = None) -> None:
        self._driver, self._cfg, self._bus = driver, config, bus
        self._rng = rng or random.Random()
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="ohbot")
        self._state = State.IDLE
        self._connected: bool | None = None
        self._next_blink = self._next_glance = self._next_retry = 0.0
        self._recentre_at: float | None = None
        self.hold_idle = False  # set while the motion test panel is open
        self._idle_after = 0.0
        self._connecting = False
        self._closed = False
        self._seen_at = -math.inf
        self._pos: dict[int, float] = {}  # last tracking position sent per motor; any other move clears it
        self._tracked = False
        self._gesturing = 0

    @property
    def connected(self) -> bool:
        return bool(self._connected)

    def start(self) -> None:
        self._pool.submit(self._connect).result()

    def flush(self) -> None:
        self._pool.submit(lambda: None).result()

    def _set_connected(self, value: bool, error: str = "") -> None:
        if value == self._connected:
            return
        self._connected = value
        self._bus.publish("robot", connected=value, **({"error": error} if error else {}))

    def _connect(self) -> None:
        try:
            ok = self._driver.connect()
        except Exception as e:
            self._set_connected(False, str(e))
            return
        finally:
            self._connecting = False
        self._set_connected(ok)
        if ok:
            self._apply_colour()

    def _submit(self, fn, *args) -> None:
        if self._closed:  # delayed lip timers can outlive close()
            return
        self._pool.submit(self._guard, fn, *args)

    def _guard(self, fn, *args) -> None:
        if not self._connected:
            return
        try:
            fn(*args)
        except Exception as e:
            self._set_connected(False, str(e))

    def _colour(self) -> list[int]:
        return getattr(self._cfg.colours, _COLOUR_KEY[self._state])

    def _apply_colour(self) -> None:
        self._guard(self._driver.eye_colour, *self._colour())

    def set_state(self, state: State) -> None:
        self._state = state
        self._submit(self._driver.eye_colour, *self._colour())

    def lips(self, top: float, bottom: float) -> None:
        r = self._cfg.robot
        self._submit(self._driver.move, TOPLIP, 10 - top if r.lip_invert_top else top, 10)
        self._submit(self._driver.move, BOTTOMLIP, 10 - bottom if r.lip_invert_bottom else bottom, 10)

    def pause_idle(self) -> None:
        self._idle_after = time.monotonic() + MANUAL_PAUSE_S
        self._recentre_at = None
        self._pos.clear()

    def move(self, motor: int, pos: float) -> None:
        """Raw motor position for the test panel: no lip inversion."""
        self.pause_idle()
        self._submit(self._driver.move, motor, min(10, max(0, pos)), 10)

    def close_mouth(self) -> None:
        self.pause_idle()
        closed = self._cfg.robot.lip_closed
        self.lips(closed, closed)

    def neutral(self) -> None:
        self._submit(self._gesture_step, _NEUTRAL)
        self.close_mouth()

    def blink(self) -> None:
        def run() -> None:
            self._driver.move(LIDBLINK, 0, 10)
            time.sleep(0.12)
            self._driver.move(LIDBLINK, 10, 10)

        self._submit(run)

    def _gesture_step(self, step: list) -> None:
        r = self._cfg.robot
        for move in step:
            if move == "flash":
                self._driver.eye_colour(*GESTURE_FLASH)
            else:
                motor, pos = move
                rest = _REST[motor]
                pos = min(10, max(0, rest + (pos - rest) * r.gesture_scale))
                self._driver.move(motor, pos, r.gesture_speed)

    async def gesture(self, name: str | None) -> None:
        """Sleeps on the event loop, not the worker thread, so lip moves keep flowing; cancel it to cut it short."""
        steps = _GESTURES.get(name)
        if not steps:
            return
        self._gesturing += 1  # a count, because a manual gesture can start before the cancelled one's finally ran
        try:
            for step in steps:
                self._submit(self._gesture_step, step)
                await asyncio.sleep(self._cfg.robot.gesture_step_s)
        finally:
            self._gesturing -= 1
            self._pos.clear()
            self._submit(self._gesture_step, _NEUTRAL)
            self._submit(self._apply_colour)

    def glance(self, now: float) -> None:
        self._pos.clear()
        self._submit(self._driver.move, HEADTURN, self._rng.uniform(3, 7), 3)
        self._submit(self._driver.move, EYETURN, self._rng.uniform(3, 7), 5)
        self._recentre_at = now + 1.0

    def _send(self, motor: int, pos: float, speed: int, step: float) -> None:
        last = self._pos.get(motor)
        if last is not None and abs(pos - last) < step:
            return
        self._pos[motor] = pos
        self._submit(self._driver.move, motor, pos, speed)

    def _track(self, face: Face) -> None:
        c = self._cfg.camera
        head_step = max(c.deadband, MIN_STEP)
        self._recentre_at = None  # a glance's recentre would yank the head off the face
        self._send(EYETURN, motor_target(face.x, c.eye_gain), EYE_SPEED, MIN_STEP)
        self._send(EYETILT, motor_target(face.y, c.eye_gain), EYE_SPEED, MIN_STEP)
        self._send(HEADTURN, motor_target(face.x, c.head_gain), HEAD_SPEED, head_step)
        self._send(HEADNOD, motor_target(face.y, c.head_gain), HEAD_SPEED, head_step)
        self._tracked = True

    def tick(self, now: float, face: Face | None = None) -> None:
        if not self._connected:
            if now >= self._next_retry and not self._connecting:
                self._next_retry = now + RETRY_S
                self._connecting = True
                self._pool.submit(self._connect)
            return
        if face is not None:
            self._seen_at = now
        if self._recentre_at is not None and now >= self._recentre_at:
            self._recentre_at = None
            self._pos.clear()
            self._submit(self._driver.move, HEADTURN, 5, 3)
            self._submit(self._driver.move, EYETURN, 5, 5)
        if self.hold_idle or now < self._idle_after or self._gesturing:
            return
        lost = now - self._seen_at >= self._cfg.camera.lost_after_s
        if face is not None:
            self._track(face)
        elif lost and self._tracked:
            self._tracked = False
            self._pos.clear()
            for motor, speed in ((HEADTURN, HEAD_SPEED), (HEADNOD, HEAD_SPEED), (EYETURN, EYE_SPEED), (EYETILT, EYE_SPEED)):
                self._submit(self._driver.move, motor, 5, speed)
        if self._state != State.IDLE:
            return
        if now >= self._next_blink:
            self._next_blink = now + self._rng.uniform(3, 6)
            self.blink()
        if lost and now >= self._next_glance:
            self._next_glance = now + self._rng.uniform(5, 10)
            self.glance(now)

    async def idle_loop(self) -> None:
        while True:
            self.tick(time.monotonic())
            await asyncio.sleep(0.25)

    def close(self) -> None:
        self._closed = True

        def run() -> None:
            try:
                if self._connected:
                    self._driver.eye_colour(0, 0, 0)
                    closed = self._cfg.robot.lip_closed
                    self._driver.move(TOPLIP, closed, 10)
                    self._driver.move(BOTTOMLIP, closed, 10)
            except Exception:
                pass
            try:
                self._driver.close()
            except Exception:
                pass

        self._pool.submit(run)
        self._pool.shutdown(wait=True)
