import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .config import Config
from .events import EventBus
from .state import State

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
        self._connecting = False

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

    def blink(self) -> None:
        def run() -> None:
            self._driver.move(LIDBLINK, 0, 10)
            time.sleep(0.12)
            self._driver.move(LIDBLINK, 10, 10)

        self._submit(run)

    def glance(self, now: float) -> None:
        self._submit(self._driver.move, HEADTURN, self._rng.uniform(3, 7), 3)
        self._submit(self._driver.move, EYETURN, self._rng.uniform(3, 7), 5)
        self._recentre_at = now + 1.0

    def tick(self, now: float) -> None:
        if not self._connected:
            if now >= self._next_retry and not self._connecting:
                self._next_retry = now + RETRY_S
                self._connecting = True
                self._pool.submit(self._connect)
            return
        if self._recentre_at is not None and now >= self._recentre_at:
            self._recentre_at = None
            self._submit(self._driver.move, HEADTURN, 5, 3)
            self._submit(self._driver.move, EYETURN, 5, 5)
        if self._state != State.IDLE:
            return
        if now >= self._next_blink:
            self._next_blink = now + self._rng.uniform(3, 6)
            self.blink()
        if now >= self._next_glance:
            self._next_glance = now + self._rng.uniform(5, 10)
            self.glance(now)

    async def idle_loop(self) -> None:
        while True:
            self.tick(time.monotonic())
            await asyncio.sleep(0.25)

    def close(self) -> None:
        def run() -> None:
            try:
                if self._connected:
                    self._driver.eye_colour(0, 0, 0)
                    self._driver.move(TOPLIP, 5, 10)
                    self._driver.move(BOTTOMLIP, 5, 10)
            except Exception:
                pass
            try:
                self._driver.close()
            except Exception:
                pass

        self._pool.submit(run)
        self._pool.shutdown(wait=True)
