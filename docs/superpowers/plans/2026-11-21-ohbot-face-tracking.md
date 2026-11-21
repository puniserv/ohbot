# Ohbot Face Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ohbot turns its eyes and head toward the largest face a camera sees, and greets a child who walks up.

**Architecture:** A new `ohbotchat/vision.py` runs a daemon thread. The thread reads camera frames (~10 fps), runs a detector callable `detect(frame) -> list[(x, y, w, h, score)]` (YuNet by default), and keeps the latest `Face` in `Vision.face`. It publishes a throttled `face` event for the UI dot. One App loop on the event loop ticks every 0.1 s. Each tick passes `vision.face` to `Robot.tick(now, face)`, which moves eyes and head through the existing single-worker executor, and to `App._auto_greet(now, face)`, which plays the existing greeting. Frames never leave the vision thread.

**Tech Stack:** Python 3.11, uv, `opencv-python-headless` (`cv2.FaceDetectorYN` + YuNet `face_detection_yunet_2023mar.onnx`), numpy, pytest + pytest-asyncio. Tests use a fake camera and a fake detector; no test opens a camera or the serial port.

Spec: `docs/superpowers/specs/2026-11-21-ohbot-face-tracking-design.md`. Branch `feat/ohbot-chat`, local only, never push.

**Tick rate:** the robot loop runs at 0.25 s today. Eyes that follow a face at 4 updates per second lag behind the camera's 10 fps, so the loop moves to `TICK_S = 0.1` (Task 5). Blink, glance and reconnect timing use timestamps and do not change. The ≥ 0.2 change threshold and the head deadband keep the serial link quiet.

---

## File map

```
ohbotchat/vision.py          NEW  Face, largest(), position(), motor_target(), load_detector(), Vision thread
ohbotchat/models/face_detection_yunet_2023mar.onnx  NEW  YuNet model (OpenCV Zoo, MIT, ~230 KB)
ohbotchat/config.py          CameraConfig, Config.camera, EDITABLE["camera"]
ohbotchat/robot.py           tick(now, face): tracking, gesture precedence, glance only without a face; idle_loop removed
ohbotchat/app.py             vision wiring, _tick_loop every TICK_S = 0.1 s, _auto_greet
ohbotchat/web.py             snapshot "camera"; web doctor reports the running camera instead of opening it
ohbotchat/events.py          type comment gains "face" | "camera"
ohbotchat/doctor.py          "Kamera" check (model loads, camera returns a frame)
web/index.html app.js style.css   "Kamera" group in Ustawienia, face dot box
config.toml                  [camera] with short comments
pyproject.toml uv.lock       opencv-python-headless
packaging/ohbotchat.spec     datas: ohbotchat/models
docs/windows.md              camera section: privacy, permissions, calibration, troubleshooting
tests/test_core.py test_vision.py (NEW) test_robot.py test_app.py test_web.py test_doctor.py
```

## Contracts (all tasks use these names)

```python
# vision.py
Box = tuple[float, float, float, float, float]       # x, y, w, h, score in pixels of the frame given to detect()
Detect = Callable[[np.ndarray], list[Box]]
@dataclass(frozen=True)
class Face: x: float; y: float; size: float; t: float  # x, y in -1..1 (y up), after mirror + offset
def largest(boxes: list[Box], min_score: float) -> Box | None
def position(box: Box, width: int, height: int, cam: CameraConfig) -> tuple[float, float]
def motor_target(v: float, gain: float) -> float      # clamp(5 + v * gain, 0, 10)
def load_detector() -> Detect                         # lazy cv2 import
class Vision:
    def __init__(self, config, bus, open_camera=_open_camera, make_detector=load_detector)
    face: Face | None; ok: bool | None
    def start(self) -> None; def stop(self) -> None; def step(self, now: float) -> None

# robot.py
def tick(self, now: float, face: Face | None = None) -> None

# app.py
App(..., vision=None); App._auto_greet(now: float, face: Face | None) -> None

# doctor.py
run_doctor(..., camera: Callable[[Config], str | None] = _default_camera)   # None = OK, else what failed

# bus events
"face"   {"visible": True, "x": float, "y": float} | {"visible": False}   (≤ 5/s)
"camera" {"ok": bool | None}
"error"  {"kind": "other", "message": "Kamera niedostępna"}                (once per failure)
```

---

### Task 1: `[camera]` config section and validation

**Files:**
- Modify: `ohbotchat/config.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing tests**

Add after `test_validate_settings_accepts_known_keys_and_normalises` in `tests/test_core.py`:

```python
def test_camera_settings_are_editable_and_normalised():
    from ohbotchat.config import editable_settings, validate_settings

    values = editable_settings(load_config(env={}))["camera"]
    assert values == {
        "enabled": True, "index": 0, "mirror": False, "offset_x": 0.0, "offset_y": 0.0, "head_gain": 3.0,
        "eye_gain": 4.0, "deadband": 0.5, "smoothing": 0.5, "min_score": 0.7, "lost_after_s": 3.0,
        "auto_greet": True, "greet_cooldown_s": 60.0,
    }
    out = validate_settings({"camera": {"index": 2, "mirror": True, "offset_x": -1, "greet_cooldown_s": 120}})
    assert out == {"camera": {"index": 2, "mirror": True, "offset_x": -1.0, "greet_cooldown_s": 120.0}}
```

Add these cases to the `bad` list of `test_validate_settings_rejects` (before `{"robot": [1]}`):

```python
        {"camera": {"index": 10}},
        {"camera": {"index": 1.5}},
        {"camera": {"offset_x": 1.1}},
        {"camera": {"smoothing": 0.96}},
        {"camera": {"min_score": 0.2}},
        {"camera": {"lost_after_s": 0.4}},
        {"camera": {"greet_cooldown_s": 5}},
        {"camera": {"enabled": "yes"}},
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_core.py -q`
Expected: `test_camera_settings_are_editable_and_normalised` FAILS with `KeyError: 'camera'`. The new reject cases already pass (unknown section) and stay as guards.

- [ ] **Step 3: Implement**

In `ohbotchat/config.py`, add after `RobotConfig`:

```python
@dataclass
class CameraConfig:
    enabled: bool = True
    index: int = 0
    mirror: bool = False  # flip when the robot looks the wrong way
    offset_x: float = 0.0  # the camera sits beside the robot, not on its axis
    offset_y: float = 0.0
    head_gain: float = 3.0
    eye_gain: float = 4.0
    deadband: float = 0.5
    smoothing: float = 0.5
    min_score: float = 0.7
    lost_after_s: float = 3.0
    auto_greet: bool = True
    greet_cooldown_s: float = 60.0
```

In `Config`, after `robot: RobotConfig = ...`:

```python
    camera: CameraConfig = field(default_factory=CameraConfig)
```

In `EDITABLE`, after the `"conversation"` entry:

```python
    "camera": {
        "enabled": _flag,
        "index": _number(int, 0, 9),
        "mirror": _flag,
        "offset_x": _number(float, -1, 1),
        "offset_y": _number(float, -1, 1),
        "head_gain": _number(float, 0, 5),
        "eye_gain": _number(float, 0, 5),
        "deadband": _number(float, 0, 3),
        "smoothing": _number(float, 0, 0.95),
        "min_score": _number(float, 0.3, 0.99),
        "lost_after_s": _number(float, 0.5, 30),
        "auto_greet": _flag,
        "greet_cooldown_s": _number(float, 10, 3600),
    },
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ohbotchat/config.py tests/test_core.py
git commit -m "feat(config): camera settings with validation"
```

---

### Task 2: Mapping pure functions

**Files:**
- Create: `ohbotchat/vision.py`
- Create: `tests/test_vision.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vision.py`:

```python
from ohbotchat.config import CameraConfig
from ohbotchat.vision import largest, motor_target, position

W, H = 320, 240


def test_largest_face_above_min_score_wins():
    small, big, unsure = (0, 0, 20, 20, 0.9), (100, 50, 60, 60, 0.8), (0, 0, 200, 200, 0.5)
    assert largest([small, big, unsure], 0.7) == big
    assert largest([unsure], 0.7) is None
    assert largest([], 0.7) is None


def test_position_is_the_centre_in_minus_one_to_one_with_y_up():
    cam = CameraConfig()
    assert position((150, 110, 20, 20, 1), W, H, cam) == (0.0, 0.0)
    assert position((0, 0, 0, 0, 1), W, H, cam) == (-1.0, 1.0)  # top-left corner
    assert position((W, H, 0, 0, 1), W, H, cam) == (1.0, -1.0)


def test_mirror_flips_x_then_offset_shifts_and_clamps():
    box = (240, 110, 0, 20, 1)  # centre at 3/4 of the width: x = 0.5
    assert position(box, W, H, CameraConfig(mirror=True)) == (-0.5, 0.0)
    assert position(box, W, H, CameraConfig(offset_x=-0.25, offset_y=0.5)) == (0.25, 0.5)
    assert position(box, W, H, CameraConfig(offset_x=0.75)) == (1.0, 0.0)


def test_motor_target_applies_gain_around_centre_and_clamps():
    assert motor_target(0, 3) == 5
    assert motor_target(0.5, 4) == 7
    assert motor_target(-1, 3) == 2
    assert motor_target(1, 5) == 10
    assert motor_target(-1, 5) == 0
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_vision.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'ohbotchat.vision'`.

- [ ] **Step 3: Implement**

Create `ohbotchat/vision.py`:

```python
from dataclasses import dataclass

from .config import CameraConfig

Box = tuple[float, float, float, float, float]  # x, y, w, h, score in pixels of the frame given to detect()


@dataclass(frozen=True)
class Face:
    x: float  # -1..1 after mirror and offset
    y: float  # -1..1, up is positive
    size: float  # face width / frame width
    t: float  # time.monotonic() of the frame


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


def largest(boxes: list[Box], min_score: float) -> Box | None:
    return max((b for b in boxes if b[4] >= min_score), key=lambda b: b[2] * b[3], default=None)


def position(box: Box, width: int, height: int, cam: CameraConfig) -> tuple[float, float]:
    """Face centre in -1..1 after `mirror` and the off-axis `offset_x/offset_y` calibration."""
    x, y, w, h, _ = box
    nx = (x + w / 2) / width * 2 - 1
    # which way the motors turn for +y is unverified on hardware; flip this sign if the head nods the wrong way
    ny = 1 - (y + h / 2) / height * 2
    if cam.mirror:
        nx = -nx
    return _clamp(nx + cam.offset_x, -1, 1), _clamp(ny + cam.offset_y, -1, 1)


def motor_target(v: float, gain: float) -> float:
    return _clamp(5 + v * gain, 0, 10)
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest tests/test_vision.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add ohbotchat/vision.py tests/test_vision.py
git commit -m "feat(vision): map a face box to a normalised position and motor targets"
```

---

### Task 3: Vision thread with a fake camera and detector

**Files:**
- Modify: `ohbotchat/vision.py`
- Test: `tests/test_vision.py`

- [ ] **Step 1: Write the failing tests**

Replace the imports at the top of `tests/test_vision.py` with:

```python
import time
from types import SimpleNamespace as NS

import numpy as np

from ohbotchat.config import CameraConfig, load_config
from ohbotchat.vision import RETRY_S, UNAVAILABLE, Face, Vision, largest, motor_target, position
```

Append:

```python
FRAME = np.zeros((H, W, 3), np.uint8)
MIDDLE = (140, 100, 40, 40, 0.9)  # centre (160, 120): x = 0, y = 0
RIGHT = (220, 100, 40, 40, 0.9)  # centre x = 240: x = 0.5


class FakeCamera:
    def __init__(self, index, opened):
        self.index, self.opened, self.frames_ok, self.released = index, opened, True, False

    def isOpened(self):
        return self.opened

    def read(self):
        return (True, FRAME) if self.frames_ok else (False, None)

    def release(self):
        self.released = True


class Bus:
    def __init__(self):
        self.events = []

    def publish(self, type, **data):
        self.events.append((type, data))


def rig(**camera):
    """Vision over a fake camera opener and a detector that returns whatever `r.boxes` holds."""
    cfg = load_config(env={})
    for k, v in camera.items():
        setattr(cfg.camera, k, v)
    r = NS(cfg=cfg, bus=Bus(), cams=[], boxes=[], opened=True, error=None)

    def open_camera(index):
        r.cams.append(FakeCamera(index, r.opened))
        return r.cams[-1]

    def detect(frame):
        if r.error:
            raise r.error
        return list(r.boxes)

    r.vision = Vision(cfg, r.bus, open_camera=open_camera, make_detector=lambda: detect)
    return r


def faces(r):
    return [d for t, d in r.bus.events if t == "face"]


def test_largest_face_is_kept_and_published():
    r = rig()
    r.boxes[:] = [(0, 0, 20, 20, 0.9), RIGHT]
    r.vision.step(10.0)
    assert r.vision.face == Face(0.5, 0.0, 40 / W, 10.0)
    assert r.cams[0].index == 0 and r.vision.ok is True
    assert r.bus.events == [("camera", {"ok": True}), ("face", {"visible": True, "x": 0.5, "y": 0.0})]


def test_no_face_clears_it_and_greys_the_dot_once():
    r = rig()
    r.boxes[:] = [RIGHT]
    r.vision.step(0)
    r.boxes.clear()
    r.vision.step(1)
    r.vision.step(2)
    assert r.vision.face is None
    assert faces(r)[1:] == [{"visible": False}]


def test_min_score_is_read_live():
    r = rig()
    r.boxes[:] = [(*RIGHT[:4], 0.75)]
    r.vision.step(0)
    assert r.vision.face is not None
    r.cfg.camera.min_score = 0.8
    r.vision.step(1)
    assert r.vision.face is None


def test_smoothing_blends_with_the_previous_position_and_resets_when_the_face_is_lost():
    r = rig(smoothing=0.5)
    r.boxes[:] = [MIDDLE]
    r.vision.step(0)
    r.boxes[:] = [RIGHT]
    r.vision.step(1)
    assert r.vision.face.x == 0.25
    r.boxes.clear()
    r.vision.step(2)
    r.boxes[:] = [RIGHT]
    r.vision.step(3)
    assert r.vision.face.x == 0.5


def test_face_events_are_throttled_to_five_per_second():
    r = rig()
    r.boxes[:] = [RIGHT]
    for now in (0, 0.1, 0.2, 0.3):
        r.vision.step(now)
    assert len(faces(r)) == 2


def test_camera_that_does_not_open_toasts_once_and_retries_every_5_s():
    r = rig()
    r.opened = False
    r.vision.step(0)
    r.vision.step(1)
    assert len(r.cams) == 1 and r.cams[0].released and r.vision.ok is False
    assert r.bus.events == [("error", {"kind": "other", "message": UNAVAILABLE}), ("camera", {"ok": False})]
    r.vision.step(RETRY_S)
    assert len(r.cams) == 2 and len(r.bus.events) == 2  # still broken: no second toast
    r.opened = True
    r.vision.step(2 * RETRY_S)
    assert r.vision.ok is True and r.bus.events[-1] == ("camera", {"ok": True})


def test_camera_that_disappears_is_released_toasted_and_drops_the_face():
    r = rig()
    r.boxes[:] = [RIGHT]
    r.vision.step(0)
    r.cams[0].frames_ok = False
    r.vision.step(1)
    assert r.vision.face is None and r.cams[0].released and r.vision.ok is False
    assert faces(r)[-1] == {"visible": False}
    assert ("error", {"kind": "other", "message": UNAVAILABLE}) in r.bus.events
    r.vision.step(2)
    assert len(r.cams) == 1
    r.vision.step(1 + RETRY_S)
    assert len(r.cams) == 2


def test_detector_error_counts_as_a_camera_failure():
    r = rig()
    r.error = RuntimeError("bad frame")
    r.vision.step(0)
    assert r.vision.ok is False and r.cams[0].released


def test_index_change_reopens_the_camera():
    r = rig()
    r.vision.step(0)
    r.cfg.camera.index = 1
    r.vision.step(0.1)
    assert r.cams[0].released and r.cams[1].index == 1


def test_disabled_releases_the_camera_until_enabled_again():
    r = rig()
    r.boxes[:] = [RIGHT]
    r.vision.step(0)
    r.cfg.camera.enabled = False
    r.vision.step(1)
    r.vision.step(2)
    assert r.cams[0].released and len(r.cams) == 1 and r.vision.face is None and r.vision.ok is None
    r.cfg.camera.enabled = True
    r.vision.step(3)
    assert len(r.cams) == 2


def test_thread_tracks_until_stopped_then_releases_the_camera():
    r = rig()
    r.boxes[:] = [RIGHT]
    r.vision.start()
    deadline = time.monotonic() + 2
    while r.vision.face is None and time.monotonic() < deadline:
        time.sleep(0.01)
    seen = r.vision.face
    r.vision.stop()
    assert seen is not None and r.cams[0].released and r.vision.face is None
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_vision.py -q`
Expected: collection error `ImportError: cannot import name 'RETRY_S' from 'ohbotchat.vision'`.

- [ ] **Step 3: Implement**

Replace the import block at the top of `ohbotchat/vision.py` with:

```python
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CameraConfig, Config

log = logging.getLogger(__name__)

MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
FPS = 10
DETECT_WIDTH = 320
MIN_SCORE_FLOOR = 0.3  # the net keeps everything above the lowest allowed min_score, so min_score applies live
RETRY_S = 5.0
PUBLISH_EVERY_S = 0.2  # the UI dot needs ~5 updates a second, not every frame
UNAVAILABLE = "Kamera niedostępna"
```

Below the `Box = ...` line add:

```python
Detect = Callable[[np.ndarray], list[Box]]
```

Append to `ohbotchat/vision.py`:

```python
def load_detector() -> Detect:
    import cv2

    net = cv2.FaceDetectorYN.create(str(MODEL), "", (DETECT_WIDTH, DETECT_WIDTH), MIN_SCORE_FLOOR)

    def detect(frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        scale = DETECT_WIDTH / w
        small = cv2.resize(frame, (DETECT_WIDTH, round(h * scale)))
        net.setInputSize((small.shape[1], small.shape[0]))
        _, found = net.detect(small)
        if found is None:
            return []
        # YuNet rows: x, y, w, h, 5 landmark pairs, score
        return [(*(float(v) / scale for v in f[:4]), float(f[14])) for f in found]

    return detect


def _open_camera(index: int):
    import cv2

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # HD frames would only be shrunk again
    return cap


class Vision:
    """Finds the largest face on a daemon thread and keeps the latest one in `face` for the robot tick.

    Frames never leave this object: not saved, logged or sent anywhere. Only the position goes on the bus.
    """

    def __init__(
        self, config: Config, bus, open_camera: Callable = _open_camera, make_detector: Callable[[], Detect] = load_detector
    ) -> None:
        self._cfg, self._bus = config, bus
        self._open_camera, self._make_detector = open_camera, make_detector
        self._detect: Detect | None = None
        self._cap = None
        self._index: int | None = None
        self._retry_at = 0.0
        self._ema: tuple[float, float] | None = None
        self._published_at = -math.inf
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        self.face: Face | None = None
        self.ok: bool | None = None  # None before the first frame and while disabled

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()
        if self._thread:
            self._thread.join(timeout=2)
        else:
            self._release()

    def _run(self) -> None:
        while not self._halt.is_set():
            self.step(time.monotonic())
            self._halt.wait(1 / FPS)
        self._release()

    def step(self, now: float) -> None:
        """One frame: (re)open the camera when due, detect, publish."""
        c = self._cfg.camera
        if not c.enabled:
            self._release()
            self._set_ok(None)
            return
        if c.index != self._index:
            self._release()
            self._index, self._retry_at = c.index, 0.0
        if self._cap is None:
            if now < self._retry_at:
                return
            try:
                self._cap = self._connect(c.index)
            except Exception:
                self._fail(now)
                return
        try:
            ok, frame = self._cap.read()
            if not ok:
                raise OSError("camera returned no frame")
            face = self._find(frame, now)
        except Exception:
            self._fail(now)
            return
        self._set_ok(True)
        self._show(now, face)

    def _connect(self, index: int):
        if self._detect is None:
            self._detect = self._make_detector()
        cap = self._open_camera(index)
        if not cap.isOpened():  # also what a denied macOS camera permission looks like
            cap.release()
            raise OSError(f"camera {index} did not open")
        return cap

    def _find(self, frame: np.ndarray, now: float) -> Face | None:
        c = self._cfg.camera
        h, w = frame.shape[:2]
        box = largest(self._detect(frame), c.min_score)
        if box is None:
            self._ema = None
            return None
        x, y = position(box, w, h, c)
        if self._ema is not None:
            a = c.smoothing
            x, y = a * self._ema[0] + (1 - a) * x, a * self._ema[1] + (1 - a) * y
        self._ema = (x, y)
        return Face(x, y, box[2] / w, now)

    def _show(self, now: float, face: Face | None) -> None:
        was_seen = self.face is not None
        self.face = face
        if face is None:
            if was_seen:
                self._bus.publish("face", visible=False)
        elif now - self._published_at >= PUBLISH_EVERY_S:
            self._published_at = now
            self._bus.publish("face", visible=True, x=round(face.x, 2), y=round(face.y, 2))

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._ema = None
        self._show(0.0, None)

    def _fail(self, now: float) -> None:
        self._release()
        self._retry_at = now + RETRY_S
        if self.ok is not False:
            log.warning("camera %s unavailable", self._index, exc_info=True)
            self._bus.publish("error", kind="other", message=UNAVAILABLE)
        self._set_ok(False)

    def _set_ok(self, ok: bool | None) -> None:
        if ok != self.ok:
            self.ok = ok
            self._bus.publish("camera", ok=ok)
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest -q`
Expected: all pass (cv2 is not imported by any test yet).

- [ ] **Step 5: Commit**

```bash
git add ohbotchat/vision.py tests/test_vision.py
git commit -m "feat(vision): camera thread that keeps the largest face, with a detector seam"
```

---

### Task 4: Robot follows the face

**Files:**
- Modify: `ohbotchat/robot.py`
- Test: `tests/test_robot.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_robot.py` add the import `from ohbotchat.vision import Face` after the `ohbotchat.state` import, then append:

```python
def face(x=0.0, y=0.0):
    return Face(x, y, 0.2, 0.0)


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
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_robot.py -q`
Expected: the new tests FAIL with `TypeError: Robot.tick() takes 2 positional arguments but 3 were given`; existing tests pass.

- [ ] **Step 3: Implement**

In `ohbotchat/robot.py`:

Add `import math` to the stdlib imports and `from .vision import Face, motor_target` after `from .state import State`.

After `MANUAL_PAUSE_S = ...` add:

```python
EYE_SPEED, HEAD_SPEED = 8, 3  # eyes lead and the head follows, as people do
MIN_STEP = 0.2  # smaller tracking moves are detector noise and would flood the serial link
```

In `Robot.__init__`, after `self._closed = False`:

```python
        self._seen_at = -math.inf
        self._pos: dict[int, float] = {}  # last tracking position sent per motor; any other move clears it
        self._tracked = False
        self._gesturing = 0
```

Replace `pause_idle` with:

```python
    def pause_idle(self) -> None:
        self._idle_after = time.monotonic() + MANUAL_PAUSE_S
        self._recentre_at = None
        self._pos.clear()
```

Replace `gesture` with:

```python
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
```

Replace `glance` and `tick` with:

```python
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
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest -q`
Expected: all pass, including the old blink/glance/pause tests (a call without a face counts as lost, so they behave as before).

- [ ] **Step 5: Commit**

```bash
git add ohbotchat/robot.py tests/test_robot.py
git commit -m "feat(robot): follow the face with eyes and head; glance only when no face is seen"
```

---

### Task 5: App wiring, 0.1 s tick and auto greeting

**Files:**
- Modify: `ohbotchat/app.py`, `ohbotchat/robot.py` (remove `idle_loop`)
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_app.py`:

Add `import time` to the imports and `from ohbotchat.vision import Face` after the `ohbotchat.robot` import. After `class FakeListener` add:

```python
class FakeVision:
    def __init__(self):
        self.face, self.ok, self.running = None, None, False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
```

In `make()`, add `vision=FakeVision(),` to the `App(...)` call. In `hands_free()`, add `vision=FakeVision()` to the `App(...)` call. In the three tests that call `app.start()` (`test_start_runs_mic_loop_when_configured_hands_free`, `test_mic_error_at_start_is_kept_for_clients_that_connect_later`, `test_start_greets_only_with_the_robot_connected`), add `vision=FakeVision()` to the `App(...)` call so no test opens a camera.

Append:

```python
FACE = Face(0.0, 0.0, 0.2, 0.0)
GREETING_PCM = b"SAY-Cze\xc5"


def arrive(app, t):
    """A face seen at t and still there 1.1 s later, as the tick loop would report it."""
    app._auto_greet(t, FACE)
    app._auto_greet(t + 1.1, FACE)


async def test_child_who_walks_up_is_greeted_once_after_a_second():
    engine = FakeEngine()
    app, _ = make(engine)
    t = time.monotonic()
    app._auto_greet(t, FACE)
    app._auto_greet(t + 0.5, FACE)
    assert app._task is None  # walking past
    app._auto_greet(t + 1.1, FACE)
    await settle(app)
    assert engine.said == [(GREETING, None)]
    arrive(app, t + 30)  # left and came back within greet_cooldown_s
    await settle(app)
    assert app.speaker.played == [GREETING_PCM]


async def test_child_arriving_after_the_cooldown_gets_a_fresh_conversation():
    app, _ = make()
    t = time.monotonic()
    arrive(app, t)
    await settle(app)
    await app.handle_utterance(LONG_ENOUGH)
    assert app.history
    arrive(app, t + 62.1)  # nobody for 61 s
    await settle(app)
    assert app.speaker.played.count(GREETING_PCM) == 2 and app.history == []


async def test_no_auto_greet_right_after_a_greeting_or_when_switched_off():
    app, _ = make()
    await app.greet()  # startup or "Nowa rozmowa"
    arrive(app, time.monotonic())
    await settle(app)
    assert app.speaker.played == [GREETING_PCM]
    app, _ = make()
    app.config.camera.auto_greet = False
    arrive(app, time.monotonic())
    await settle(app)
    assert app.speaker.played == []


async def test_no_auto_greet_for_a_child_already_talking_to_the_robot():
    speaker = FakeSpeaker(block=True)
    app, _ = make(speaker=speaker)
    task = app.start_turn(LONG_ENOUGH)
    await speaker.started.wait()
    t = time.monotonic()
    arrive(app, t)
    speaker.stop()
    await asyncio.gather(task, return_exceptions=True)
    app._auto_greet(t + 1.5, FACE)
    await settle(app)
    assert GREETING_PCM not in speaker.played


async def test_auto_greet_in_wake_mode_opens_the_window_for_the_child():
    engine = Recorder("opowiedz dowcip")
    app, mic = hands_free("wake", engine)
    arrive(app, time.monotonic())
    await settle(app)
    await mic.say()
    await settle(app)
    assert engine.heard == ["opowiedz dowcip"]  # no "Ohbot" needed


async def test_auto_greet_in_ptt_mode_only_greets():
    app, _ = make()
    arrive(app, time.monotonic())
    await settle(app)
    assert app.speaker.played == [GREETING_PCM] and app._wake_until == 0.0 and app.state == "idle"


async def test_start_runs_the_camera_and_the_tick_feeds_its_face_to_the_robot():
    cfg = load_config(env={})
    driver, vision = NullDriver(), FakeVision()
    vision.face = FACE
    app = App(cfg, engine=FakeEngine(), speaker=FakeSpeaker(), fallback=FakeFallback(), listener=FakeListener(), robot=Robot(driver, cfg, EventBus()), vision=vision, mic_stream=MicStreams())
    await app.start()
    await asyncio.sleep(0.05)
    app.robot.flush()
    assert vision.running and ("move", EYETURN, 5.0, 8) in driver.calls
    await app.close()
    assert not vision.running
```

Add `EYETURN` to the `from ohbotchat.robot import ...` line.

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_app.py -q`
Expected: every test errors with `TypeError: App.__init__() got an unexpected keyword argument 'vision'`.

- [ ] **Step 3: Implement**

In `ohbotchat/app.py`:

Add `import math` and `import time` to the stdlib imports and `from .vision import Vision` after `from .state import State, transition`. After `_FALLBACK_KEY = ...` add:

```python
TICK_S = 0.1  # eyes follow the face at camera speed (~10 fps); the old 0.25 s made the gaze lag
GREET_AFTER_S = 1.0  # a child walking past must not be greeted
```

Change the `__init__` signature to end with `listener=None, mic_stream=None, vision=None`, and after `self.listener = ...` add:

```python
        self.vision = vision or Vision(config, self.bus)
```

After `self._wake_until = 0.0` add:

```python
        self._face_since = self._face_last = self._greeted_at = -math.inf
        self._greet_due = False
```

In `start()`, replace `self._background = [asyncio.create_task(self.robot.idle_loop())]` with:

```python
        self.vision.start()
        self._background = [asyncio.create_task(self._tick_loop())]
```

In `close()`, before `await asyncio.to_thread(self.robot.close)` add:

```python
        await asyncio.to_thread(self.vision.stop)
```

Add after `close()`:

```python
    async def _tick_loop(self) -> None:
        while True:
            now, face = time.monotonic(), self.vision.face
            self.robot.tick(now, face)
            self._auto_greet(now, face)
            await asyncio.sleep(TICK_S)

    def _auto_greet(self, now: float, face) -> None:
        """Greets a child who walked up: a face for GREET_AFTER_S after nobody was there for greet_cooldown_s."""
        c = self.config.camera
        if face is None:
            return
        if now - self._face_last >= c.lost_after_s:  # a new arrival, not a missed detection
            self._face_since = now
            self._greet_due = now - self._face_last >= c.greet_cooldown_s
        self._face_last = now
        if not self._greet_due or now - self._face_since < GREET_AFTER_S:
            return
        self._greet_due = False  # one chance per arrival: a busy robot is already talking with them
        if not (c.auto_greet and self.ready()) or now - self._greeted_at < c.greet_cooldown_s:
            return
        self.new_conversation()  # whoever talked before left at least greet_cooldown_s ago
        task = self.greet()
        if self.config.conversation.mode == "wake":  # the child may answer without saying "Ohbot"
            task.add_done_callback(lambda t: t.cancelled() or self._open_wake_window())
```

Replace `greet()` with:

```python
    def greet(self) -> asyncio.Task:
        self._greeted_at = time.monotonic()  # startup and "Nowa rozmowa" greetings count for the auto greet cooldown
        return self.say(self.config.conversation.greeting, cache=True)
```

In `ohbotchat/robot.py`, delete the `idle_loop` method (the App loop replaced it).

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ohbotchat/app.py ohbotchat/robot.py tests/test_app.py
git commit -m "feat(app): tick robot and greeter from the camera every 0.1 s; greet a child who walks up"
```

---

### Task 6: Web: "Kamera" settings group and face dot

**Files:**
- Modify: `ohbotchat/web.py`, `ohbotchat/events.py`, `web/index.html`, `web/app.js`, `web/style.css`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_web.py`, add `import re` to the imports. In `FakeApp.__init__`, after `self.listener = ...` add:

```python
        self.vision = type("V", (), {"ok": True})()
```

Append:

```python
def test_snapshot_reports_the_camera(client, app):
    app.vision.ok = False
    with connect(client) as ws:
        assert snapshot(ws)["camera"] is False


def test_camera_settings_are_served_and_applied(client, app):
    assert client.get("/api/settings").json()["values"]["camera"]["index"] == 0
    r = client.post("/api/settings", json={"camera": {"index": 1, "mirror": True}}, headers=ORIGIN)
    assert r.status_code == 200 and app.config.camera.index == 1 and app.config.camera.mirror is True
    assert client.post("/api/settings", json={"camera": {"index": 10}}, headers=ORIGIN).status_code == 400


def test_every_settings_field_in_the_page_is_an_editable_key(client):
    from ohbotchat.config import EDITABLE

    keys = re.findall(r'data-key="(\w+)\.(\w+)"', client.get("/").text)
    assert all(k in EDITABLE[s] for s, k in keys)
    assert {k for s, k in keys if s == "camera"} == set(EDITABLE["camera"])
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_web.py -q`
Expected: `test_snapshot_reports_the_camera` FAILS with `KeyError: 'camera'`; `test_every_settings_field_in_the_page_is_an_editable_key` FAILS (no camera keys in the page). The settings test already passes through the generic settings API.

- [ ] **Step 3: Implement**

`ohbotchat/web.py`, in `snapshot()` after `"mic_error": app.mic_error,`:

```python
            "camera": app.vision.ok,
```

`ohbotchat/events.py`, replace the `type` comment:

```python
    type: str  # "state" | "stage" | "heard" | "reply" | "error" | "robot" | "level" | "info" | "face" | "camera"
```

`web/index.html`, insert before the `<fieldset><legend>Prompt` line:

```html
  <fieldset><legend>Kamera</legend>
    <p class="note">Obraz nie opuszcza aplikacji: nie jest zapisywany ani wysyłany. Robot zna tylko położenie twarzy.</p>
    <div id="face-box" title="Gdzie robot widzi twarz"><div id="face-dot"></div><span id="face-error" hidden>Kamera niedostępna</span></div>
    <label><input type="checkbox" data-key="camera.enabled"> Śledzenie twarzy</label>
    <label>Numer kamery <input type="number" data-key="camera.index" min="0" max="9" step="1"></label>
    <label><input type="checkbox" data-key="camera.mirror"> Odbicie lustrzane</label>
    <label>Przesunięcie poziome <input type="range" data-key="camera.offset_x" min="-1" max="1" step="0.05"><output></output></label>
    <label>Przesunięcie pionowe <input type="range" data-key="camera.offset_y" min="-1" max="1" step="0.05"><output></output></label>
    <label>Ruch głowy <input type="range" data-key="camera.head_gain" min="0" max="5" step="0.1"><output></output></label>
    <label>Ruch oczu <input type="range" data-key="camera.eye_gain" min="0" max="5" step="0.1"><output></output></label>
    <label>Martwa strefa głowy <input type="range" data-key="camera.deadband" min="0" max="3" step="0.1"><output></output></label>
    <label>Wygładzanie <input type="range" data-key="camera.smoothing" min="0" max="0.95" step="0.05"><output></output></label>
    <label>Pewność wykrycia <input type="range" data-key="camera.min_score" min="0.3" max="0.99" step="0.01"><output></output></label>
    <label>Brak twarzy po (s) <input type="number" data-key="camera.lost_after_s" min="0.5" max="30" step="0.5"></label>
    <label><input type="checkbox" data-key="camera.auto_greet"> Powitanie, gdy ktoś podejdzie</label>
    <label>Przerwa między powitaniami (s) <input type="number" data-key="camera.greet_cooldown_s" min="10" max="3600" step="10"></label>
  </fieldset>
```

`web/style.css`, append before the `[hidden]` rule:

```css
#face-box { position: relative; width: 240px; height: 180px; border: 2px solid var(--dim); border-radius: 12px; background: var(--bg); }
#face-dot { position: absolute; left: 50%; top: 50%; width: 24px; height: 24px; margin: -12px 0 0 -12px; border-radius: 50%; background: var(--dim); opacity: .3; }
#face-dot.seen { background: var(--accent); opacity: 1; }
#face-error { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 18px; color: #f87171; }
```

`web/app.js`, add after `function setRobot(...) {...}`:

```js
function setCamera(ok) {
  $("face-error").hidden = ok !== false;
}
```

In `handlers.snapshot`, after `setRobot(d.robot);` add `setCamera(d.camera);`. Add two handlers after `robot: (d) => setRobot(d.connected),`:

```js
  camera: (d) => setCamera(d.ok),
  face: (d) => {
    const dot = $("face-dot");
    dot.classList.toggle("seen", d.visible);
    if (d.visible) {
      dot.style.left = 50 + d.x * 50 + "%";
      dot.style.top = 50 - d.y * 50 + "%"; // y is up
    }
  },
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ohbotchat/web.py ohbotchat/events.py web/index.html web/app.js web/style.css tests/test_web.py
git commit -m "feat(web): Kamera settings group and face position dot"
```

---

### Task 7: Doctor camera check

**Files:**
- Modify: `ohbotchat/doctor.py`, `ohbotchat/web.py`
- Test: `tests/test_doctor.py`, `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_doctor.py`, replace the `run` helper with:

```python
async def run(client, cfg=None, interactive=False, driver=None, speaker=None, mic_fn=mic, camera_fn=lambda cfg: None):
    return await run_doctor(
        cfg or cfg_with_key(),
        interactive=interactive,
        client=client,
        driver=driver or NullDriver(),
        speaker=speaker or FakeSpeaker(),
        mic=mic_fn,
        camera=camera_fn,
    )
```

In `test_all_good_non_interactive_skips_hardware_actions`, change the two asserts to:

```python
    assert len(res) == 9
    assert [r.ok for r in res] == [True, True, True, True, True, True, None, True, True]
```

Append:

```python
async def test_camera_problem_is_reported_with_a_hint():
    res = await run(FakeClient(), camera_fn=lambda cfg: "kamera 0 nie zwraca obrazu")
    assert res[8].name == "Kamera" and res[8].ok is False
    assert res[8].detail == "kamera 0 nie zwraca obrazu" and "Prywatność" in res[8].hint


async def test_disabled_camera_is_skipped_without_opening_it():
    cfg = cfg_with_key()
    cfg.camera.enabled = False
    opened = []
    res = await run(FakeClient(), cfg=cfg, camera_fn=opened.append)
    assert res[8].ok is None and opened == []
```

In `tests/test_web.py`, `test_web_doctor_never_touches_the_robot_driver`, change the last line to:

```python
    assert touched == [] and next(r for r in results if r.name == "Ohbot").ok is True
    assert next(r for r in results if r.name == "Kamera").ok is True  # from the running app, not a second open
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_doctor.py tests/test_web.py -q`
Expected: doctor tests FAIL with `TypeError: run_doctor() got an unexpected keyword argument 'camera'`. The web doctor test FAILS with `StopIteration` (no "Kamera" result).

- [ ] **Step 3: Implement**

In `ohbotchat/doctor.py`, after `NO_KEY = ...` add:

```python
CAMERA_HINT = (
    "Podłącz kamerę, sprawdź numer kamery w Ustawieniach i nadaj dostęp do kamery "
    "(macOS: Ustawienia systemowe > Prywatność i ochrona > Kamera, dla Terminala; "
    "Windows: Ustawienia > Prywatność i zabezpieczenia > Kamera). Brak modelu oznacza niepełną instalację."
)
```

After `_robot_motion` add:

```python
def _default_camera(config: Config) -> str | None:
    """None when the model loads and the camera returns a frame, else what failed."""
    import cv2

    from .vision import load_detector

    try:
        load_detector()
    except cv2.error as e:
        return f"model YuNet nie wczytany ({e})"
    cap = cv2.VideoCapture(config.camera.index)
    try:
        ok = cap.isOpened() and cap.read()[0]
    finally:
        cap.release()
    return None if ok else f"kamera {config.camera.index} nie zwraca obrazu"
```

Add the parameter to `run_doctor` after `mic=...`:

```python
    camera: Callable[[Config], str | None] = _default_camera,
```

Inside `run_doctor`, after `robot_check`, add:

```python
    async def camera_check() -> CheckResult:
        name = "Kamera"
        if not config.camera.enabled:
            return CheckResult(name, None, f"{SKIPPED} (wyłączona w Ustawieniach)")
        problem = await asyncio.to_thread(camera, config)
        if problem:
            return CheckResult(name, False, problem, CAMERA_HINT)
        return CheckResult(name, True, f"kamera {config.camera.index} działa, model wczytany")
```

and append to the final `results += [...]` list:

```python
        await guarded("Kamera", camera_check),
```

In `ohbotchat/web.py`, after `class _RunningRobot` add:

```python
def _running_camera(app):
    """The app's vision thread holds the camera and a second open fails on Windows, so doctor only reports its state."""
    return lambda _config: None if app.vision.ok else "kamera niedostępna w działającej aplikacji"
```

and in `_run_doctor` change the `run_doctor(...)` call to:

```python
        lambda: asyncio.run(
            run_doctor(app.config, interactive=False, driver=_RunningRobot(app.robot), camera=_running_camera(app))
        )
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ohbotchat/doctor.py ohbotchat/web.py tests/test_doctor.py tests/test_web.py
git commit -m "feat(doctor): camera and face model check"
```

---

### Task 8: OpenCV dependency, YuNet model, packaging

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via uv), `packaging/ohbotchat.spec`
- Create: `ohbotchat/models/face_detection_yunet_2023mar.onnx`
- Test: `tests/test_vision.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_vision.py` add `MODEL, load_detector` to the `from ohbotchat.vision import ...` line, then append:

```python
def test_yunet_model_is_bundled():
    assert MODEL.is_file() and 200_000 < MODEL.stat().st_size < 300_000  # a Git LFS pointer would be ~130 bytes


def test_real_detector_finds_nothing_in_a_black_frame():
    assert load_detector()(np.zeros((480, 640, 3), np.uint8)) == []  # loads the model; no camera involved
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_vision.py -q`
Expected: `test_yunet_model_is_bundled` FAILS (`assert False`); `test_real_detector_finds_nothing_in_a_black_frame` FAILS with `ModuleNotFoundError: No module named 'cv2'`.

- [ ] **Step 3: Add the dependency and the model (needs network)**

```bash
uv add opencv-python-headless
mkdir -p ohbotchat/models
curl -fL -o ohbotchat/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
wc -c ohbotchat/models/face_detection_yunet_2023mar.onnx
```

Expected: `pyproject.toml` lists `opencv-python-headless>=…` and `uv.lock` changed. The model is about 230 KB (roughly 232 000 bytes). A file of ~130 bytes is a Git LFS pointer and a few KB is an HTML error page; delete it and retry the download.

In `packaging/ohbotchat.spec`, add to `datas` after the `config.toml` line:

```python
        (f"{ROOT}/ohbotchat/models", "ohbotchat/models"),
```

With `contents_directory="."`, frozen modules resolve `__file__` under the exe folder, so `vision.MODEL` finds `ohbotchat/models/` there.

The wheel needs no `pyproject.toml` change: hatch's `packages = ["ohbotchat"]` ships every non-ignored file in the package. Verify:

```bash
uv build --wheel -o /tmp/ohbot-wheel && unzip -l /tmp/ohbot-wheel/ohbotchat-*.whl | grep yunet
```

Expected: one line with `ohbotchat/models/face_detection_yunet_2023mar.onnx`.

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest -q`
Expected: all pass.

Optional on macOS (slow, smoke test of the spec only): `uv run --group build pyinstaller packaging/ohbotchat.spec --noconfirm && ls dist/ohbot-chat/ohbotchat/models/ dist/ohbot-chat/cv2 | head`. Expected: the `.onnx` file and the `cv2` package are listed. `dist/` and `build/` are git-ignored.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock packaging/ohbotchat.spec ohbotchat/models/face_detection_yunet_2023mar.onnx tests/test_vision.py
git commit -m "build: opencv-python-headless and the YuNet face model, bundled in the exe"
```

---

### Task 9: `[camera]` in config.toml

**Files:**
- Modify: `config.toml`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core.py`:

```python
def test_config_toml_lists_every_camera_key():
    import tomllib
    from dataclasses import fields

    from ohbotchat.config import ROOT, CameraConfig

    raw = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    # load_config skips unknown keys silently, so a typo here would quietly fall back to the default
    assert set(raw["camera"]) == {f.name for f in fields(CameraConfig)}
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_core.py -q`
Expected: FAILS with `KeyError: 'camera'`.

- [ ] **Step 3: Implement**

Append to `config.toml`:

```toml

[camera]
# Face tracking. Frames stay in the app: never saved, logged or sent anywhere.
enabled = true
# 0 = built-in camera; try 1 or 2 for a USB camera.
index = 0
# Flip if the robot looks away from the child instead of at them.
mirror = false
# Camera beside the robot: shift (-1..1) until it looks at a child straight in front of it.
offset_x = 0.0
offset_y = 0.0
# How far head and eyes turn for a face at the edge of the picture (0-5).
head_gain = 3.0
eye_gain = 4.0
# The head moves only when its target is this far off; the eyes follow every frame.
deadband = 0.5
# 0 = jumpy, 0.95 = slow and smooth.
smoothing = 0.5
# Detector confidence (0.3-0.99); raise it if posters or shadows count as faces.
min_score = 0.7
# Seconds without a face before the head recentres and looks around again.
lost_after_s = 3.0
# Greet a child who walks up after nobody was there for greet_cooldown_s.
auto_greet = true
greet_cooldown_s = 60.0
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add config.toml tests/test_core.py
git commit -m "chore(config): camera keys in config.toml"
```

---

### Task 10: Docs: camera, privacy, troubleshooting

**Files:**
- Modify: `docs/windows.md`

No README exists, so the spec's "README/windows.md camera section" lives in `docs/windows.md` only.

- [ ] **Step 1: Write the section**

In `docs/windows.md`, insert before `## Build`:

```markdown
## Camera

The robot turns its eyes and head toward the nearest face and greets a child who walks up. Settings are in "Ustawienia" > "Kamera" and in `[camera]` in `config.toml`.

**Privacy:** camera frames never leave the app. They are not saved, not logged, and not sent to OpenAI or to the browser. The UI receives only the face position (the dot in "Kamera").

- Windows: Settings > Privacy & security > Camera: turn on "Camera access" and "Let desktop apps access your camera".
- macOS (run from source): System Settings > Privacy & Security > Camera: allow the terminal app that runs `uv run ohbotchat`.
- Calibration: stand straight in front of the robot and move "Przesunięcie poziome" until the robot looks at you. If it turns away from you, tick "Odbicie lustrzane".
- `ohbot-chat.exe doctor` checks that the camera returns a frame and that the face model loads.
- To turn the camera off, untick "Śledzenie twarzy" or set `enabled = false`. The camera is then released.

The face model is YuNet (`ohbotchat/models/face_detection_yunet_2023mar.onnx`) from OpenCV Zoo, MIT licence.
```

In `## Troubleshooting`, add after the microphone block:

```markdown
**"Kamera niedostępna"**
- Another app may hold the camera (Teams, Zoom, the Camera app). Close it. The app retries every 5 s.
- With more than one camera, try "Numer kamery" 1 or 2 in "Ustawienia".
- Check the camera permission (see "Camera" above). The rest of the app keeps working without the camera.
```

- [ ] **Step 2: Commit**

```bash
git add docs/windows.md
git commit -m "docs: camera setup, privacy and troubleshooting"
```

---

### Task 11: Final verification and hardware check

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: all pass. No test opens a camera or the serial port. `test_real_detector_finds_nothing_in_a_black_frame` loads the model only.

- [ ] **Step 2: Hardware check (the user, not an agent)**

The user starts `uv run ohbotchat` with the robot beside the MacBook and checks:
1. Head and eyes turn toward them. If they turn away, tick "Odbicie lustrzane". If the head nods the wrong way, flip the sign of `ny` in `vision.position()` (there is no y-mirror setting).
2. Calibrate `offset_x` for the off-axis MacBook camera, then click "Zapisz".
3. Smoothness: tune "Wygładzanie" and "Martwa strefa głowy"; the serial link must not stutter the lips while talking.
4. False greetings: posters or people passing. Raise "Pewność wykrycia" or "Przerwa między powitaniami".
5. In wake mode, after an auto greeting, answering without "Ohbot" works.
