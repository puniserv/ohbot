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

Box = tuple[float, float, float, float, float]  # x, y, w, h, score in pixels of the frame given to detect()
Detect = Callable[[np.ndarray], list[Box]]


@dataclass(frozen=True)
class Face:
    x: float  # -1..1 after mirror and offset
    y: float  # -1..1, up is positive
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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # else detection runs on stale queued frames and the gaze lags
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
        self._halt.clear()  # sleep stops and wake starts it again
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
        self._set_ok(None)

    def step(self, now: float) -> None:
        """One frame: (re)open the camera when due, detect, publish."""
        c = self._cfg.camera
        if not c.enabled:
            self._release()
            self._set_ok(None)
            self._retry_at = 0.0  # re-enabling retries at once
            return
        if c.index != self._index:
            self._release()
            self._set_ok(None)  # so a failure of the new camera toasts again
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
        return Face(x, y, now)

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
