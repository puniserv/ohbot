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
