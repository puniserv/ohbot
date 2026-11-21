import time
from types import SimpleNamespace as NS

import numpy as np

from ohbotchat.config import CameraConfig, Config
from ohbotchat.vision import MODEL, RETRY_S, UNAVAILABLE, Face, Vision, largest, load_detector, motor_target, position

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
    cfg = Config()  # code defaults: tuning config.toml on the robot must not break these tests
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
    assert r.vision.face == Face(0.5, 0.0, 10.0)
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
    assert r.vision.ok is None  # asleep, not failed
    r.vision.start()  # wake after sleep
    deadline = time.monotonic() + 2
    while r.vision.face is None and time.monotonic() < deadline:
        time.sleep(0.01)
    r.vision.stop()
    assert len(r.cams) == 2 and r.cams[1].released


def test_yunet_model_is_bundled():
    assert MODEL.is_file() and 200_000 < MODEL.stat().st_size < 300_000  # a Git LFS pointer would be ~130 bytes


def test_real_detector_finds_nothing_in_a_black_frame():
    assert load_detector()(np.zeros((480, 640, 3), np.uint8)) == []  # loads the model; no camera involved


def test_camera_keeps_one_buffered_frame_so_detection_sees_the_latest(monkeypatch):
    import cv2

    from ohbotchat import vision

    props = {}

    class Cap:
        def __init__(self, index):
            pass

        def set(self, prop, value):
            props[prop] = value

    monkeypatch.setattr(cv2, "VideoCapture", Cap)  # no real camera is opened
    vision._open_camera(0)
    assert props[cv2.CAP_PROP_BUFFERSIZE] == 1


def test_re_enabling_a_failed_camera_retries_at_once():
    r = rig()
    r.opened = False
    r.vision.step(0)
    r.cfg.camera.enabled = False
    r.vision.step(1)
    r.cfg.camera.enabled = True
    r.opened = True
    r.vision.step(2)
    assert len(r.cams) == 2 and r.vision.ok is True


def test_a_new_camera_index_that_also_fails_toasts_again():
    r = rig()
    r.opened = False
    r.vision.step(0)
    r.cfg.camera.index = 1
    r.vision.step(1)
    assert len(r.cams) == 2 and [t for t, _ in r.bus.events].count("error") == 2
