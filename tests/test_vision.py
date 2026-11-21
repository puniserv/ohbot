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
