# Ohbot face tracking — design

Date: 2026-11-21. Status: approved in chat, pending spec review.

## Goal
Ohbot looks at the child it talks to. A camera finds faces; the robot turns eyes and head toward the largest one,
and greets a child who walks up. Part 1 of "more natural Ohbot"; motion polish (blink, glance, gestures) is a
separate follow-up spec after the user's notes.

## Cameras
- Now (tests): the MacBook built-in camera, robot standing next to the laptop (camera off the robot's axis).
- Later: a Windows PC (packaging/) or a USB camera on/next to the robot.
- So the camera index, mirroring and the off-axis offset are config values, editable live in "Ustawienia".

## Detection
- OpenCV `cv2.FaceDetectorYN` with the YuNet model (`face_detection_yunet_2023mar.onnx`, OpenCV Zoo, MIT, ~230 KB),
  committed under `ohbotchat/models/` and included in the PyInstaller spec.
- New dependency: `opencv-python-headless` (no GUI needed, smaller than `opencv-python`).
- Rejected: Haar cascades (lose profile faces and dim light), MediaPipe (heavy, Windows packaging trouble).

## Privacy
Frames never leave the process: not saved, not logged, not sent to OpenAI or to the browser. The UI receives only the
normalised face position. State this in docs and in the README/windows.md camera section.

## Components
- **`ohbotchat/vision.py`**
  - `Vision` runs a daemon thread. It opens `cv2.VideoCapture([camera] index)`, grabs ~10 fps, downsizes to ~320 px
    width, and runs the detector.
  - It picks the largest face with score ≥ `min_score`.
  - It publishes `Face(x, y, size, t)` with x, y in -1..1, after `mirror` and the `offset_x/offset_y` calibration, or
    `None` when no face. It is read by the robot tick; the latest value also goes on the event bus, throttled to ~5/s,
    for the UI dot.
  - The detector sits behind a tiny seam (`detect(frame) -> list[box]`) so tests feed fake boxes without OpenCV or a
    camera.
  - Camera open failure, denied macOS permission, or a camera that disappears: tracking stops and the bus gets an
    error toast ("Kamera niedostępna"). The rest of the app keeps working. Retry every 5 s, like the robot.
- **Mapping (pure functions, in vision.py or robot.py)**
  - Target eye position = 5 + x · `eye_gain`, head position = 5 + x · `head_gain` (same for y with EYETILT/HEADNOD),
    clamped 0..10.
  - Deadband: the head moves only when the target differs from the current head position by more than `deadband`.
    Eyes follow every frame.
  - Smoothing: exponential moving average on x, y (`smoothing` 0..1).
  - Eyes move fast (speed ~8), head slower (speed ~3), so eyes lead and the head follows, as people do.
- **Robot integration (robot.py tick)**
  - Tracking runs in every state (idle, listening, thinking, speaking) when a face is visible and `[camera] enabled`.
  - A running gesture owns the head motors; tracking resumes after it ends. Blink still runs on top of tracking.
  - Tracking pauses while the "Test ruchów" panel is open and during the manual-move pause, like idle moves.
  - Random glance runs only when no face has been seen for `lost_after_s` (default 3 s); then the head recentres.
  - Motor writes go through the existing single-worker executor; tracking sends a move only when the position changed
    by ≥ 0.2, to keep the serial link quiet.
- **Auto greeting (app.py)**
  - Fires when: `auto_greet` is on; a face has been visible for ≥ 1 s; no face was seen for the preceding
    `greet_cooldown_s` (default 60) and the robot has not greeted within that time; the robot is ready (same
    `ready()` as test commands).
  - It plays the existing greeting path (output moderation, cached audio).
  - In wake mode it then opens the wake window, so the child can speak without saying "ohbot". In continuous mode
    listening resumes as usual. In PTT mode it only greets.

## Config `[camera]`
| key | default | range |
|---|---|---|
| enabled | true | bool |
| index | 0 | 0..9 |
| mirror | false | bool — flip when the robot looks the wrong way |
| offset_x | 0.0 | -1..1 |
| offset_y | 0.0 | -1..1 |
| head_gain | 3.0 | 0..15 |
| eye_gain | 10.0 | 0..15 — faces sit near the frame centre with a wide laptop camera, so eyes need a large gain to move visibly |
| deadband | 0.5 | 0..3 |
| smoothing | 0.5 | 0..0.95 |
| min_score | 0.7 | 0.3..0.99 |
| lost_after_s | 3.0 | 0.5..30 |
| auto_greet | true | bool |
| greet_cooldown_s | 60 | 10..3600 |

All keys are validated and saved by the existing settings mechanism (editable-key list in config.py). Changing
`index` reopens the camera.

## UI
"Ustawienia" gets a "Kamera" group with the keys above and a small box with a dot showing where the robot sees the
face (no image). The dot greys out when no face is visible; the box shows "Kamera niedostępna" on error.

## Doctor
`ohbotchat doctor` checks that the camera opens and returns a frame, and that the model file loads.

## Testing
- Unit: mapping (gain, clamp, mirror, offset, deadband, smoothing), largest-face choice, min_score filter, lost-face
  timing, glance only without a face, gesture precedence, pause with the test panel, auto-greet conditions and
  cooldown, wake window opened after auto-greet, camera failure → toast + retry. All with a fake detector/capture and
  the existing fake robot driver.
- Hardware (user): direction of head/eyes vs where the child stands, calibration of offset_x for the MacBook camera,
  smoothness, false greetings.

## Out of scope
Recognising who the child is, gaze/emotion detection, following movement without a face, multiple-face attention
switching, showing the camera image in the UI.
