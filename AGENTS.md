# Ohbot chat

Voice chat for the Ohbot robot, used with children. Setup and run commands: `README.md`.

## Traps

- **Tests build `Config()`; `load_config()` is for the entry points in `__main__.py`.** It reads the repository's `config.toml` and `.env`,
  so a test would change with the operator's tuning and could pick up a real API key.
- **`ohbot` is imported only inside `OhbotDriver.connect`.** The library opens the serial port on first import and
  creates `ohbotData/` in the current directory. Tests use `NullDriver`.
- **Every driver call goes through `Robot._submit` / `Robot._guard`.** Serial is not thread-safe; one worker thread
  owns it. A direct `self._driver.move(...)` from the event loop is a bug.
- **Nothing a child said reaches the UI or the speaker before moderation.** `heard` is published from the `on_safe`
  callback; `info` events carry a fixed reason only. Text that is spoken but did not come from the
  chat model goes through `engine.say`, which moderates it.
- **The web UI has no auth.** The server binds 127.0.0.1 only. Every new HTTP route needs
  `dependencies=[Depends(guard)]`; the WebSocket checks `_same_origin`.
- **User-facing text is Polish** (UI, spoken phrases, error messages, `doctor` output). Code, comments, tests and
  docs are English.
- **Motor directions and positions cannot be checked without the robot.** When a change depends on them, report it as
  unverified on hardware; a simulated result counts as a simulation only.

## Checks

- `uv run pytest` — must pass; needs no robot, microphone, camera or network.
- No linter is configured.

## Where things go

- New setting editable in the UI: skill `add-setting`.
- Specs: `docs/specs/<slug>/spec.md`; plans: `docs/plans/<slug>.md`.
- Windows packaging: `packaging/`, described in `docs/windows.md`.
