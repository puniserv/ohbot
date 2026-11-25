---
name: add-setting
description: Add a configuration key to ohbotchat, or make an existing one editable in the "Ustawienia" panel. Use when a task adds a tunable value (motion, camera, voice, conversation), exposes a config.toml key in the web UI, or changes the allowed range of a setting.
---

# Add a setting

A setting lives in up to five places. Missing one fails silently: the key loads as its default, or the UI control
does nothing.

## Steps

1. **Dataclass field** in `ohbotchat/config.py` (`RobotConfig`, `CameraConfig`, ...), with a default and a type
   annotation. `load_config` checks `config.toml` values against the annotation, so annotate `float` for a value
   that may be written as `8` or `8.0`.
2. **`config.toml`**: add the key with a comment that tells the operator what to change and when. Keep the value
   equal to the dataclass default unless the committed file is meant to differ.
3. Stop here for a key that is only set in the file.

To make it editable in the UI:

4. **`EDITABLE`** in `ohbotchat/config.py`: add `section -> key -> checker` (`_number(kind, lo, hi)`, `_flag`,
   `_text(max_len)`, `_texts(max_items, max_len)`). The checker is the only validation of browser input.
5. **`web/index.html`**: add a control inside `#settings-panel` with `data-key="section.key"`. `app.js` wires every
   `data-key` control on its own. The control's `min`/`max`/`step` must match the checker's range. A range input
   takes an `<output></output>` sibling to show its value.
6. **`App.apply_settings`** in `ohbotchat/app.py`: only when something caches the value. Most code reads the config
   each time it is used and needs nothing here; `vad.silence_ms` (copied into the segmenter) and the TTS voice
   (cache directory) are the existing exceptions.

## Tests

- `tests/test_core.py`: add a rejected value to `test_validate_settings_rejects`, and the key to the expected dict in
  `test_validate_settings_accepts_known_keys_and_normalises` when its section is listed there.
- `test_config_toml_lists_every_camera_key` fails when a `CameraConfig` field is missing from `config.toml`; the
  `data-key` test in `tests/test_web.py` fails when a `data-key` has no checker or an editable `[camera]` key has no
  control.
- Tests build `Config()` and set the field; they do not read `config.toml`.

Done when `uv run pytest` passes and, for an editable key, changing the control in the UI changes the behaviour
without a restart and "Zapisz" writes the line into `config.toml`.
