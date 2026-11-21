# Ohbot chat on Windows

## Install (prebuilt folder)

1. Copy the whole `ohbot-chat` folder (from `dist\ohbot-chat`) to the target PC, for example to `C:\Ohbot\ohbot-chat`. The exe needs every file in that folder.
2. Create `.env` in the same folder as `ohbot-chat.exe` (use `.env.example` from the repo as a template):

   ```
   OPENAI_API_KEY=sk-...
   ```

   The app reads `.env`, `config.toml`, `personas\` and `web\` from the exe's folder, and writes `cache\` and `ohbotData\` there, whatever directory you start it from. The folder must be writable, so do not put it in `C:\Program Files`; if it is read-only, the robot shows as unplugged.
   You can edit `config.toml` and the persona files in place. Restart the app after you change them.
3. Plug in the robot, then double-click `ohbot-chat.exe` or run it from a terminal:

   ```
   ohbot-chat.exe doctor     # checks key, internet, models, mic, speaker, robot, camera
   ohbot-chat.exe            # web UI in the default browser
   ohbot-chat.exe --window   # web UI in its own window
   ```

`--window` uses Edge WebView2, which ships with Windows 10/11. If pywebview is missing (a source install without the `desktop` group), the app logs a warning and opens the browser.

## Camera

The robot turns its eyes and head toward the nearest face and greets a child who walks up. Settings are in "Ustawienia" > "Kamera" and in `[camera]` in `config.toml`.

**Privacy:** camera frames never leave the app. They are not saved, not logged, and not sent to OpenAI or to the browser. The UI receives only the face position (the dot in "Kamera").

- Windows: Settings > Privacy & security > Camera: turn on "Camera access" and "Let desktop apps access your camera".
- macOS (run from source): System Settings > Privacy & Security > Camera: allow the terminal app that runs `uv run ohbotchat`.
- Calibration: stand straight in front of the robot and move "Przesunięcie poziome" until the robot looks at you. If it turns away from you, tick "Odbicie lustrzane".
- `ohbot-chat.exe doctor` checks that the camera returns a frame and that the face model loads.
- To turn the camera off, untick "Śledzenie twarzy" or set `enabled = false`. The camera is then released.

The face model is YuNet (`ohbotchat/models/face_detection_yunet_2023mar.onnx`) from OpenCV Zoo, MIT licence.

## Build

Run this on Windows with [uv](https://docs.astral.sh/uv/) installed:

```
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

The script runs `uv sync` with the `build` and `desktop` groups, then runs PyInstaller with `packaging\ohbotchat.spec`.
The result is `dist\ohbot-chat\` (a one-folder build). `dist\` and `build\` are git-ignored.
PyInstaller cannot cross-compile, so a Windows exe has to be built on Windows.

## Run from source

```
uv sync --group desktop
uv run ohbotchat --window
```

Here `.env` goes in the repo root.

## Privacy (for the operator)

- **What leaves the computer:** what children say and the robot's replies go to OpenAI: their speech for transcription, the transcript and the reply for moderation, the conversation for the chat model, and the reply text for speech synthesis. Camera frames never leave the app; only the face position is used.
- **Retention:** per OpenAI's "Your data" page for the API, data sent to the API is not used to train models unless the account opts in. Abuse monitoring logs are kept for up to 30 days, longer only where the law requires it. Check the current wording at developers.openai.com before a workshop.
- **Children:** tell the school or the parents before a session and get parental consent where the law requires it (in the EU, GDPR). OpenAI's terms set a minimum age and need a parent's or guardian's permission for minors, and its usage policies cover services used by minors. Check that your use follows them.
- **Modes:** only push-to-talk ("Przycisk") limits what is uploaded to what a child says while holding the button. In "Słowo „Ohbot”" and "Ciągła rozmowa" every utterance in the room is sent to OpenAI, because the wake word is only matched after transcription. The app shows this under the mode selector.
- The app saves no recordings to disk.

## Troubleshooting

**Robot not found** (`doctor` shows "Ohbot: robot nie odpowiada")
- The `ohbot` library scans every COM port and uses the first one that answers. There is no port setting.
- Open Device Manager, then "Ports (COM & LPT)". The robot should appear as a USB serial device when plugged in and powered. If it shows a warning icon, the USB serial driver is missing. Install the one from the Ohbot software package.
- Only one program can hold the port. Close the Ohbot desktop app, Arduino IDE, or another `ohbot-chat.exe`.
- Use a data USB cable, not a charge-only one, and turn on the robot's power before starting the app. The app retries the connection every 5 s, so you can plug it in later.

**Microphone silent or missing**
- The app records through the Windows default input device, not through the browser. Set it in Settings > System > Sound > Input.
- Settings > Privacy & security > Microphone: turn on "Microphone access" and "Let desktop apps access your microphone".
- Run `ohbot-chat.exe doctor` in a terminal. It lists the input devices and, when run interactively, shows the level while you speak.
- If another app holds the mic in exclusive mode (Teams, Zoom), close it or turn off "Allow applications to take exclusive control" in the device's advanced properties.

**"Kamera niedostępna"**
- Another app may hold the camera (Teams, Zoom, the Camera app). Close it. The app retries every 5 s.
- With more than one camera, try "Numer kamery" 1 or 2 in "Ustawienia".
- Check the camera permission (see "Camera" above). The rest of the app keeps working without the camera.

**`--window` opens the browser instead**: the window could not start, usually because Edge WebView2 is missing (the log says "brak WebView2?"). Install the WebView2 Runtime from Microsoft, or just use the browser.

**Port 8765 in use**: start with `--port 8766`.
