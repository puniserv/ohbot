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
   ohbot-chat.exe doctor     # checks key, internet, models, mic, speaker, robot
   ohbot-chat.exe            # web UI in the default browser
   ohbot-chat.exe --window   # web UI in its own window
   ```

`--window` uses Edge WebView2, which ships with Windows 10/11. If pywebview is missing (a source install without the `desktop` group), the app logs a warning and opens the browser.

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

**Port 8765 in use**: start with `--port 8766`.
