# Ohbot chat

Polish voice chat for the [Ohbot](https://www.ohbot.co.uk/) robot head, built for workshops with children.
A child talks, the robot answers aloud, moves its lips with the sound, follows the child's face with eyes and head,
and picks a gesture for each reply.

Each turn goes through OpenAI: speech to text, moderation, chat model, moderation again, text to speech.
The web UI (in Polish) shows the conversation and which stage is running.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An OpenAI API key
- A microphone and a speaker on the computer (audio does not go through the browser)
- Optional: an Ohbot on USB, a camera. The app runs without either.

## Setup

```
uv sync                 # add --group desktop for the --window mode
cp .env.example .env    # then put OPENAI_API_KEY in it
uv run ohbotchat doctor # checks key, internet, models, microphone, speaker, robot, camera
```

## Run

```
uv run ohbotchat             # web UI at http://127.0.0.1:8765/ in the default browser
uv run ohbotchat --window    # the same UI in its own window (pywebview)
uv run ohbotchat --no-web    # terminal only: Enter starts recording, Enter stops
uv run ohbotchat --port 8766
```

The UI has no login, so the server listens on 127.0.0.1 only.

### Conversation modes

| Mode in the UI | `mode` | What is sent to OpenAI |
|---|---|---|
| Przycisk | `ptt` | only what is said while the button (or Space) is held |
| Słowo „Ohbot” | `wake` | every utterance in the room; the robot answers those that start with the wake word |
| Ciągła rozmowa | `continuous` | every utterance in the room, and the robot answers each |

Read "Privacy (for the operator)" in [docs/windows.md](docs/windows.md) before a session with children.

## Configuration

- `config.toml`: models, voice, conversation mode, wake words, eye colours, motion tuning, camera. Every key is
  commented there. A syntax error or a value of the wrong type stops the app at start with a message naming the key.
- "Ustawienia" in the UI changes a subset of those keys live; "Zapisz" writes them back to `config.toml` and keeps
  the comments.
- `personas/*.md`: the system prompt per persona, editable in the UI. Child-safety rules are added to every persona
  in code and cannot be edited away.
- "Test ruchów" in the UI: gestures, a slider per motor, a test sentence, and the camera calibration.

Generated speech for fixed phrases, fillers and the greeting is cached in `cache/`, per TTS model and voice.

## Development

```
uv run pytest
```

Tests run without a robot, microphone, camera or network: hardware and OpenAI sit behind injected fakes.

| Path | What |
|---|---|
| `ohbotchat/app.py` | one conversation turn at a time: state, turn tasks, sleep/wake |
| `ohbotchat/engine_chain.py` | the OpenAI pipeline |
| `ohbotchat/robot.py` | motors, eye colour, gestures, face pursuit, on one worker thread |
| `ohbotchat/vision.py` | camera thread and face detection (YuNet) |
| `ohbotchat/listen.py`, `loop.py`, `speak.py` | microphone (push-to-talk, hands-free with VAD) and playback |
| `ohbotchat/web.py`, `web/` | FastAPI app, WebSocket events, static UI |
| `ohbotchat/doctor.py` | the `doctor` checks |
| `docs/specs/`, `docs/plans/` | design specs and plans |

Windows build, install and troubleshooting: [docs/windows.md](docs/windows.md).
