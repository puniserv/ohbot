# Ohbot Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ohbot talks Polish with workshop participants: mic -> OpenAI STT -> chat -> TTS -> speaker + robot motion, with a local web UI.

**Architecture:** One Python 3.11 process. asyncio event bus connects listener, engine, speaker, robot and web UI; none of them call each other directly except through the orchestrator (`app.py`). Engine is an interface so a Realtime engine can be added later.

**Tech Stack:** Python 3.11, uv, `openai`, `sounddevice` + `numpy` (audio I/O), `webrtcvad-wheels` (VAD), `fastapi` + `uvicorn` (UI, WebSocket), `ohbot` (PyPI, motors/eyes), `pytest` + `pytest-asyncio`.

Spec: `docs/superpowers/specs/2026-11-21-ohbot-chat-design.md`. Deviation from spec layout: code lives in package `ohbotchat/` (not `core/`) so `uv run ohbotchat` and `python -m ohbotchat` work.

---

## File map

```
pyproject.toml               project, deps, console script ohbotchat = ohbotchat.__main__:main
config.toml                  defaults (models, mode, persona, colours, VAD, barge-in)
.env.example                 OPENAI_API_KEY=
personas/dzieci.md pokaz.md domowy.md
ohbotchat/
  __main__.py                CLI: `ohbotchat` (run), `ohbotchat doctor`, `ohbotchat --no-web` (terminal PTT)
  config.py                  load config.toml + .env -> Config dataclass
  events.py                  Event dataclass + EventBus (asyncio pub/sub)
  state.py                   State enum + pure transition function
  audio.py                   envelope(pcm) -> lip positions; level(); pcm helpers
  listen.py                  Listener: mic + VAD; modes ptt | wake | continuous; wake-word filter
  speak.py                   Speaker: play PCM, emit lip positions, stop() for interruption
  engine.py                  Engine protocol + Reply dataclass + EngineError kinds
  engine_chain.py            ChainEngine: STT -> moderation -> chat -> moderation -> TTS
  robot.py                   Robot (state colours, blink, idle glance, lips, gestures) over RobotDriver; OhbotDriver, NullDriver
  fallback.py                pre-generated fallback phrases cache (cache/fallback/*.pcm)
  doctor.py                  diagnostic checks -> list[CheckResult]
  app.py                     orchestrator: wires bus, listener, engine, speaker, robot; conversation history
  web.py                     FastAPI app, /ws WebSocket, static web/
web/index.html app.js style.css
tests/                       pytest, no hardware, no network
```

## Contracts (all stages must respect these)

```python
# events.py
@dataclass
class Event:
    type: str          # "state" | "stage" | "heard" | "reply" | "error" | "robot" | "level" | "info"
    data: dict
class EventBus:
    def subscribe(self) -> asyncio.Queue[Event]: ...
    def publish(self, type: str, **data) -> None: ...   # thread-safe (uses loop.call_soon_threadsafe)

# state.py
class State(StrEnum): IDLE, LISTENING, TRANSCRIBING, THINKING, SPEAKING, ERROR

# engine.py
class EngineError(Exception):
    kind: Literal["network", "quota", "auth", "moderated", "empty", "other"]
@dataclass
class Reply:
    heard: str         # transcript of user utterance
    text: str          # assistant reply text
    pcm: bytes         # 24 kHz mono s16le
    gesture: str | None = None   # stage 4
class Engine(Protocol):
    async def transcribe(self, pcm16k: bytes) -> str: ...
    async def respond(self, heard: str, history: list[dict]) -> Reply: ...   # publishes "stage" events

# robot.py
class RobotDriver(Protocol):
    def move(self, motor: int, pos: float, speed: int) -> None: ...   # pos 0..10
    def eye_colour(self, r: int, g: int, b: int) -> None: ...
    def connected(self) -> bool: ...
```

Audio formats: mic 16 kHz mono s16le (VAD frames 30 ms = 480 samples); TTS `response_format="pcm"` = 24 kHz mono s16le.

`ohbot` library gotchas (verified 2026-11-21): importing `ohbot.ohbot` auto-runs `init()` (scans ports, plays a silence wav, may call `say`); `init(portName)` is broken (`checkPort` indexes the string); on macOS only ports containing `usb` are scanned; motors: HEADNOD=0 HEADTURN=1 EYETURN=2 LIDBLINK=3 TOPLIP=4 BOTTOMLIP=5 EYETILT=6; `move(m, pos 0-10, spd 0-10)`, `eyeColour(r,g,b)` with 0-10 per channel, `close()` detaches motors. Import it lazily inside `OhbotDriver` only, in a worker thread, never in tests.

## Stage 1: core in terminal (CR after stage)

### Task 1.1: project skeleton, config, events, state
- Create `pyproject.toml` (uv, Python >=3.11, deps above, dev deps pytest pytest-asyncio), `config.toml`, `.env.example`, `personas/*.md`, `ohbotchat/config.py`, `events.py`, `state.py`.
- `config.toml` keys: `[openai] stt_model, chat_model, tts_model, tts_voice, moderation_model, max_tokens=150, timeout_s=20`; `[conversation] persona="dzieci", history_turns=10, mode="ptt"`; `[wake] words=["ohbot","hej ohbot","ochbot","obot","ołbot"]`; `[vad] aggressiveness=2, silence_ms=800, min_speech_ms=300`; `[barge_in] enabled=false, threshold=0.15, hold_ms=300`; `[colours] idle=[0,0,3] listening=[0,10,0] thinking=[0,0,10] speaking=[10,10,10] error=[10,0,0]`.
- Verify current OpenAI model names with Context7 / OpenAI docs before choosing defaults (STT supporting Polish, small chat model, TTS model with a voice that sounds good in Polish, `omni-moderation-latest`). Record choice in config comments.
- Personas in Polish. `dzieci.md`: friendly robot Ohbot, ages 7-12, max 2-3 short sentences, simple words, safe topics, redirects off-topic, never asks for personal data. `pokaz.md`: explains how it works (microphone, speech recognition, language model, speech synthesis, motors), workshop tone. `domowy.md`: relaxed adult assistant.
- Tests: config loads defaults and env key; missing key -> `Config.api_key is None` (no crash); EventBus delivers to multiple subscribers and from a foreign thread; state transition table (e.g. SPEAKING + "stop" -> LISTENING).

### Task 1.2: audio helpers + speaker
- `audio.py`: `level(pcm) -> float 0..1` (RMS normalised), `envelope(pcm, rate, frame_ms=40) -> list[float]` (0..1 per frame, smoothed), `lip_positions(env) -> list[tuple[top, bottom]]` (closed ~ (5,5) baseline, open proportional), `resample` not needed if Speaker plays at 24 kHz.
- `speak.py`: `Speaker.play(pcm, rate=24000, on_frame=callback)` async, plays via `sounddevice.RawOutputStream`, calls `on_frame(index)` per envelope frame for lip sync, `stop()` aborts within 100 ms; output device injectable for tests (fake stream).
- Tests: level of silence = 0, of full-scale sine ~0.7; envelope length = duration/frame; lip positions closed for silence; `stop()` ends `play()` early with fake stream.

### Task 1.3: engine
- `engine.py` contracts above. `engine_chain.py` `ChainEngine(client, config, bus)`: `transcribe` (wav-wrap 16 kHz pcm, language="pl"), `respond`: moderation(heard) -> chat (system = persona file, history, max_tokens) -> moderation(reply) -> TTS pcm; publishes `stage` events `{"name": "stt"|"moderation_in"|"chat"|"moderation_out"|"tts", "status": "start"|"done"}`. Maps openai exceptions: `APIConnectionError/APITimeoutError` -> network, `RateLimitError` with code `insufficient_quota` -> quota, `AuthenticationError` -> auth, flagged -> moderated, blank transcript -> empty.
- Tests with a fake client object (no network): happy path returns Reply; each error mapping; moderated input never calls chat; stages published in order.

### Task 1.4: robot
- `robot.py`: `Robot(driver, config, bus)`: `set_state(State)` -> eye colour; idle loop task: blink every 3-6 s random, small head/eye glance every 5-10 s, only in IDLE; `lips(top, bottom)`; `close()`. All driver calls go through one worker thread (serial is not thread-safe). `OhbotDriver`: lazy import, catches serial errors -> `connected()=False`, reconnect attempt every 5 s publishes `robot` events. `NullDriver` records calls (used in tests and when no robot).
- Tests with NullDriver: state colours from config; blink only when IDLE; lips mapped; driver exception switches to disconnected and publishes event.

### Task 1.5: listener (PTT + VAD core) and fallback phrases
- `listen.py`: `Listener(config, bus, input_stream_factory)`: `start_ptt()/stop_ptt() -> pcm` for push-to-talk; VAD segmenter class `Segmenter(aggressiveness, silence_ms, min_speech_ms)` fed 30 ms frames -> yields utterances (pure, testable). Modes wake/continuous wired in Stage 3, but `Segmenter` and `wake_match(text, words) -> str|None` (returns text after the wake word, normalised: lowercase, strip punctuation/diacritics-insensitive) are built and tested now.
- `fallback.py`: phrases `no_network, not_heard, moderated, error`; `ensure(engine)` generates missing ones via TTS into `cache/fallback/<key>.pcm`; `get(key) -> bytes | None`.
- Tests: Segmenter on synthetic frames (silence/speech pattern via injected `is_speech` fn); min_speech drops clicks; wake_match variants ("Ohbot, jak masz?" -> "jak masz?", "ochbot ile..." ok, "robot" -> None); fallback get/ensure with fake engine and tmp dir.

### Task 1.6: orchestrator, doctor, CLI
- `app.py`: `App(config)` builds bus, engine, speaker, robot, listener; `async handle_utterance(pcm)`: state LISTENING->TRANSCRIBING->THINKING->SPEAKING->IDLE, keeps history (`history_turns`), on EngineError plays fallback + publishes `error` with kind; `stop()` interrupts speaker and returns to LISTENING/IDLE; `new_conversation()` clears history. Lip sync: Speaker on_frame -> robot.lips.
- `doctor.py`: checks from spec section "Diagnostyka" (8 checks), each returns `CheckResult(name, ok, detail_pl, hint_pl)`; interactive steps (mic, speaker, robot motion) take `interactive: bool`. Quota check: 1-token chat request.
- `__main__.py`: `ohbotchat doctor`; `ohbotchat --no-web`: terminal loop "Enter = start, Enter = stop", prints heard/reply. (`ohbotchat` without flag = web, Stage 2.)
- Tests: App with fake engine/speaker/NullDriver: full cycle states order, error -> fallback played + error event, stop() during SPEAKING, history trimmed; doctor with fakes: missing key, quota error, robot absent.

**Stage 1 CR**, then fixes. Manual check by orchestrator: `uv run ohbotchat doctor` non-interactive parts.

## Stage 2: web UI (CR after stage)

### Task 2.1: web server + UI
- `web.py`: FastAPI; `/` serves `web/index.html`; `/ws` streams bus events as JSON and accepts commands `{"cmd": "ptt_start"|"ptt_stop"|"stop"|"new"|"mode", "value"?}`, `{"cmd": "persona", "value"}`, `{"cmd": "barge_in", "enabled", "threshold"}`, `{"cmd": "doctor"}`. Binds 127.0.0.1 only. `__main__` default: start uvicorn + open browser.
- UI (Polish, large font, projector friendly, no build step, vanilla JS): left: subtitles (child bubble = heard, robot bubble = reply); right: pipeline stages Mikrofon -> Tekst -> Filtr -> Model AI -> Filtr -> Głos -> Robot, current stage highlighted, robot state dot in state colour; bottom bar: big hold-to-talk button + Space key, Stop (Esc), Nowa rozmowa, mode select, persona select, barge-in toggle + slider, Diagnostyka (shows doctor results), robot connection badge, mic level meter.
- Tests: FastAPI TestClient websocket: receives published events; commands call App methods (App faked).

**Stage 2 CR**, then fixes.

## Stage 3: listening modes + barge-in (CR after stage)

### Task 3.1
- Listener continuous loop: mic stream -> Segmenter -> utterance -> App. Mode `continuous`: every utterance handled. Mode `wake`: transcribe, `wake_match`; no match -> ignore silently (publish `info` "ignored"), match -> respond to remainder (if remainder empty, robot says short "Słucham?" via fallback phrase `listening` and next utterance is accepted without wake word within 8 s).
- Mic muted (frames dropped) while SPEAKING unless barge-in enabled; barge-in: level > threshold for hold_ms -> App.stop() -> LISTENING; publish `level` events (throttled ~10/s) for UI meter.
- Mode switch at runtime from UI without restart.
- Tests: mode logic with fake mic frames and fake engine; barge-in trigger and non-trigger below threshold/hold; muted while speaking.

**Stage 3 CR**, then fixes.

## Stage 4: gestures (CR after stage)

### Task 4.1
- Chat returns structured output: JSON `{"text": str, "gesture": "none"|"yes"|"no"|"surprised"|"happy"|"thinking"}` (OpenAI structured outputs / json_schema); fall back to plain text if parsing fails. Robot `gesture(name)`: yes = nod twice, no = head turn left/right twice, surprised = eye lids wide + head back, happy = quick head tilt + eye colour flash, thinking = eyes up/aside. Performed at start of speaking, in parallel with lips.
- Persona files get one line about gestures.
- Tests: parse JSON/fallback; gesture -> expected driver call sequence with NullDriver.

**Stage 4 CR**, then fixes.

## Stage 5: Windows packaging

### Task 5.1
- `packaging/windows/build.ps1` (uv sync + PyInstaller one-folder build, includes web/, personas/, config.toml), `ohbotchat --window` option uses pywebview if installed (optional dependency group `desktop`), else browser. `docs/windows.md`: install, `.env` placement next to exe, build steps. Cannot be built on macOS; verify the PyInstaller spec builds on macOS as a smoke test (`--onedir`) if feasible, otherwise document.

## Stage 6: Realtime engine

Deferred by spec ("if latency bothers"). Not implemented now; `Engine` protocol keeps the door open.

## Final

- Full audit/CR on model `fable`, fixes, full test run, `uv run ohbotchat doctor`, start web UI for the user.
