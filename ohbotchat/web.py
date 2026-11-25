import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import partial

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import MODES, VOICES, editable_settings, save_settings, validate_settings, write_atomic
from .doctor import run_doctor
from .engine import GESTURES, EngineError

log = logging.getLogger(__name__)

HOST = "127.0.0.1"  # the UI has no auth, so it must never listen on a network interface
DEFAULT_PORT = 8765
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})
MANUAL_GESTURES = tuple(g for g in GESTURES if g != "none")
MOTORS = range(7)
MANUAL = ("gesture", "motor", "close_mouth", "neutral", "calibrate", "say", "voice_sample")
MAX_SAY = 300
FILLER_ERRORS = {
    "moderated": "Fraza „{}” nie przeszła filtra bezpieczeństwa – zmień ją",
    "nokey": "Brak klucza API – frazy „{}” nie da się sprawdzić",
    "other": "Nie udało się sprawdzić frazy „{}” (brak internetu?) – spróbuj ponownie",
}
MAX_PERSONA = 8000
BUSY = "Robot jest zajęty – spróbuj, gdy skończy"
ASLEEP = "Robot śpi – kliknij „Obudź”"


class _RunningRobot:
    """Doctor must not reconnect the serial port the app already holds, so it only reports its state."""

    def __init__(self, robot) -> None:
        self._robot = robot

    def connect(self) -> bool:
        return self._robot.connected


def _running_camera(app):
    """The app's vision thread holds the camera and a second open fails on Windows, so doctor only reports its state."""
    def check(_config):
        if app.vision.ok is None:  # before the first frame
            return None, "kamera uruchamia się"
        return app.vision.ok, "kamera działa" if app.vision.ok else "kamera niedostępna w działającej aplikacji"

    return check


async def _run_doctor(app) -> list:
    return await asyncio.to_thread(
        lambda: asyncio.run(
            run_doctor(app.config, interactive=False, driver=_RunningRobot(app.robot), camera=_running_camera(app))
        )
    )


def server_config(web_app: FastAPI, port: int = DEFAULT_PORT) -> uvicorn.Config:
    # an open camera preview is an endless response; without the timeout Ctrl+C would wait for it forever
    return uvicorn.Config(web_app, host=HOST, port=port, log_level="warning", timeout_graceful_shutdown=2)


async def mjpeg(vision, disconnected: Callable[[], Awaitable[bool]], interval: float = 0.1):
    """multipart/x-mixed-replace parts of the latest camera frame, each new frame once."""
    vision.viewers += 1
    try:
        last = None
        while not await disconnected():
            jpeg = vision.jpeg
            if jpeg is not None and jpeg is not last:
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(interval)
    finally:
        vision.viewers -= 1


def _same_origin(conn: WebSocket | Request, allowed_hosts) -> bool:
    """Any web page can open ws://127.0.0.1 or POST to it, so Origin must match; the Host check also stops DNS rebinding."""
    host = conn.headers.get("host", "")
    return host.partition(":")[0] in allowed_hosts and conn.headers.get("origin") == f"http://{host}"


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


async def _json(request: Request):
    try:
        return await request.json()
    except ValueError:
        raise HTTPException(400, "invalid JSON") from None


def create_app(app, doctor: Callable[..., Awaitable[list]] = _run_doctor, allowed_hosts=ALLOWED_HOSTS) -> FastAPI:
    """`app` is the ohbotchat App (or a fake with the same surface); its close() runs on server shutdown."""
    cfg = app.config
    web_dir = cfg.root / "web"

    @asynccontextmanager
    async def lifespan(_):
        yield
        # uvicorn re-raises SIGINT after serve(), which cancels the caller's own cleanup at its first await
        await app.close()

    api = FastAPI(lifespan=lifespan)
    tasks: set[asyncio.Task] = set()
    sleeps: set[asyncio.Task] = set()  # keeps a reference, else the event loop may drop the running task
    panels: set[WebSocket] = set()  # connections with the motion test panel open

    def set_panel(ws: WebSocket, open_: bool) -> None:
        (panels.add if open_ else panels.discard)(ws)
        app.robot.hold_idle = bool(panels)

    def guard(request: Request) -> None:
        # a same-origin GET sends no Origin, so reads check the Host only; writes need both
        if request.method == "GET":
            ok = request.headers.get("host", "").partition(":")[0] in allowed_hosts
        else:
            ok = _same_origin(request, allowed_hosts)
        if not ok:
            raise HTTPException(403)

    def personas() -> list[str]:
        return sorted(p.stem for p in (cfg.root / "personas").glob("*.md"))

    def settings() -> dict:
        return {
            "mode": cfg.conversation.mode,
            "persona": cfg.conversation.persona,
            "barge_in": {"enabled": cfg.barge_in.enabled, "threshold": cfg.barge_in.threshold},
        }

    def snapshot() -> dict:
        return {
            "state": str(app.state),
            "personas": personas(),
            "robot": app.robot.connected,
            "has_key": bool(cfg.api_key),
            "colours": asdict(cfg.colours),
            "mic_error": app.mic_error,
            "camera": app.vision.ok,
            "asleep": app.asleep,
            **settings(),
        }

    async def run_doctor_task() -> None:
        try:
            results = await doctor(app)
            app.bus.publish("doctor", results=[asdict(r) for r in results])
        except Exception as e:
            log.exception("doctor failed")
            app.bus.publish("error", kind="other", message=f"Diagnostyka nie powiodła się: {e}")

    def command(msg: dict, ws: WebSocket) -> None:
        cmd = msg.get("cmd")
        if app.asleep and cmd in ("ptt_start", "new", *MANUAL):
            app.bus.publish("error", kind="other", message=ASLEEP)
            return
        if cmd == "ptt_start":
            app.ptt_start()
        elif cmd == "ptt_stop":
            app.ptt_stop()  # the turn runs in App's own task; awaiting it here would block "stop"
        elif cmd == "stop":
            app.stop()
        elif cmd == "new":
            app.new_conversation()
            app.greet()
        elif cmd == "test_panel":
            if not isinstance(msg.get("open"), bool):
                raise ValueError("bad test_panel")
            set_panel(ws, msg["open"])
        elif cmd in MANUAL:
            manual(cmd, msg)
        elif cmd == "sleep":
            if not isinstance(msg.get("asleep"), bool):
                raise ValueError("bad sleep")
            task = asyncio.create_task(app.set_asleep(msg["asleep"]))
            sleeps.add(task)
            task.add_done_callback(sleeps.discard)
        elif cmd == "mode" and msg.get("value") in MODES:
            app.set_mode(msg["value"])
            app.bus.publish("settings", **settings())
        elif cmd == "persona" and msg.get("value") in personas():
            cfg.conversation.persona = msg["value"]
            app.new_conversation()
            app.bus.publish("settings", **settings())
        elif cmd == "barge_in":
            enabled, threshold = msg.get("enabled"), msg.get("threshold")
            if not isinstance(enabled, bool) or isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise ValueError("bad barge_in")
            cfg.barge_in.enabled, cfg.barge_in.threshold = enabled, float(threshold)
            app.bus.publish("settings", **settings())
        elif cmd == "doctor":
            if tasks:
                return
            task = asyncio.create_task(run_doctor_task())
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        else:
            raise ValueError(f"unknown command {cmd!r}")

    def manual(cmd: str, msg: dict) -> None:
        """Test panel commands; validated first so a bad one is ignored, not reported as busy."""
        if cmd == "gesture":
            if msg.get("name") not in MANUAL_GESTURES:
                raise ValueError("bad gesture")
            run = partial(app.manual_gesture, msg["name"])
        elif cmd == "motor":
            motor, pos = msg.get("motor"), msg.get("pos")
            if isinstance(motor, bool) or motor not in MOTORS or not _number(pos):
                raise ValueError("bad motor")
            run = partial(app.robot.move, motor, min(10, max(0, pos)))
        elif cmd == "close_mouth":
            run = app.robot.close_mouth
        elif cmd == "neutral":
            run = app.robot.neutral
        elif cmd == "calibrate":
            def run() -> None:
                app.bus.publish("error", kind="other", message=app.calibrate())  # "other" errors are the UI's plain toast
        elif cmd == "say":
            text = msg.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_SAY:
                raise ValueError("bad say")
            run = partial(app.say, text.strip())
        else:
            if msg.get("voice") not in VOICES:
                raise ValueError("bad voice")
            run = partial(app.voice_sample, msg["voice"])
        if not app.ready():
            app.bus.publish("error", kind="other", message=BUSY)
            return
        run()

    async def pump(ws: WebSocket, q: asyncio.Queue) -> None:
        while True:
            e = await q.get()
            try:
                await ws.send_json({"type": e.type, "data": e.data})
            except Exception:
                log.warning("websocket send failed, dropping the stream")
                return

    @api.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        if not _same_origin(ws, allowed_hosts):
            await ws.close(code=1008)
            return
        await ws.accept()
        q = app.bus.subscribe()
        sender = None
        try:
            await ws.send_json({"type": "snapshot", "data": snapshot()})
            sender = asyncio.create_task(pump(ws, q))
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                    if not isinstance(msg, dict):
                        raise ValueError("not an object")
                    command(msg, ws)
                except ValueError as e:  # includes JSONDecodeError
                    log.warning("ignored websocket command %.100r: %s", text, e)
                except Exception:
                    log.exception("websocket command %.100r failed", text)
        except WebSocketDisconnect:
            pass
        finally:
            if sender:
                sender.cancel()
            app.bus.unsubscribe(q)
            if app.listener.recording:
                app.ptt_abort()
            set_panel(ws, False)  # a closed tab cannot close its test panel

    @api.get("/camera.mjpg", dependencies=[Depends(guard)])
    def camera(request: Request) -> StreamingResponse:
        return StreamingResponse(mjpeg(app.vision, request.is_disconnected), media_type="multipart/x-mixed-replace; boundary=frame")

    @api.get("/api/settings", dependencies=[Depends(guard)])
    def get_settings() -> dict:
        return {"values": editable_settings(cfg), "voices": VOICES, "gestures": MANUAL_GESTURES}

    @api.post("/api/settings", dependencies=[Depends(guard)])
    async def post_settings(request: Request) -> dict:
        try:
            values = validate_settings(await _json(request))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        if phrases := values.get("conversation", {}).get("filler_phrases"):
            try:
                await app.prepare_fillers(phrases)
            except EngineError as e:
                raise HTTPException(400, FILLER_ERRORS.get(e.kind, FILLER_ERRORS["other"]).format(e)) from None
        app.apply_settings(values)
        app.bus.publish("settings", **settings())
        return {"values": editable_settings(cfg)}

    @api.post("/api/settings/save", dependencies=[Depends(guard)])
    async def save() -> dict:  # async runs it on the event loop, so two saves cannot interleave
        save_settings(cfg.root / "config.toml", editable_settings(cfg))
        return {"ok": True}

    def persona_path(name: str):
        if name not in personas():  # existing files only, so no path can be smuggled in
            raise HTTPException(404)
        return cfg.root / "personas" / f"{name}.md"

    @api.get("/api/persona/{name}", dependencies=[Depends(guard)])
    def get_persona(name: str) -> dict:
        return {"text": persona_path(name).read_text(encoding="utf-8")}

    @api.put("/api/persona/{name}", dependencies=[Depends(guard)])
    async def put_persona(name: str, request: Request) -> dict:
        path = persona_path(name)
        body = await _json(request)
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_PERSONA:
            raise HTTPException(400, f"expected 1..{MAX_PERSONA} characters")
        write_atomic(path, text)
        return {"ok": True}

    api.mount("/", StaticFiles(directory=web_dir, html=True), name="static")
    return api
