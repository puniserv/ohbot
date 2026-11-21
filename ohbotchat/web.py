import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .doctor import run_doctor

log = logging.getLogger(__name__)

HOST = "127.0.0.1"  # the UI has no auth, so it must never listen on a network interface
DEFAULT_PORT = 8765
MODES = ("ptt", "wake", "continuous")


class _RunningRobot:
    """Doctor must not reconnect the serial port the app already holds, so it only reports its state."""

    def __init__(self, robot) -> None:
        self._robot = robot

    def connect(self) -> bool:
        return self._robot.connected


async def _run_doctor(app) -> list:
    return await asyncio.to_thread(
        lambda: asyncio.run(run_doctor(app.config, interactive=False, driver=_RunningRobot(app.robot)))
    )


def server_config(web_app: FastAPI, port: int = DEFAULT_PORT) -> uvicorn.Config:
    return uvicorn.Config(web_app, host=HOST, port=port, log_level="warning")


def create_app(app, doctor: Callable[..., Awaitable[list]] = _run_doctor) -> FastAPI:
    """`app` is the ohbotchat App (or a fake with the same surface)."""
    cfg = app.config
    web_dir = cfg.root / "web"
    api = FastAPI()
    tasks: set[asyncio.Task] = set()

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
            **settings(),
        }

    async def run_doctor_task() -> None:
        try:
            results = await doctor(app)
            app.bus.publish("doctor", results=[asdict(r) for r in results])
        except Exception as e:
            log.exception("doctor failed")
            app.bus.publish("error", kind="other", message=f"Diagnostyka nie powiodła się: {e}")

    def command(msg: dict) -> None:
        cmd = msg.get("cmd")
        if cmd == "ptt_start":
            app.ptt_start()
        elif cmd == "ptt_stop":
            app.ptt_stop()  # the turn runs in App's own task; awaiting it here would block "stop"
        elif cmd == "stop":
            app.stop()
        elif cmd == "new":
            app.new_conversation()
        elif cmd == "mode" and msg.get("value") in MODES:
            cfg.conversation.mode = msg["value"]
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
            task = asyncio.create_task(run_doctor_task())
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        else:
            raise ValueError(f"unknown command {cmd!r}")

    async def pump(ws: WebSocket, q: asyncio.Queue) -> None:
        while True:
            e = await q.get()
            await ws.send_json({"type": e.type, "data": e.data})

    @api.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
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
                    command(msg)
                except Exception as e:
                    log.warning("ignored websocket command %.100r: %s", text, e)
        except WebSocketDisconnect:
            pass
        finally:
            if sender:
                sender.cancel()
            app.bus.unsubscribe(q)

    @api.get("/")
    async def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    api.mount("/", StaticFiles(directory=web_dir), name="static")
    return api
