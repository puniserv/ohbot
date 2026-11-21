import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .doctor import run_doctor

log = logging.getLogger(__name__)

HOST = "127.0.0.1"  # the UI has no auth, so it must never listen on a network interface
DEFAULT_PORT = 8765
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})
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


def _same_origin(ws: WebSocket, allowed_hosts) -> bool:
    """Any web page can open ws://127.0.0.1, so Origin must match; the Host check also stops DNS rebinding."""
    host = ws.headers.get("host", "")
    return host.partition(":")[0] in allowed_hosts and ws.headers.get("origin") == f"http://{host}"


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
                    command(msg)
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

    api.mount("/", StaticFiles(directory=web_dir, html=True), name="static")
    return api
