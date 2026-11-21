import argparse
import asyncio
import logging
import os
import sys
import threading
import webbrowser

from .config import app_root, load_config
from .doctor import CheckResult, run_doctor
from .engine import EngineError
from .web import DEFAULT_PORT

log = logging.getLogger(__name__)

_MARK = {True: "✔", False: "✘", None: "-"}
_ERROR_HINTS = {
    "quota": "Brak środków na koncie OpenAI: doładuj konto.",
    "auth": "Klucz API odrzucony: sprawdź OPENAI_API_KEY w .env.",
}


def format_result(r: CheckResult) -> str:
    line = f"{_MARK[r.ok]} {r.name}: {r.detail}"
    return f"{line}\n    -> {r.hint}" if r.hint and r.ok is not True else line


async def _doctor() -> int:
    results = await run_doctor(load_config(), interactive=sys.stdin.isatty())
    for r in results:
        print(format_result(r))
    return 0 if all(r.ok is not False for r in results) else 1


async def _print_events(q: asyncio.Queue) -> None:
    while True:
        e = await q.get()
        if e.type == "heard":
            print(f"Usłyszałem: {e.data['text']}")
        elif e.type == "reply":
            print(f"Ohbot: {e.data['text']}")
        elif e.type == "error":
            print(f"Błąd ({e.data['kind']}): {_ERROR_HINTS.get(e.data['kind'], e.data['message'])}")
        elif e.type == "robot" and not e.data["connected"]:
            print("Robot niepodłączony, działam bez ruchu.")


async def terminal_loop(app, lines: asyncio.Queue) -> None:
    """Enter starts recording, the next Enter stops it; EOF (None) or cancellation ends the loop."""
    printer = asyncio.create_task(_print_events(app.bus.subscribe()))
    try:
        await app.start()
        print("Enter = start nagrywania, Enter = stop, Ctrl+C = wyjście")
        while await lines.get() is not None:
            app.ptt_start()
            print("Nagrywam...")
            if await lines.get() is None:
                break
            if task := app.ptt_stop():
                await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)  # let the printer flush the last events
    finally:
        printer.cancel()
        await app.close()


def _stdin_lines(loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
    # a daemon thread, because a blocked input() in an executor would hang exit on Ctrl+C
    q: asyncio.Queue = asyncio.Queue()

    def read() -> None:
        while True:
            line = sys.stdin.readline()
            loop.call_soon_threadsafe(q.put_nowait, line if line else None)
            if not line:
                return

    threading.Thread(target=read, daemon=True).start()
    return q


async def _terminal() -> int:
    config = load_config()
    if not config.api_key:
        print("Brak OPENAI_API_KEY. Uruchom `ohbotchat doctor`, żeby zobaczyć co poprawić.")
        return 1
    from .app import App

    await terminal_loop(App(config), _stdin_lines(asyncio.get_running_loop()))
    return 0


class NoKeyEngine:
    """Lets the web UI start without a key; every turn then reports the problem on screen."""

    async def _fail(self, *args, **kwargs):
        raise EngineError("nokey", "Brak klucza API – dodaj OPENAI_API_KEY do pliku .env")

    transcribe = respond = synthesize = _fail


async def _web(port: int, on_ready) -> int:
    import uvicorn

    from .app import App
    from .web import create_app, server_config

    config = load_config()
    app = App(config, engine=None if config.api_key else NoKeyEngine())
    server = uvicorn.Server(server_config(create_app(app), port))

    async def ready() -> None:
        while not server.started:
            await asyncio.sleep(0.1)
        on_ready(f"http://127.0.0.1:{port}/", server)

    opener = None
    try:
        await app.start()
        opener = asyncio.create_task(ready())
        print(f"Ohbot działa na http://127.0.0.1:{port}/ (Ctrl+C = wyjście)")
        await server.serve()
    finally:
        if opener:
            opener.cancel()
        await app.close()
    return 0


def _window(port: int) -> int:
    try:
        import webview
    except ImportError:
        log.warning("Brak pywebview (uv sync --group desktop), otwieram przeglądarkę")
        return asyncio.run(_web(port, lambda url, server: webbrowser.open(url)))

    # pywebview must own the main thread, so the server runs in a second one
    # uvicorn installs no signal handlers off the main thread, so Ctrl+C reaches the server only through `stop`
    ready, stop, box = threading.Event(), threading.Event(), {}

    def on_ready(url, server) -> None:
        box.update(url=url, server=server)
        server.should_exit = stop.is_set()
        ready.set()

    worker = threading.Thread(target=lambda: box.update(code=asyncio.run(_web(port, on_ready))))
    worker.start()
    try:
        while not ready.wait(0.1):
            if not worker.is_alive():
                return box.get("code", 1)
        try:
            webview.create_window("Ohbot", box["url"])
            webview.start()
        except Exception:
            log.exception("Nie mogę otworzyć okna (brak WebView2?), otwieram przeglądarkę")
            webbrowser.open(box["url"])
            while worker.is_alive():
                worker.join(0.5)  # a timeout keeps Ctrl+C deliverable on Windows
    finally:
        stop.set()
        if "server" in box:
            box["server"].should_exit = True
        worker.join()
    return 0


def main(argv: list[str] | None = None) -> int:
    if getattr(sys, "frozen", False):
        os.chdir(app_root())  # the ohbot library writes ohbotData/ into the cwd
    parser = argparse.ArgumentParser(prog="ohbotchat", description="Rozmowa głosowa z robotem Ohbot")
    parser.add_argument("command", nargs="?", choices=["doctor"])
    parser.add_argument("--no-web", action="store_true", help="pętla w terminalu zamiast strony w przeglądarce")
    parser.add_argument("--window", action="store_true", help="osobne okno zamiast przeglądarki (wymaga pywebview)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port strony (tylko 127.0.0.1)")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return asyncio.run(_doctor())
        if args.no_web:
            return asyncio.run(_terminal())
        if args.window:
            return _window(args.port)
        return asyncio.run(_web(args.port, lambda url, server: webbrowser.open(url)))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
