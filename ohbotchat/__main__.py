import argparse
import asyncio
import sys
import threading

from .config import load_config
from .doctor import CheckResult, run_doctor

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
            await app.ptt_stop()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ohbotchat", description="Rozmowa głosowa z robotem Ohbot")
    parser.add_argument("command", nargs="?", choices=["doctor"])
    parser.add_argument("--no-web", action="store_true", help="pętla w terminalu zamiast strony w przeglądarce")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return asyncio.run(_doctor())
        if args.no_web:
            return asyncio.run(_terminal())
    except KeyboardInterrupt:
        print()
        return 0
    print("Interfejs w przeglądarce jeszcze nie jest gotowy; użyj `ohbotchat --no-web`.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
