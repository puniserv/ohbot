import asyncio

from ohbotchat.__main__ import format_result, main, terminal_loop
from ohbotchat.doctor import CheckResult


def test_format_result():
    assert format_result(CheckResult("A", True, "ok")).startswith("✔ A")
    bad = format_result(CheckResult("B", False, "źle", "napraw to"))
    assert bad.startswith("✘ B") and "napraw to" in bad
    assert format_result(CheckResult("C", None, "pominięto")).startswith("- C")


async def test_no_key_engine_reports_nokey_error_instead_of_crashing():
    import pytest

    from ohbotchat.__main__ import NoKeyEngine
    from ohbotchat.engine import EngineError

    for call in (NoKeyEngine().transcribe(b""), NoKeyEngine().respond("", []), NoKeyEngine().synthesize("x", publish=False)):
        with pytest.raises(EngineError) as e:
            await call
        assert e.value.kind == "nokey"


class FakeApp:
    def __init__(self):
        from ohbotchat.events import EventBus

        self.bus, self.log = EventBus(), []

    async def start(self):
        self.log.append("start")

    def ptt_start(self):
        self.log.append("ptt_start")

    def ptt_stop(self):
        self.log.append("ptt_stop")

        async def turn():
            self.bus.publish("heard", text="Cześć")
            self.bus.publish("reply", text="Hej!")

        return asyncio.create_task(turn())

    async def close(self):
        self.log.append("close")


async def test_terminal_loop_enter_cycle_and_close_on_eof(capsys):
    app, lines = FakeApp(), asyncio.Queue()
    for item in ("", "", None):
        lines.put_nowait(item)
    await terminal_loop(app, lines)
    assert app.log == ["start", "ptt_start", "ptt_stop", "close"]
    out = capsys.readouterr().out
    assert "Cześć" in out and "Hej!" in out


async def test_terminal_loop_closes_on_cancel():
    app, lines = FakeApp(), asyncio.Queue()
    task = asyncio.create_task(terminal_loop(app, lines))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert app.log[-1] == "close"


def _fake_web(calls):
    from types import SimpleNamespace

    async def web(port, on_ready):
        server = SimpleNamespace(should_exit=False)
        on_ready(f"http://127.0.0.1:{port}/", server)
        while not server.should_exit:
            await asyncio.sleep(0.01)
        calls.append("stopped")
        return 0

    return web


def test_window_opens_a_native_window_and_stops_the_server_when_it_closes(monkeypatch):
    import sys
    from types import SimpleNamespace

    import ohbotchat.__main__ as cli

    calls = []
    webview = SimpleNamespace(create_window=lambda title, url: calls.append(url), start=lambda: calls.append("start"))
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setattr(cli, "_web", _fake_web(calls))
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: calls.append("browser"))

    assert main(["--window", "--port", "9999"]) == 0
    assert calls == ["http://127.0.0.1:9999/", "start", "stopped"]


def test_window_without_pywebview_falls_back_to_the_browser_with_a_warning(monkeypatch, caplog):
    import sys

    import ohbotchat.__main__ as cli

    calls = []

    async def web(port, on_ready):
        on_ready(f"http://127.0.0.1:{port}/", None)
        return 0

    monkeypatch.setitem(sys.modules, "webview", None)
    monkeypatch.setattr(cli, "_web", web)
    monkeypatch.setattr(cli.webbrowser, "open", calls.append)

    assert main(["--window", "--port", "9999"]) == 0
    assert calls == ["http://127.0.0.1:9999/"]
    assert "pywebview" in caplog.text


def test_frozen_build_works_in_the_exe_folder_so_ohbotdata_lands_there(monkeypatch, tmp_path):
    import os
    import sys

    import pytest

    monkeypatch.chdir(os.getcwd())  # restores the cwd after the test
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "ohbot-chat.exe"))
    with pytest.raises(SystemExit):
        main(["--help"])
    assert os.getcwd() == str(tmp_path.resolve())
