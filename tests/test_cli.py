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

    for call in (NoKeyEngine().transcribe(b""), NoKeyEngine().respond("", []), NoKeyEngine().synthesize("x", publish=False), NoKeyEngine().say("x")):
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


def _install_webview(monkeypatch, start):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(create_window=lambda title, url: None, start=start))


def test_window_ctrl_c_before_the_server_is_ready_still_stops_it(monkeypatch):
    # regression: should_exit was only set once ready, so the worker never ended and join() hung
    import _thread
    import time
    from types import SimpleNamespace

    import ohbotchat.__main__ as cli

    calls = []

    async def web(port, on_ready):
        _thread.interrupt_main()
        await asyncio.sleep(0.3)
        server = SimpleNamespace(should_exit=False)
        on_ready("http://127.0.0.1:1/", server)
        deadline = time.monotonic() + 3
        while not server.should_exit and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        calls.append("stopped" if server.should_exit else "hung")
        return 0

    _install_webview(monkeypatch, start=lambda: calls.append("window"))
    monkeypatch.setattr(cli, "_web", web)

    assert main(["--window"]) == 0
    assert calls == ["stopped"]


def test_window_that_fails_to_open_falls_back_to_the_browser(monkeypatch, caplog):
    from types import SimpleNamespace

    import ohbotchat.__main__ as cli

    calls, box = [], {}

    async def web(port, on_ready):
        box["server"] = SimpleNamespace(should_exit=False)
        on_ready("http://127.0.0.1:1/", box["server"])
        while not box["server"].should_exit:
            await asyncio.sleep(0.01)
        calls.append("stopped")
        return 0

    def no_webview2():
        raise RuntimeError("WebView2 missing")

    def browser(url):
        calls.append(url)
        box["server"].should_exit = True  # stands in for the user pressing Ctrl+C later

    _install_webview(monkeypatch, start=no_webview2)
    monkeypatch.setattr(cli, "_web", web)
    monkeypatch.setattr(cli.webbrowser, "open", browser)

    assert main(["--window"]) == 0
    assert calls == ["http://127.0.0.1:1/", "stopped"]
    assert "WebView2" in caplog.text


class _WebApp:
    def __init__(self, config, engine=None):
        from ohbotchat.events import EventBus

        self.config, self.bus, self.log, self.closed = config, EventBus(), [], False

    async def start(self):
        self.log.append("start")

    async def close(self):
        self.log.append("close")


def _web_apps(monkeypatch):
    made = []

    def make(config, engine=None):
        made.append(_WebApp(config, engine))
        return made[-1]

    monkeypatch.setattr("ohbotchat.app.App", make)
    return made


async def test_web_on_a_busy_port_neither_starts_the_robot_nor_claims_to_run(monkeypatch, capsys):
    import socket

    import pytest

    from ohbotchat.__main__ import _web

    made = _web_apps(monkeypatch)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        with pytest.raises(SystemExit):
            await _web(busy.getsockname()[1], lambda url, server: pytest.fail("opened the page"))
    assert "start" not in made[0].log and "działa" not in capsys.readouterr().out


async def test_web_starts_the_app_and_announces_only_after_binding(monkeypatch, capsys):
    import socket

    from ohbotchat.__main__ import _web

    made = _web_apps(monkeypatch)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    seen = []

    def on_ready(url, server):
        seen.append((list(made[0].log), url, server.started))
        server.should_exit = True

    assert await _web(port, on_ready) == 0
    assert seen == [(["start"], f"http://127.0.0.1:{port}/", True)]
    assert "działa" in capsys.readouterr().out and made[0].log[-1] == "close"


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_web_closed_during_start_neither_announces_nor_opens_the_page(monkeypatch, capsys):
    import pytest

    from ohbotchat.__main__ import _web

    made = _web_apps(monkeypatch)

    async def start_then_ctrl_c():
        made[0].closed = True  # the lifespan shutdown ran while the robot was connecting

    monkeypatch.setattr(_WebApp, "start", lambda self: start_then_ctrl_c())
    await asyncio.wait_for(_web(_free_port(), lambda url, server: pytest.fail("opened the page")), 5)
    assert "działa" not in capsys.readouterr().out


async def test_web_exits_with_1_when_the_app_fails_to_start(monkeypatch):
    import pytest

    from ohbotchat.__main__ import _web

    _web_apps(monkeypatch)

    async def broken(self):
        raise OSError("no robot port")

    monkeypatch.setattr(_WebApp, "start", broken)
    assert await asyncio.wait_for(_web(_free_port(), lambda url, server: pytest.fail("opened the page")), 5) == 1
