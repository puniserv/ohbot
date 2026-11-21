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
