import asyncio

import pytest
from fastapi.testclient import TestClient

from ohbotchat.config import load_config
from ohbotchat.doctor import CheckResult
from ohbotchat.events import EventBus
from ohbotchat.web import DEFAULT_PORT, create_app, server_config


class FakeRobot:
    connected = True


class FakeApp:
    def __init__(self):
        self.config = load_config(env={"OPENAI_API_KEY": "sk-test"})
        self.bus = EventBus()
        self.state = "idle"
        self.robot = FakeRobot()
        self.calls = []

    def ptt_start(self):
        self.calls.append("ptt_start")

    def ptt_stop(self):
        self.calls.append("ptt_stop")

    def stop(self):
        self.calls.append("stop")

    def new_conversation(self):
        self.calls.append("new")


async def fake_doctor(app):
    return [CheckResult("Klucz", True, "jest"), CheckResult("Mikrofon", False, "brak", "podłącz")]


@pytest.fixture
def app():
    return FakeApp()


@pytest.fixture
def client(app):
    return TestClient(create_app(app, doctor=fake_doctor))


def snapshot(ws):
    msg = ws.receive_json()
    assert msg["type"] == "snapshot"
    return msg["data"]


def sync(ws, app):
    """Round-trips an event so earlier commands are processed."""
    app.bus.publish("info", text="sync")
    while (msg := ws.receive_json())["type"] != "info":
        pass


def test_index_and_static_are_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Ohbot" in r.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_snapshot_contents(client, app):
    with client.websocket_connect("/ws") as ws:
        s = snapshot(ws)
    assert s["state"] == "idle" and s["mode"] == "ptt" and s["persona"] == "dzieci"
    assert {"dzieci", "pokaz", "domowy"} <= set(s["personas"])
    assert s["robot"] is True and s["has_key"] is True
    assert s["barge_in"] == {"enabled": False, "threshold": app.config.barge_in.threshold}
    assert s["colours"]["listening"] == app.config.colours.listening


def test_missing_key_is_reported_in_snapshot(app):
    app.config.api_key = None
    with TestClient(create_app(app, doctor=fake_doctor)).websocket_connect("/ws") as ws:
        assert snapshot(ws)["has_key"] is False


def test_bus_events_are_streamed(client, app):
    with client.websocket_connect("/ws") as ws:
        snapshot(ws)
        app.bus.publish("heard", text="Cześć")
        assert ws.receive_json() == {"type": "heard", "data": {"text": "Cześć"}}


@pytest.mark.parametrize(
    "cmd,call",
    [("ptt_start", "ptt_start"), ("ptt_stop", "ptt_stop"), ("stop", "stop"), ("new", "new_conversation")],
)
def test_simple_commands_call_app(client, app, cmd, call):
    with client.websocket_connect("/ws") as ws:
        snapshot(ws)
        ws.send_json({"cmd": cmd})
        sync(ws, app)
    assert app.calls == [cmd if cmd != "new" else "new"]


def test_mode_persona_and_barge_in_are_stored_and_broadcast(client, app):
    with client.websocket_connect("/ws") as ws:
        snapshot(ws)
        ws.send_json({"cmd": "mode", "value": "wake"})
        assert ws.receive_json()["data"]["mode"] == "wake"
        ws.send_json({"cmd": "persona", "value": "pokaz"})
        assert ws.receive_json()["data"]["persona"] == "pokaz"
        ws.send_json({"cmd": "barge_in", "enabled": True, "threshold": 0.4})
        assert ws.receive_json()["data"]["barge_in"] == {"enabled": True, "threshold": 0.4}
    assert (app.config.conversation.mode, app.config.conversation.persona) == ("wake", "pokaz")
    assert app.config.barge_in.enabled and app.config.barge_in.threshold == 0.4
    assert app.calls == ["new"]  # persona switch starts a fresh conversation


@pytest.mark.parametrize(
    "msg",
    [
        {"cmd": "mode", "value": "bogus"},
        {"cmd": "persona", "value": "../../etc/passwd"},
        {"cmd": "barge_in", "enabled": "x", "threshold": "y"},
        {"cmd": "barge_in", "enabled": True, "threshold": 7},
        {"cmd": "nope"},
        {"no_cmd": 1},
        [1, 2],
        "str",
        None,
    ],
)
def test_invalid_commands_are_ignored(client, app, msg):
    with client.websocket_connect("/ws") as ws:
        snapshot(ws)
        ws.send_json(msg)
        sync(ws, app)
    assert app.config.conversation.mode == "ptt" and app.config.conversation.persona == "dzieci"
    assert not app.config.barge_in.enabled and app.config.barge_in.threshold == 0.15
    assert app.calls == []


def test_malformed_json_keeps_socket_open(client, app):
    with client.websocket_connect("/ws") as ws:
        snapshot(ws)
        ws.send_text("{not json")
        ws.send_json({"cmd": "stop"})
        sync(ws, app)
    assert app.calls == ["stop"]


def test_doctor_results_are_published_as_event(client, app):
    with client.websocket_connect("/ws") as ws:
        snapshot(ws)
        ws.send_json({"cmd": "doctor"})
        msg = ws.receive_json()
    assert msg["type"] == "doctor"
    assert msg["data"]["results"] == [
        {"name": "Klucz", "ok": True, "detail": "jest", "hint": ""},
        {"name": "Mikrofon", "ok": False, "detail": "brak", "hint": "podłącz"},
    ]


def test_disconnect_unsubscribes(client, app):
    with client.websocket_connect("/ws") as ws:
        snapshot(ws)
        assert len(app.bus._subs) == 1
    client.get("/")  # lets the server side finish the handler
    assert app.bus._subs == []


def test_server_binds_localhost_only_on_given_port(app):
    c = server_config(create_app(app), 8799)
    assert (c.host, c.port) == ("127.0.0.1", 8799)
    assert server_config(create_app(app)).port == DEFAULT_PORT == 8765
