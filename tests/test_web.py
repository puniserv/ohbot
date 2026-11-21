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
        self.listener = type("L", (), {"recording": False})()
        self.closed = 0
        self.mic_error = None

    def set_mode(self, mode):
        self.calls.append(f"mode:{mode}")
        self.config.conversation.mode = mode

    def ptt_abort(self):
        self.calls.append("ptt_abort")

    async def close(self):
        self.closed += 1

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
    return TestClient(create_app(app, doctor=fake_doctor, allowed_hosts={"testserver"}))


ORIGIN = {"origin": "http://testserver"}


def connect(client, headers=ORIGIN):
    return client.websocket_connect("/ws", headers=headers)


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
    with connect(client) as ws:
        s = snapshot(ws)
    assert s["state"] == "idle" and s["mode"] == "ptt" and s["persona"] == "dzieci"
    assert {"dzieci", "pokaz", "domowy"} <= set(s["personas"])
    assert s["robot"] is True and s["has_key"] is True
    assert s["barge_in"] == {"enabled": False, "threshold": app.config.barge_in.threshold}
    assert s["colours"]["listening"] == app.config.colours.listening
    assert s["mic_error"] is None


def test_snapshot_carries_a_mic_error_raised_before_the_client_connected(client, app):
    app.mic_error = "Nie mogę włączyć mikrofonu"
    with connect(client) as ws:
        assert snapshot(ws)["mic_error"] == "Nie mogę włączyć mikrofonu"


def test_missing_key_is_reported_in_snapshot(app):
    app.config.api_key = None
    with connect(TestClient(create_app(app, doctor=fake_doctor, allowed_hosts={"testserver"}))) as ws:
        assert snapshot(ws)["has_key"] is False


def test_bus_events_are_streamed(client, app):
    with connect(client) as ws:
        snapshot(ws)
        app.bus.publish("heard", text="Cześć")
        assert ws.receive_json() == {"type": "heard", "data": {"text": "Cześć"}}


@pytest.mark.parametrize(
    "cmd,call",
    [("ptt_start", "ptt_start"), ("ptt_stop", "ptt_stop"), ("stop", "stop"), ("new", "new")],
)
def test_simple_commands_call_app(client, app, cmd, call):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json({"cmd": cmd})
        sync(ws, app)
    assert app.calls == [call]


def test_mode_persona_and_barge_in_are_stored_and_broadcast(client, app):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json({"cmd": "mode", "value": "wake"})
        assert ws.receive_json()["data"]["mode"] == "wake"
        ws.send_json({"cmd": "persona", "value": "pokaz"})
        assert ws.receive_json()["data"]["persona"] == "pokaz"
        ws.send_json({"cmd": "barge_in", "enabled": True, "threshold": 0.4})
        assert ws.receive_json()["data"]["barge_in"] == {"enabled": True, "threshold": 0.4}
    assert (app.config.conversation.mode, app.config.conversation.persona) == ("wake", "pokaz")
    assert app.config.barge_in.enabled and app.config.barge_in.threshold == 0.4
    assert app.calls == ["mode:wake", "new"]  # persona switch starts a fresh conversation


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
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json(msg)
        sync(ws, app)
    assert app.config.conversation.mode == "ptt" and app.config.conversation.persona == "dzieci"
    assert not app.config.barge_in.enabled and app.config.barge_in.threshold == 0.15
    assert app.calls == []


def test_malformed_json_keeps_socket_open(client, app):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_text("{not json")
        ws.send_json({"cmd": "stop"})
        sync(ws, app)
    assert app.calls == ["stop"]


def test_doctor_results_are_published_as_event(client, app):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json({"cmd": "doctor"})
        msg = ws.receive_json()
    assert msg["type"] == "doctor"
    assert msg["data"]["results"] == [
        {"name": "Klucz", "ok": True, "detail": "jest", "hint": ""},
        {"name": "Mikrofon", "ok": False, "detail": "brak", "hint": "podłącz"},
    ]


def test_disconnect_unsubscribes(client, app):
    with connect(client) as ws:
        snapshot(ws)
        assert len(app.bus._subs) == 1
    client.get("/")  # lets the server side finish the handler
    assert app.bus._subs == []


def test_server_binds_localhost_only_on_given_port(app):
    c = server_config(create_app(app), 8799)
    assert (c.host, c.port) == ("127.0.0.1", 8799)
    assert server_config(create_app(app)).port == DEFAULT_PORT == 8765


def test_foreign_missing_and_rebinding_origins_are_rejected(client):
    from starlette.websockets import WebSocketDisconnect

    for headers in ({"origin": "http://evil.example"}, {}, {"origin": "http://testserver:9999"}, {"origin": "null"}):
        with pytest.raises(WebSocketDisconnect) as e, connect(client, headers):
            pass
        assert e.value.code == 1008


def test_unlisted_host_is_rejected_even_with_matching_origin(app):
    from starlette.websockets import WebSocketDisconnect

    c = TestClient(create_app(app, doctor=fake_doctor))  # default hosts: localhost only
    with pytest.raises(WebSocketDisconnect), connect(c):
        pass


def test_same_origin_is_accepted(client):
    with connect(client) as ws:
        assert ws.receive_json()["type"] == "snapshot"


def test_shutdown_closes_the_app(app):
    with TestClient(create_app(app, doctor=fake_doctor, allowed_hosts={"testserver"})):
        assert app.closed == 0
    assert app.closed == 1


def test_disconnect_while_recording_discards_audio(client, app):
    app.listener.recording = True
    with connect(client) as ws:
        snapshot(ws)
    client.get("/")
    assert app.calls == ["ptt_abort"]


def test_doctor_is_ignored_while_a_run_is_pending(app):
    started = []

    async def slow(app):
        started.append(1)
        await asyncio.sleep(0.3)
        return []

    c = TestClient(create_app(app, doctor=slow, allowed_hosts={"testserver"}))
    with connect(c) as ws:
        snapshot(ws)
        ws.send_json({"cmd": "doctor"})
        ws.send_json({"cmd": "doctor"})
        assert ws.receive_json()["type"] == "doctor"
    assert started == [1]


async def test_web_doctor_never_touches_the_robot_driver(monkeypatch):
    from ohbotchat import doctor, web

    touched = []

    class Spy:
        def __init__(self):
            touched.append("created")

        def connect(self):
            touched.append("connect")
            return True

        def close(self):
            touched.append("close")

    class Quiet:
        async def play(self, pcm, *a, **k):
            return True

    monkeypatch.setattr(doctor, "OhbotDriver", Spy)
    monkeypatch.setattr("ohbotchat.speak.Speaker", Quiet)
    app = FakeApp()
    app.config.api_key = None
    results = await web._run_doctor(app)
    assert touched == [] and next(r for r in results if r.name == "Ohbot").ok is True
